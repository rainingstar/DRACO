"""Multi-source GPD heads for DRACO.

Each of K perturbation shells maintains a GPD head that estimates the
shape (xi) and scale (beta) parameters of the Generalized Pareto Distribution
fitted to the tail of Z_C(s'_k, tau) for tau > eta.

Math (Pickands-Balkema-de Haan):
    For Y > u with u high enough,
        P(Y - u > y | Y > u) ~ GPD(y; xi, beta)
        f(y) = (1/beta)(1 + xi*y/beta)^(-1/xi - 1)

Closed-form CVaR_eta of a GPD-tailed distribution above threshold u:
    CVaR_eta(Y) = u + beta/(1-xi) + xi*(u - mu_low)/(1-xi)
        when xi < 1; for xi >= 1, mean is undefined (we clamp xi <= 0.99).

For the safe-RL use case, we approximate:
    CVaR_eta(Z_C | s'_k) ~= u_k + beta_k/(1-xi_k)
where u_k is the data-driven threshold (eta-quantile of IQN samples at s'_k).
This is the standard Peaks-Over-Threshold (POT) CVaR estimator.

Architecture:
    Per shell k: state encoder + (xi_k, beta_k) head.
    Output transforms:  xi  via tanh -> [-0.5, 0.99] (avoid heavy-tail blow-up)
                        beta via softplus + small constant (positive)

We support two modes:
    (a) per-shell separate networks (K independent GPDHeadSubnet)
    (b) shared backbone + K heads + shell embedding   (parameter efficient)
We default to (b) for K>=4 (saves params, batches well).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else (output_activation or nn.Identity)
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MultiSourceGPDHead(nn.Module):
    """K GPD heads sharing a state encoder, parameterized by shell embedding.

    Args:
        obs_dim: state dim.
        n_shells: K (number of perturbation shells / risk sources).
        shell_emb_dim: embedding dim for shell index.
        hidden_sizes: encoder hidden sizes.
        xi_max: maximum allowed xi (we clamp away from 1.0 since CVaR diverges).
                Default 0.95.
        xi_min: minimum allowed xi. Default -0.5 (mild left-bound, light tails).
        beta_min: minimum scale (additive offset). Default 1e-3.
    """

    def __init__(
        self,
        obs_dim: int,
        n_shells: int = 4,
        shell_emb_dim: int = 8,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        xi_max: float = 0.95,
        xi_min: float = -0.5,
        beta_min: float = 1e-3,
    ):
        super().__init__()
        self.n_shells = n_shells
        self.xi_max = xi_max
        self.xi_min = xi_min
        self.beta_min = beta_min

        self.shell_emb = nn.Embedding(n_shells, shell_emb_dim)
        encoder_in = obs_dim + shell_emb_dim
        self.encoder = _mlp(
            [encoder_in] + list(hidden_sizes),
            activation=nn.ReLU,
        )
        # Two heads: xi (raw) and beta (raw, will softplus)
        self.xi_head = nn.Linear(hidden_sizes[-1], 1)
        self.beta_head = nn.Linear(hidden_sizes[-1], 1)
        # Init beta head bias slightly positive so initial scale ~1
        nn.init.constant_(self.beta_head.bias, 0.5)

    def forward(
        self,
        obs: torch.Tensor,           # (B, obs_dim)
        shell_idx: torch.Tensor,     # (B,) long, in [0, K-1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (xi, beta) for each (state, shell) pair.

        xi:   (B,) in [xi_min, xi_max]
        beta: (B,) > beta_min
        """
        emb = self.shell_emb(shell_idx)  # (B, shell_emb_dim)
        x = torch.cat([obs, emb], dim=-1)
        h = self.encoder(x)
        xi_raw = self.xi_head(h).squeeze(-1)
        beta_raw = self.beta_head(h).squeeze(-1)
        # transforms
        xi_range = self.xi_max - self.xi_min
        xi = self.xi_min + xi_range * torch.sigmoid(xi_raw)
        beta = F.softplus(beta_raw) + self.beta_min
        return xi, beta

    def forward_all_shells(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all K shells in one batched call.

        Args:
            obs: (B, obs_dim)
        Returns:
            xi:   (B, K)
            beta: (B, K)
        """
        B = obs.shape[0]
        K = self.n_shells
        # Expand obs to (B*K, obs_dim) and shell_idx to (B*K,)
        obs_exp = obs.unsqueeze(1).expand(B, K, obs.shape[-1]).reshape(B * K, -1)
        shell_idx = torch.arange(K, device=obs.device).repeat(B)
        xi, beta = self.forward(obs_exp, shell_idx)
        return xi.view(B, K), beta.view(B, K)


# ============================================================================
# GPD likelihood + CVaR closed forms
# ============================================================================

def gpd_log_likelihood(
    excesses: torch.Tensor,   # (B, n_tail)  values y = z - u  >= 0
    xi: torch.Tensor,         # (B,)
    beta: torch.Tensor,       # (B,)
    mask: Optional[torch.Tensor] = None,  # (B, n_tail) 1 where excess valid
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-state log-likelihood of GPD(xi, beta).

    For y >= 0 and xi != 0:
        log f(y) = -log(beta) - (1/xi + 1) log(1 + xi*y/beta)
    For xi -> 0 (exponential limit):
        log f(y) = -log(beta) - y/beta

    Returns: (B,) log-likelihood (sum over tail samples per state, normalized
             by valid count).
    """
    B = excesses.shape[0]
    if mask is None:
        mask = torch.ones_like(excesses)

    xi_b = xi.unsqueeze(-1).clamp(min=-0.49, max=0.95)
    beta_b = beta.unsqueeze(-1).clamp(min=eps)
    y = excesses.clamp(min=0.0)

    # Nudge xi away from zero with a STRAIGHT-THROUGH estimator. torch.where
    # evaluates both branches (NaN), and a hard clamp on |xi| zeros out
    # gradients near 0. STE: forward uses the safe value, backward uses the
    # raw xi gradient. This lets xi cross 0 during training.
    sign = torch.where(xi_b >= 0, torch.ones_like(xi_b), -torch.ones_like(xi_b))
    xi_clipped = sign * xi_b.abs().clamp(min=1e-3)
    xi_safe = xi_b + (xi_clipped - xi_b).detach()
    arg = 1.0 + xi_safe * y / beta_b
    arg = arg.clamp(min=eps)
    log_f = -torch.log(beta_b) - (1.0 / xi_safe + 1.0) * torch.log(arg)
    # Mask invalid entries
    log_f = log_f * mask
    n_valid = mask.sum(dim=-1).clamp(min=1.0)
    return log_f.sum(dim=-1) / n_valid


def gpd_cvar(
    threshold_u: torch.Tensor,   # (B,) the eta-quantile of Z_C
    xi: torch.Tensor,            # (B,)
    beta: torch.Tensor,           # (B,)
) -> torch.Tensor:
    """Closed-form CVaR for POT-fitted GPD.

    CVaR_eta(Z) = u + beta/(1-xi) + xi*u/(1-xi)*0   (POT version)

    Standard POT-CVaR (above threshold u, conditional expectation):
        E[Z | Z > u] = u + beta/(1-xi)   for xi < 1

    Returns (B,) CVaR estimate per state.
    """
    xi_safe = xi.clamp(max=0.95)  # avoid divergence at xi -> 1
    cvar = threshold_u + beta / (1.0 - xi_safe)
    return cvar


def fit_threshold_and_excesses(
    z_samples: torch.Tensor,   # (B, N) IQN quantile samples
    tau: torch.Tensor,         # (B, N) corresponding tau in [0,1]
    eta: float = 0.7,
    min_tail: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-state POT threshold u and excesses y = z - u for tau > eta.

    Returns:
        u:        (B,) threshold (data-driven eta-quantile estimate from IQN).
        excesses: (B, n_tail_max) padded with 0; mask says which are valid.
        mask:     (B, n_tail_max) 1 where the excess is real (tau > eta).

    Notes:
        - We use IQN samples as the empirical estimator of the conditional
          quantile function. u_hat = sample mean of z[tau in (eta-delta, eta+delta)]
          would be more accurate, but for simplicity we just take min(z[tau>eta]).
        - If too few tail samples, we use the full sample range (fallback).
    """
    B, N = z_samples.shape
    mask = (tau > eta).float()  # (B, N) 1 where tau > eta
    n_tail = mask.sum(dim=-1)   # (B,)

    # Threshold u: per-state min of z over tau > eta region (i.e. z at tau ~= eta)
    # Use a high penalty to push masked entries up so min ignores them.
    z_masked = z_samples + (1 - mask) * 1e9
    u, _ = z_masked.min(dim=-1)  # (B,)
    # If a state has < min_tail tail samples, fall back to mean over all samples
    fallback_u = z_samples.mean(dim=-1)
    u = torch.where(n_tail >= min_tail, u, fallback_u)

    # Excesses = z - u, kept only where tau > eta
    excesses = (z_samples - u.unsqueeze(-1)).clamp(min=0.0) * mask
    return u, excesses, mask
