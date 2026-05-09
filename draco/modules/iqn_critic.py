"""IQN-based distributional cost critic for DRACO.

Implements the Implicit Quantile Network (Dabney et al., 2018, ICML).

For an input state s and quantile fraction tau in [0,1], the network outputs
Z_C(s, tau) which approximates F_C^{-1}(tau | s), i.e. the tau-th quantile of
the cumulative cost-to-go distribution given state s.

Architecture:
    state encoder (MLP) -> phi(s)  in R^d_state
    tau cosine basis           -> psi(tau) in R^n_cos
    merge: phi(s) * (W psi(tau) + b)        (Hadamard product, IQN paper)
    head MLP                  -> Z_C(s, tau) scalar

Loss:
    quantile Huber regression. For target sample z* with quantile tau' and
    prediction Z = Z_C(s, tau):
        rho_kappa^tau(d) = |tau - 1{d<0}| * Huber_kappa(d)
    where d = z* - Z. Average over (tau, tau') pairs.

Notes:
    - We sample N_tau quantiles per forward; N_target_tau quantiles per target.
    - Risk-neutral expectation E[Z_C(s)] = mean over many tau samples.
    - Tail estimate (used by GPD heads): take samples with tau > eta and either
      use empirical mean (CVaR_eta) or fit GPD on (z - threshold).
"""
from __future__ import annotations
import math
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


class IQNCostCritic(nn.Module):
    """Implicit Quantile Network cost critic.

    Args:
        obs_dim: state dimension.
        hidden_sizes: tuple of MLP hidden layer sizes.
        n_cos: number of cosine basis functions for tau embedding (paper uses 64).
        activation: activation class.
        nonneg_output: if True, apply softplus on output (cumulative cost is >= 0
                       in safe RL with non-negative cost). Default True.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        n_cos: int = 64,
        activation: type = nn.ReLU,
        nonneg_output: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_cos = n_cos
        self.nonneg_output = nonneg_output

        # State encoder: obs -> phi(s) in R^{hidden_sizes[0]}
        self.state_encoder = _mlp(
            [obs_dim, hidden_sizes[0], hidden_sizes[0]],
            activation=activation,
        )
        feature_dim = hidden_sizes[0]

        # tau embedding: psi(tau) = cos(pi * i * tau) for i = 1..n_cos.
        # Then linear projection psi -> R^{feature_dim}.
        # Buffer of i values [1, 2, ..., n_cos], shape (1, 1, n_cos)
        self.register_buffer(
            "cos_basis_idx",
            torch.arange(1, n_cos + 1, dtype=torch.float32).view(1, 1, n_cos) * math.pi,
        )
        self.tau_proj = nn.Linear(n_cos, feature_dim)

        # Head: feature_dim -> 1 scalar Z_C(s, tau)
        head_sizes = [feature_dim] + list(hidden_sizes[1:]) + [1]
        self.head = _mlp(head_sizes, activation=activation)

    def forward(self, obs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Compute Z_C(s, tau).

        Args:
            obs: (B, obs_dim) tensor of states.
            tau: (B, N) tensor of quantile fractions in [0, 1].

        Returns:
            z: (B, N) tensor of quantile values.
        """
        B, N = tau.shape
        # State features: (B, F)
        phi = self.state_encoder(obs)
        F_dim = phi.shape[-1]

        # tau cosine embedding: (B, N, n_cos)
        tau_3d = tau.unsqueeze(-1)  # (B, N, 1)
        cos_emb = torch.cos(tau_3d * self.cos_basis_idx)  # (B, N, n_cos)
        psi = F.relu(self.tau_proj(cos_emb))  # (B, N, F)

        # Hadamard merge: (B, 1, F) * (B, N, F) -> (B, N, F)
        merged = phi.unsqueeze(1) * psi
        # Apply head: flatten to (B*N, F) -> (B*N, 1) -> (B, N)
        z = self.head(merged.view(B * N, F_dim)).view(B, N)
        if self.nonneg_output:
            z = F.softplus(z)
        return z

    @torch.no_grad()
    def value(self, obs: torch.Tensor, n_tau: int = 64) -> torch.Tensor:
        """Risk-neutral expectation E[Z_C(s)] = mean over uniform tau samples.

        Used as a fallback / sanity check. The 'risk-aware' version is
        computed via multi_source_gpd.compute_robust_value.
        """
        B = obs.shape[0]
        tau = torch.rand(B, n_tau, device=obs.device)
        z = self.forward(obs, tau)
        return z.mean(dim=-1)

    def sample_quantiles(
        self,
        obs: torch.Tensor,
        n_tau: int,
        tau_min: float = 0.0,
        tau_max: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tau uniformly in [tau_min, tau_max] and return (tau, Z(s,tau))."""
        B = obs.shape[0]
        tau = torch.rand(B, n_tau, device=obs.device) * (tau_max - tau_min) + tau_min
        z = self.forward(obs, tau)
        return tau, z


def quantile_huber_loss(
    pred_z: torch.Tensor,        # (B, N) predicted quantile values at tau
    tau: torch.Tensor,           # (B, N) corresponding tau fractions
    target_z: torch.Tensor,      # (B, M) target quantile samples (no grad)
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss for IQN training.

    L = mean over (i, j) of [|tau_i - 1{td_ij < 0}| * H_kappa(td_ij)]
    where td_ij = target_z[..., j] - pred_z[..., i].

    Returns a scalar loss (per-sample mean).
    """
    # td: (B, N, M)
    td = target_z.unsqueeze(1) - pred_z.unsqueeze(2)  # broadcast pred over M
    # Huber
    abs_td = td.abs()
    huber = torch.where(
        abs_td <= kappa,
        0.5 * td.pow(2),
        kappa * (abs_td - 0.5 * kappa),
    )
    # Asymmetric weight |tau - 1{td<0}|
    tau_3d = tau.unsqueeze(2)  # (B, N, 1)
    weight = (tau_3d - (td.detach() < 0).float()).abs()
    loss = (weight * huber / kappa).sum(dim=2).mean(dim=1)  # avg over M, sum over N
    return loss.mean()


def empirical_cvar_from_iqn(
    z: torch.Tensor,    # (B, N) quantile samples
    tau: torch.Tensor,  # (B, N) corresponding tau
    eta: float = 0.7,
) -> torch.Tensor:
    """Empirical CVaR_{eta}(s) = E[Z_C(s) | tau > eta].

    Used by Dist-Only baseline (no GPD extrapolation). Differentiable through
    z (the IQN output) but not through tau (it's a sampled mask).
    """
    mask = (tau > eta).float()  # (B, N)
    # avoid div-by-zero
    denom = mask.sum(dim=-1).clamp(min=1.0)
    cvar = (z * mask).sum(dim=-1) / denom
    return cvar
