"""Sigmoid-Choquet aggregator for DRACO.

Ported from FuzRL (Wan et al., NeurIPS 2025) with the B1 fix applied:
    - sigmoid activation (not softmax) so capacity g is genuinely non-additive
    - non-additivity regularizer pulling sum(g) away from 1

Sugeno consistency:
    prod_{k=1..K} (1 + lambda * g_k) = 1 + lambda
    sum(g)=1 -> lambda=0 (additive, Choquet collapses to weighted mean)
    sum(g)>1 -> lambda<0 (sub-additive)
    sum(g)<1 -> lambda>0 (super-additive)

Discrete Choquet integral with descending sort:
    (C) integral V dm = sum_{i=1..K} v_(i) * (m(A_i) - m(A_{i+1}))
where A_i = {(i),...,(K)} are tail sets after sorting v in descending order.

For DRACO, we use `choquet_integral_dual` (i.e. CVaR-style aggregation
that emphasizes the maximum source) since high cost = bad.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _solve_lambda(g: torch.Tensor, max_iter: int = 50, tol: float = 1e-7) -> torch.Tensor:
    """Newton-Raphson solver for Sugeno consistency. Detached output: gradients
    flow into g only through the explicit Choquet formula, not through this loop.
    """
    B, K = g.shape
    sum_g = g.sum(dim=1, keepdim=True)

    additive_mask = (sum_g - 1.0).abs() < 1e-3
    sub_mask = sum_g > 1.0 + 1e-3

    lam = torch.where(
        additive_mask, torch.zeros_like(sum_g),
        torch.where(sub_mask, torch.full_like(sum_g, -0.5),
                    torch.full_like(sum_g, 0.5)),
    )
    for _ in range(max_iter):
        terms = (1.0 + lam * g).clamp(min=1e-8)
        log_terms = torch.log(terms)
        sum_log = log_terms.sum(dim=1, keepdim=True)
        prod = torch.exp(sum_log)
        f = prod - 1.0 - lam
        f_prime = (g * torch.exp(sum_log.expand_as(g) - log_terms)).sum(dim=1, keepdim=True) - 1.0
        delta = f / (f_prime + 1e-10)
        lam = (lam - delta).clamp(min=-0.99, max=10.0)
        if torch.abs(delta).max().item() < tol:
            break

    lam = torch.where(additive_mask, torch.zeros_like(lam), lam)
    return lam.squeeze(-1).detach()


class ChoquetAggregator(nn.Module):
    """State-conditional capacity g_k(s) for K sources, sigmoid + reg.

    Args:
        state_dim: state dimension.
        n_sources: K (number of perturbation shells / risk sources).
        hidden_dim: capacity network hidden dim.
        nonadditive_target_per_dim:
            target average value of g_k. With K=4 and target=0.3, we want
            sum(g) ~= 1.2, so lambda < 0 (sub-additive: emphasizes worst sources).
    """

    def __init__(
        self,
        state_dim: int,
        n_sources: int,
        hidden_dim: int = 32,
        clamp_eps: float = 1e-4,
        nonadditive_target_per_dim: float = 0.3,
    ):
        super().__init__()
        self.K = n_sources
        self.clamp_eps = clamp_eps
        self.nonadditive_target = nonadditive_target_per_dim * n_sources
        self.target_per_dim = nonadditive_target_per_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sources),
        )
        # Init final-layer bias so sigmoid(bias) == target_per_dim
        with torch.no_grad():
            init_bias = float(torch.logit(torch.tensor(nonadditive_target_per_dim)))
            self.net[-1].bias.fill_(init_bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) -> g: (B, K), each g_k in (eps, 1-eps)."""
        logits = self.net(state)
        g = torch.sigmoid(logits)
        return g.clamp(self.clamp_eps, 1.0 - self.clamp_eps)

    def nonadditive_regularizer(self, g: torch.Tensor) -> torch.Tensor:
        """Soft penalty: (sum(g) - target)^2, pushes capacity away from additive."""
        sum_g = g.sum(dim=1)
        return ((sum_g - self.nonadditive_target) ** 2).mean()


def _measure_of_tail(g: torch.Tensor, lam: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    """Compute m(A_i) for tail sets A_i = {(i),...,(K)} after sorting.
    Returns m_tail of shape (B, K+1) with m_tail[:,0]=m(X), m_tail[:,K]=0.
    """
    B, K = g.shape
    g_sorted = torch.gather(g, 1, sort_idx)
    lam_b = lam.unsqueeze(1)

    factors = 1.0 + lam_b * g_sorted
    factors_rev = torch.flip(factors, dims=[1])
    cum_rev = torch.cumprod(factors_rev, dim=1)
    cum_right = torch.flip(cum_rev, dims=[1])
    cum_right = torch.cat([cum_right, torch.ones(B, 1, device=g.device, dtype=g.dtype)], dim=1)
    m_nonzero = (cum_right - 1.0) / (lam_b + 1e-12 * (lam_b == 0).to(lam_b.dtype))

    g_rev = torch.flip(g_sorted, dims=[1])
    cum_g = torch.cumsum(g_rev, dim=1)
    cum_g = torch.flip(cum_g, dims=[1])
    cum_g = torch.cat([cum_g, torch.zeros(B, 1, device=g.device, dtype=g.dtype)], dim=1)

    is_zero_lam = lam_b.abs() < 1e-6
    m_tail = torch.where(is_zero_lam, cum_g, m_nonzero)
    m_tail = torch.cat([m_tail[:, :K], torch.zeros(B, 1, device=g.device, dtype=g.dtype)], dim=1)
    return m_tail


def choquet_integral(g: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Choquet integral with descending sort (rewards interpretation).

    Args:
        g: (B, K) capacity values per source.
        values: (B, K) source values to aggregate.
    Returns:
        (B, 1) aggregated value.
    """
    B, K = values.shape
    lam = _solve_lambda(g)
    v_sorted, sort_idx = torch.sort(values, dim=1, descending=True)
    m_tail = _measure_of_tail(g, lam, sort_idx)
    weights = m_tail[:, :K] - m_tail[:, 1:K + 1]
    return (v_sorted * weights).sum(dim=1, keepdim=True)


def choquet_integral_dual(g: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Dual (cost) Choquet: emphasizes the maximum sources.

    For DRACO V_C^robust aggregation we use this version (high CVaR = bad).
    """
    return -choquet_integral(g, -values)
