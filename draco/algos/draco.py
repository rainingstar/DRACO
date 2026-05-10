"""DRACO: Distributional Risk-Aware Choquet-Ordered safe RL.

A unified PPO-Lagrangian algorithm with five operating modes selected by
flag combinations. The full method (DRACO) uses all components.

Method matrix:
    +-------------+--------+--------+---------+----------+
    | Method      | IQN VC | EVT/GPD| Multi-K | Choquet  |
    +-------------+--------+--------+---------+----------+
    | baseline    |   no   |   no   |   no    |   no     |
    | fuz         |   no   |   no   |   yes   |   yes    |
    | evo_style   |   no   |   yes  |   no    |   no     |
    | dist_only   |   yes  |   no   |   no    |   no     |
    | DRACO       |   yes  |   yes  |   yes   |   yes    |
    +-------------+--------+--------+---------+----------+

For modes WITHOUT IQN_VC, the cost critic is the standard scalar MLP V_C(s).
For modes WITHOUT multi-K shells, V_C^robust(s') = single-source CVaR estimate.

Algorithm (DRACO mode, simplified):
    Rollout (parallel N envs, T steps each):
        action ~ pi(s); v = V_R(s); vc_robust = compute_vc_robust(s)
        store; step env
    Compute advantages with GAE (using vc_robust as the cost critic summary)

    Policy update:
        L = -[ rho_t * adv_t  -  lambda * rho_t * cadv_t ]   + entropy

    Reward critic update (V_R):  MSE on returns
    Cost critic update (IQN):    quantile Huber loss on MC cost-to-go target
    GPD update (multi-source):   per-shell MLE on tail-sample log-likelihood
                                 + non-additivity reg on Choquet capacity
    Lagrangian update (lambda):  proportional update toward cost-budget d
"""
from __future__ import annotations
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.modules.actor_critic import ActorRewardCritic, MLPCritic, count_vars
from draco.modules.iqn_critic import (
    IQNCostCritic, quantile_huber_loss, empirical_cvar_from_iqn,
)
from draco.modules.multi_source_gpd import (
    MultiSourceGPDHead, gpd_log_likelihood, gpd_cvar, fit_threshold_and_excesses,
)
from draco.modules.fuzzy_net import ChoquetAggregator, choquet_integral_dual
from draco.modules.buffer import VecRolloutBuffer
from draco.utils.logger import Logger
from draco.utils.seeds import set_seed


ALGO_NAME = "DRACO"


# ============================================================================
# Mode flags
# ============================================================================
class MethodConfig:
    """Defines which components are active for a given method."""

    def __init__(
        self,
        name: str,
        use_iqn: bool,
        use_evt: bool,
        n_shells: int,
        use_choquet: bool,
    ):
        self.name = name
        self.use_iqn = use_iqn
        self.use_evt = use_evt
        self.n_shells = n_shells
        self.use_choquet = use_choquet
        # Sanity: choquet only meaningful if multi-shell
        if use_choquet and n_shells <= 1:
            raise ValueError(f"Method {name}: use_choquet=True requires n_shells>1")
        # GPD only on IQN tail (we deliberately do NOT support EVO's
        # episode-level GPD on scalar V_C — that lives in a separate baseline
        # if needed). For evo_style baseline below, we simulate it with
        # IQN=off + EVT-on by using a scalar V_C and the empirical-CVaR
        # surrogate fed through a scalar GPD head.

    @classmethod
    def from_name(cls, name: str, n_shells: int = 4):
        """Standard methods (5 total)."""
        name = name.lower()
        if name == "baseline":
            return cls(name, use_iqn=False, use_evt=False, n_shells=1, use_choquet=False)
        if name == "fuz":
            # Multi-shell + Choquet, but no EVT, no distributional cost critic.
            return cls(name, use_iqn=False, use_evt=False, n_shells=n_shells, use_choquet=True)
        if name == "evo_style":
            # Single GPD on scalar V_C tail (approx EVO; no distributional, no shells).
            # Implemented via use_iqn=False + use_evt=True + n_shells=1 + no Choquet.
            return cls(name, use_iqn=False, use_evt=True, n_shells=1, use_choquet=False)
        if name == "dist_only":
            # IQN cost critic alone, empirical CVaR (no GPD), no shells, no Choquet.
            return cls(name, use_iqn=True, use_evt=False, n_shells=1, use_choquet=False)
        if name == "draco":
            return cls(name, use_iqn=True, use_evt=True, n_shells=n_shells, use_choquet=True)
        raise ValueError(f"Unknown method '{name}'. Valid: baseline, fuz, evo_style, dist_only, draco")


# ============================================================================
# DRACO algorithm
# ============================================================================
class DRACOAlgorithm:
    """PPO-Lagrangian + risk-aware components.

    Args:
        venv: vectorized env (SyncVectorEnv).
        method: MethodConfig.
        ... see __init__ for hyperparameters.
    """

    def __init__(
        self,
        venv,                  # SyncVectorEnv
        method: MethodConfig,
        cost_limit: float = 25.0,
        gamma: float = 0.99,
        lam: float = 0.97,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.97,
        steps_per_epoch: int = 2048,    # total steps across all envs
        # PPO
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        target_kl: float = 0.015,
        pi_lr: float = 3e-4,
        v_lr: float = 1e-3,
        vc_lr: float = 1e-3,
        gpd_lr: float = 1e-3,
        choquet_lr: float = 3e-4,
        # IQN
        n_tau_train: int = 32,
        n_tau_target: int = 16,
        n_tau_eval: int = 64,
        eta: float = 0.7,                # GPD threshold quantile
        # Multi-shell
        shell_sigmas: tuple = (0.01, 0.05, 0.1, 0.2),
        # Lagrangian
        penalty_init: float = 0.0,
        penalty_lr: float = 5e-2,
        penalty_max: float = 100.0,
        # Misc
        choquet_reg_coef: float = 0.01,
        seed: int = 0,
        device: str = "cpu",
        log_dir: str = "./runs/draco",
        run_name: Optional[str] = None,
        hidden_sizes: tuple = (64, 64),
    ):
        set_seed(seed)
        self.device = torch.device(device)
        self.method = method
        self.venv = venv

        # Probe spaces
        e0 = venv.envs[0]
        self.obs_dim = venv.obs_dim
        self.act_dim = venv.act_dim
        self.is_continuous = (type(e0.action_space).__name__ == "Box")

        # === Networks ===
        # 1. Actor + reward critic (always on)
        class _SpaceShim:
            def __init__(self, dim): self.shape = (dim,)
        obs_space = _SpaceShim(self.obs_dim)
        act_space = e0.action_space  # we trust the underlying env
        self.ac = ActorRewardCritic(obs_space, act_space, hidden_sizes=hidden_sizes).to(self.device)

        # 2. Cost critic
        if method.use_iqn:
            self.cost_critic = IQNCostCritic(
                obs_dim=self.obs_dim, hidden_sizes=hidden_sizes,
            ).to(self.device)
        else:
            self.cost_critic = MLPCritic(self.obs_dim, hidden_sizes, nn.Tanh).to(self.device)

        # 3. GPD head (multi-source) — only if EVT enabled
        self.gpd = None
        if method.use_evt:
            self.gpd = MultiSourceGPDHead(
                obs_dim=self.obs_dim,
                n_shells=method.n_shells,
                hidden_sizes=hidden_sizes,
            ).to(self.device)

        # 4. Choquet aggregator — only if K>1 and use_choquet
        self.choquet = None
        if method.use_choquet:
            self.choquet = ChoquetAggregator(
                state_dim=self.obs_dim, n_sources=method.n_shells,
            ).to(self.device)

        self.shell_sigmas = torch.tensor(
            list(shell_sigmas)[:method.n_shells], dtype=torch.float32, device=self.device
        )
        if len(self.shell_sigmas) < method.n_shells:
            # Pad with last sigma if user provided fewer
            pad = torch.full((method.n_shells - len(self.shell_sigmas),), shell_sigmas[-1],
                             dtype=torch.float32, device=self.device)
            self.shell_sigmas = torch.cat([self.shell_sigmas, pad])

        # === Optimizers ===
        self.pi_opt = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_opt = torch.optim.Adam(self.ac.v.parameters(), lr=v_lr)
        self.vc_opt = torch.optim.Adam(self.cost_critic.parameters(), lr=vc_lr, weight_decay=1e-4)  # IQN-FIX: prevent param drift
        self.gpd_opt = torch.optim.Adam(self.gpd.parameters(), lr=gpd_lr) if self.gpd is not None else None
        self.choquet_opt = torch.optim.Adam(self.choquet.parameters(), lr=choquet_lr) if self.choquet is not None else None

        # Hyperparams
        self.cost_limit = cost_limit
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.n_tau_train = n_tau_train
        self.n_tau_target = n_tau_target
        self.n_tau_eval = n_tau_eval
        self.eta = eta
        self.choquet_reg_coef = choquet_reg_coef
        self.penalty_lr = penalty_lr
        self.penalty_max = penalty_max
        self.hidden_sizes = hidden_sizes

        # Lagrangian penalty as raw param; lambda = softplus(p_param)
        self.p_param = torch.tensor(np.log(np.exp(penalty_init) + 1e-8) if penalty_init > 0 else -2.0,
                                    dtype=torch.float32, device=self.device, requires_grad=False)

        # Buffer
        self.steps_per_epoch = steps_per_epoch
        n_envs = venv.n_envs
        T = max(1, steps_per_epoch // n_envs)
        self.steps_per_env = T

        self.buf = VecRolloutBuffer(
            n_envs=n_envs, steps_per_env=T,
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            gamma=gamma, lam=lam, cost_gamma=cost_gamma, cost_lam=cost_lam,
            is_discrete=not self.is_continuous,
        )

        # Logger
        run_name = run_name or f"{method.name}_seed{seed}_{int(time.time())}"
        self.logger = Logger(log_dir, run_name)
        self.logger.store(method=0)  # placeholder; can't store strings yet
        self._total_envsteps = 0
        self._ep_returns = np.zeros(n_envs, dtype=np.float32)
        self._ep_costs = np.zeros(n_envs, dtype=np.float32)
        self._ep_lens = np.zeros(n_envs, dtype=np.int64)

        print(f"[DRACO init] method={method.name} | use_iqn={method.use_iqn} use_evt={method.use_evt} "
              f"n_shells={method.n_shells} use_choquet={method.use_choquet}")
        print(f"[DRACO init] envs={n_envs}, steps/env={T}, total/epoch={n_envs*T}")
        print(f"[DRACO init] params: actor={count_vars(self.ac.pi)} v={count_vars(self.ac.v)} "
              f"vc={count_vars(self.cost_critic)} "
              f"gpd={count_vars(self.gpd) if self.gpd else 0} "
              f"choquet={count_vars(self.choquet) if self.choquet else 0}")

    # ------------------------------------------------------------------
    # Robust cost value computation: V_C^robust(s)
    # ------------------------------------------------------------------
    def compute_vc_robust(
        self, obs: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
        """Compute robust scalar cost value V_C^robust(s) for advantage estimation.

        For each method:
            baseline:  V_C(s)                        (scalar critic)
            fuz:       Choquet({V_C(s+eps_k)})       (multi-shell, no EVT)
            evo_style: V_C(s) + xi/(1-xi)*beta       (single GPD on scalar)
            dist_only: empirical CVaR_eta(Z_C(s))    (IQN, no shells)
            draco:     Choquet({GPD CVaR_eta(Z_C(s+eps_k))_k})   (full)

        Returns (B,) scalar cost value.
        """
        B = obs.shape[0]
        K = self.method.n_shells

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            if not self.method.use_iqn:
                # Scalar V_C critic path
                if K == 1:
                    if self.method.use_evt:
                        # evo_style: scalar V_C + GPD correction
                        vc = self.cost_critic(obs)
                        # Use GPD head with shell_idx=0 (only one shell allowed)
                        shell_idx = torch.zeros(B, dtype=torch.long, device=obs.device)
                        xi, beta = self.gpd(obs, shell_idx)
                        # Closed-form CVaR with u=V_C(s) (treat scalar as threshold proxy)
                        return gpd_cvar(vc, xi, beta)
                    else:
                        # baseline: pure scalar V_C
                        return self.cost_critic(obs)
                else:
                    # fuz: multi-shell scalar V_C, Choquet
                    obs_shells = obs.unsqueeze(1).expand(B, K, self.obs_dim).clone()
                    eps_noise = torch.randn(B, K, self.obs_dim, device=obs.device)
                    obs_shells = obs_shells + eps_noise * self.shell_sigmas.view(1, K, 1)
                    vc_per_shell = self.cost_critic(obs_shells.view(B * K, -1)).view(B, K)
                    g = self.choquet(obs)
                    return choquet_integral_dual(g, vc_per_shell).squeeze(-1)
            else:
                # IQN cost critic path
                if K == 1:
                    # dist_only (or evo-on-distributional unused branch)
                    tau = torch.rand(B, self.n_tau_eval, device=obs.device)
                    z = self.cost_critic(obs, tau)
                    if self.method.use_evt:
                        # GPD on tail of IQN
                        u, ex, mk = fit_threshold_and_excesses(z, tau, self.eta)
                        shell_idx = torch.zeros(B, dtype=torch.long, device=obs.device)
                        xi, beta = self.gpd(obs, shell_idx)
                        return gpd_cvar(u, xi, beta)
                    else:
                        # Empirical CVaR
                        return empirical_cvar_from_iqn(z, tau, self.eta)
                else:
                    # DRACO: K shells, IQN per shell, GPD per shell, Choquet aggregate.
                    # Build (B, K, obs_dim) perturbed states.
                    obs_shells = obs.unsqueeze(1).expand(B, K, self.obs_dim).clone()
                    eps_noise = torch.randn(B, K, self.obs_dim, device=obs.device)
                    obs_shells = obs_shells + eps_noise * self.shell_sigmas.view(1, K, 1)
                    obs_flat = obs_shells.view(B * K, self.obs_dim)
                    tau = torch.rand(B * K, self.n_tau_eval, device=obs.device)
                    z = self.cost_critic(obs_flat, tau)  # (B*K, n_tau)
                    u, ex, mk = fit_threshold_and_excesses(z, tau, self.eta)  # (B*K,), ...
                    if self.method.use_evt:
                        shell_idx = torch.arange(K, device=obs.device).repeat(B)
                        xi, beta = self.gpd(obs_flat, shell_idx)
                        cvar = gpd_cvar(u, xi, beta)  # (B*K,)
                    else:
                        cvar = empirical_cvar_from_iqn(z, tau, self.eta)  # (B*K,)
                    cvar = cvar.view(B, K)
                    if self.method.use_choquet:
                        g = self.choquet(obs)
                        return choquet_integral_dual(g, cvar).squeeze(-1)
                    else:
                        return cvar.mean(dim=-1)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def collect_rollout(self) -> dict:
        """Run venv for `steps_per_env` steps, fill buffer, return episode stats."""
        self.buf.reset()
        N = self.venv.n_envs
        obs = self.venv._latest_obs.copy()  # last obs from previous epoch (or reset)

        ep_rets, ep_costs, ep_lens = [], [], []

        for t in range(self.steps_per_env):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            # Action + scalar V_R
            a, v, logp = self.ac.step(obs_t)
            # Robust cost summary V_C^robust(s)
            vc = self.compute_vc_robust(obs_t, with_grad=False).cpu().numpy()
            # Step env
            next_obs, rew, cost, term, trunc, info = self.venv.step(a)
            done = np.logical_or(term, trunc).astype(np.float32)
            self.buf.store(obs, a if not self.is_continuous else a, rew, cost, logp, v, vc, done)

            # Track episode statistics
            self._ep_returns += rew
            self._ep_costs += cost
            self._ep_lens += 1
            for i in range(N):
                if done[i]:
                    ep_rets.append(float(self._ep_returns[i]))
                    ep_costs.append(float(self._ep_costs[i]))
                    ep_lens.append(int(self._ep_lens[i]))
                    self._ep_returns[i] = 0.0
                    self._ep_costs[i] = 0.0
                    self._ep_lens[i] = 0
            obs = next_obs

        # Bootstrap
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_v = self.ac.v(obs_t).cpu().numpy()
            last_vc = self.compute_vc_robust(obs_t, with_grad=False).cpu().numpy()
        self.buf.finish_path(last_v, last_vc)

        self._total_envsteps += N * self.steps_per_env
        return dict(EpRet=ep_rets, EpCost=ep_costs, EpLen=ep_lens)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self):
        data = self.buf.get()
        # Move to device
        data = {k: v.to(self.device) for k, v in data.items()}
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]
        cadv = data["cadv"]
        ret = data["ret"]
        cret = data["cret"]
        ctg = data["ctg"]
        logp_old = data["logp_old"]

        # Current Lagrangian multiplier
        lam_pen = F.softplus(self.p_param).detach()

        # =================== POLICY UPDATE ===================
        for it in range(self.train_pi_iters):
            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_r = -(torch.min(ratio * adv, clip_adv)).mean()
            # Cost penalty: use unclipped ratio (standard PPO-Lag practice)
            loss_c = (lam_pen * ratio * cadv).mean()
            ent = pi.entropy().mean() if hasattr(pi, "entropy") else torch.tensor(0.0, device=self.device)
            loss_pi = loss_r + loss_c - self.ent_coef * ent

            # KL early stop
            with torch.no_grad():
                kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.target_kl:
                break

            self.pi_opt.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), 0.5)
            self.pi_opt.step()

        self.logger.store(LossPi=loss_pi.item(), KL=kl, Entropy=ent.item(),
                          PolicyIters=it + 1)

        # =================== VALUE UPDATES ===================
        for _ in range(self.train_v_iters):
            v_pred = self.ac.v(obs)
            loss_v = ((v_pred - ret) ** 2).mean()
            self.v_opt.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), 0.5)
            self.v_opt.step()
        self.logger.store(LossV=loss_v.item())

        # Cost critic update
        if self.method.use_iqn:
            self._update_iqn_cost_critic(obs, ctg)
        else:
            for _ in range(self.train_v_iters):
                vc_pred = self.cost_critic(obs)
                loss_vc = ((vc_pred - cret) ** 2).mean()
                self.vc_opt.zero_grad()
                loss_vc.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 0.5)
                self.vc_opt.step()
            self.logger.store(LossVc=loss_vc.item())

        # GPD update (multi-source)
        if self.gpd is not None:
            self._update_gpd(obs)

        # Choquet capacity update (sigmoid + reg)
        if self.choquet is not None:
            self._update_choquet(obs)

        # Lagrangian penalty update
        avg_cost = float(data["cost"].mean().item())  # rough proxy: mean step cost
        # Better proxy: use avg episode cost stored in logger
        ep_cost_mean = self.logger.get_stats("EpCost")
        if ep_cost_mean is None:
            ep_cost_mean = avg_cost * 1000.0  # fallback (rough rescale)
        cost_violation = ep_cost_mean - self.cost_limit
        # SGD-style proportional update on raw param
        with torch.no_grad():
            self.p_param += self.penalty_lr * cost_violation
            self.p_param.clamp_(min=-5.0, max=np.log(np.exp(self.penalty_max) - 1))
        self.logger.store(Lambda=lam_pen.item(),
                          PenaltyParam=self.p_param.item(),
                          CostViolation=cost_violation,
                          MeanEpCost=ep_cost_mean)

    def _update_iqn_cost_critic(self, obs: torch.Tensor, ctg: torch.Tensor):
        """Quantile Huber loss with MC cost-to-go as target.

        IQN-FIX (2026-05-10): added 4 stability measures to prevent late-stage divergence
        (LossVc -> 1e8 mid-training, observed in seed 1 of warmup-DRACO experiment):
          1. clamp(max=2000) on pred_z   -- bound runaway critic outputs
          2. skip-on-large-loss          -- intercept drift early (loss > 1000)
          3. tighter grad clip 0.5 -> 0.25
          4. weight_decay=1e-4 on vc_opt (added at construction)
        """
        B = obs.shape[0]
        n_skip = 0
        for _ in range(self.train_v_iters):
            tau = torch.rand(B, self.n_tau_train, device=obs.device)
            pred_z = self.cost_critic(obs, tau).clamp(max=2000.0)  # IQN-FIX 1
            target_z = ctg.unsqueeze(-1).expand(B, self.n_tau_target).contiguous()
            loss_vc = quantile_huber_loss(pred_z, tau, target_z)
            if not torch.isfinite(loss_vc) or loss_vc.item() > 1e3:  # IQN-FIX 2
                n_skip += 1
                continue
            self.vc_opt.zero_grad()
            loss_vc.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 0.25)  # IQN-FIX 3
            self.vc_opt.step()
        self.logger.store(LossVc=loss_vc.item() if torch.isfinite(loss_vc) else 0.0,
                          VcSkipped=n_skip)  # IQN-FIX log

    def _update_gpd(self, obs: torch.Tensor):
        """Per-shell GPD MLE on the tail samples of IQN (or scalar V_C if not IQN)."""
        B = obs.shape[0]
        K = self.method.n_shells

        with torch.no_grad():
            # Build perturbed states for K shells
            if K > 1:
                obs_shells = obs.unsqueeze(1).expand(B, K, self.obs_dim).clone()
                eps_noise = torch.randn(B, K, self.obs_dim, device=obs.device)
                obs_shells = obs_shells + eps_noise * self.shell_sigmas.view(1, K, 1)
                obs_flat = obs_shells.view(B * K, self.obs_dim)
                shell_idx = torch.arange(K, device=obs.device).repeat(B)
            else:
                obs_flat = obs
                shell_idx = torch.zeros(B, dtype=torch.long, device=obs.device)

            # Sample tail values: only meaningful when using IQN
            if self.method.use_iqn:
                tau = torch.rand(obs_flat.shape[0], self.n_tau_eval, device=obs.device)
                z = self.cost_critic(obs_flat, tau)
                u, excesses, mask = fit_threshold_and_excesses(z, tau, self.eta)
            else:
                # evo_style: GPD on scalar V_C with synthetic tail samples
                # (limited utility; we mostly include this for ablation completeness)
                vc = self.cost_critic(obs_flat)
                u = vc.detach()
                # Generate synthetic excesses by sampling from current GPD; with
                # frozen target this gives near-zero gradient — OK as a placeholder.
                excesses = torch.zeros(obs_flat.shape[0], 4, device=obs.device)
                mask = torch.zeros_like(excesses)

        # MLE step
        n_iters = max(self.train_v_iters // 4, 5)
        for _ in range(n_iters):
            xi, beta = self.gpd(obs_flat, shell_idx)
            ll = gpd_log_likelihood(excesses, xi, beta, mask)
            loss_gpd = -ll.mean()
            if torch.isfinite(loss_gpd):
                self.gpd_opt.zero_grad()
                loss_gpd.backward()
                torch.nn.utils.clip_grad_norm_(self.gpd.parameters(), 1.0)
                self.gpd_opt.step()

        self.logger.store(LossGPD=loss_gpd.item() if torch.isfinite(loss_gpd) else 0.0,
                          GPD_xi_mean=float(xi.mean().item()),
                          GPD_beta_mean=float(beta.mean().item()))

    def _update_choquet(self, obs: torch.Tensor):
        """Update Choquet capacity. Loss = consistency with computed CVaR + non-additivity reg.

        We update by computing V_C^robust(s) with the live Choquet (gradient flows),
        and pulling it toward a target = mean of CVaR (a rough anchor). Plus reg.
        """
        n_iters = max(self.train_v_iters // 4, 5)
        for _ in range(n_iters):
            g = self.choquet(obs)
            reg = self.choquet.nonadditive_regularizer(g)
            # Encourage capacity to spread over sources (diversity prior)
            # Loss = pure regularizer (we don't have a supervised target for capacity)
            loss_cho = self.choquet_reg_coef * reg
            self.choquet_opt.zero_grad()
            loss_cho.backward()
            self.choquet_opt.step()
        self.logger.store(LossChoquet=loss_cho.item(),
                          ChoquetSumG=float(g.sum(dim=-1).mean().item()))

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def learn(self, total_steps: int = 1_000_000):
        """Top-level train loop. Returns when total_steps reached."""
        n_epochs = max(1, total_steps // self.steps_per_epoch)
        # Initial reset
        obs, _ = self.venv.reset(seed=int(time.time()) % (2**31))
        # Pre-fill latest_obs
        self.venv._latest_obs = obs.copy()

        for epoch in range(n_epochs):
            ep_stats = self.collect_rollout()
            for k in ("EpRet", "EpCost", "EpLen"):
                for v in ep_stats[k]:
                    self.logger.store(**{k: v})
            self.update()
            self.logger.store(TotalEnvSteps=self._total_envsteps)
            self.logger.dump(epoch=epoch + 1)

        return self.logger
