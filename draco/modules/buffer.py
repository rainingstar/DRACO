"""Vectorized rollout buffer for DRACO.

Stores (n_envs, T) trajectories. Computes:
    - Reward GAE advantages + returns (using scalar V_R critic)
    - Cost GAE advantages + returns (using a scalar V_C^{robust}(s) summary
      that is computed by the algorithm at advantage time)
    - Cost-go targets for IQN training (Monte-Carlo cost-to-go from the
      observed trajectory; standard distributional RL target)

API:
    buf = VecRolloutBuffer(n_envs, T, obs_dim, act_dim, gamma=0.99, lam=0.97,
                           cost_gamma=0.99, cost_lam=0.97)
    for t in range(T):
        buf.store(obs, act, rew, cost, logp, v, vc_robust, done)
        obs = next_obs
    buf.finish_path(last_v, last_vc_robust, last_done_mask)
    data = buf.get()  # flatten to (n_envs*T) batch
"""
from __future__ import annotations
import numpy as np
import torch


def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Reverse-cumulative discounted sum along last axis.
    x[..., t] -> sum_{i>=t} discount^(i-t) * x[..., i]
    Vectorized to handle (N, T) arrays.
    """
    out = np.zeros_like(x)
    running = np.zeros(x.shape[:-1], dtype=x.dtype)
    for t in reversed(range(x.shape[-1])):
        running = x[..., t] + discount * running
        out[..., t] = running
    return out


def _gae(
    rewards: np.ndarray,        # (N, T)
    values: np.ndarray,         # (N, T+1)  values[..., T] = bootstrap
    dones: np.ndarray,          # (N, T)    1 if episode ended at step t (mask future)
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Compute GAE advantages.
    delta_t = r_t + gamma * v_{t+1} * (1 - done_t) - v_t
    A_t     = delta_t + gamma*lam*(1-done_t)*A_{t+1}
    """
    N, T = rewards.shape
    adv = np.zeros((N, T), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[:, t] = last_gae
    return adv


class VecRolloutBuffer:
    """Vectorized rollout buffer for N envs * T steps.

    Stored fields per step:
        obs       (N, T, obs_dim)
        act       (N, T, act_dim)  (or (N, T) for discrete)
        rew       (N, T)
        cost      (N, T)
        logp      (N, T)
        v         (N, T)         scalar reward value V_R(s_t)
        vc_robust (N, T)         scalar robust cost value V_C^robust(s_t)
        done      (N, T)         1 if episode ended at step t

    Computed in finish_path:
        ret       (N, T)         reward returns (= adv + v)
        adv       (N, T)         reward GAE
        cret      (N, T)         cost returns (= cadv + vc_robust)
        cadv      (N, T)         cost GAE
        cost_to_go(N, T)         pure MC cost-to-go (no bootstrap, no GAE),
                                 used as IQN distributional target (one
                                 sample per state of the cost CDF).
    """

    def __init__(
        self,
        n_envs: int,
        steps_per_env: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        lam: float = 0.97,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.97,
        is_discrete: bool = False,
    ):
        self.N = n_envs
        self.T = steps_per_env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.is_discrete = is_discrete

        T, N = steps_per_env, n_envs
        self.obs_buf = np.zeros((N, T, obs_dim), dtype=np.float32)
        if is_discrete:
            self.act_buf = np.zeros((N, T), dtype=np.int64)
        else:
            self.act_buf = np.zeros((N, T, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((N, T), dtype=np.float32)
        self.cost_buf = np.zeros((N, T), dtype=np.float32)
        self.logp_buf = np.zeros((N, T), dtype=np.float32)
        self.v_buf = np.zeros((N, T), dtype=np.float32)
        self.vc_buf = np.zeros((N, T), dtype=np.float32)
        self.done_buf = np.zeros((N, T), dtype=np.float32)

        self.adv_buf = np.zeros((N, T), dtype=np.float32)
        self.ret_buf = np.zeros((N, T), dtype=np.float32)
        self.cadv_buf = np.zeros((N, T), dtype=np.float32)
        self.cret_buf = np.zeros((N, T), dtype=np.float32)
        # Cost-to-go (pure MC, used as IQN target).
        self.ctg_buf = np.zeros((N, T), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,        # (N, obs_dim)
        act: np.ndarray,        # (N, act_dim) or (N,)
        rew: np.ndarray,        # (N,)
        cost: np.ndarray,       # (N,)
        logp: np.ndarray,       # (N,)
        v: np.ndarray,          # (N,)
        vc: np.ndarray,         # (N,)
        done: np.ndarray,       # (N,) bool/float
    ):
        t = self.ptr
        assert t < self.T, "buffer overrun"
        self.obs_buf[:, t] = obs
        self.act_buf[:, t] = act
        self.rew_buf[:, t] = rew
        self.cost_buf[:, t] = cost
        self.logp_buf[:, t] = logp
        self.v_buf[:, t] = v
        self.vc_buf[:, t] = vc
        self.done_buf[:, t] = np.asarray(done, dtype=np.float32)
        self.ptr += 1

    def finish_path(
        self,
        last_v: np.ndarray,        # (N,) bootstrap value V_R(s_T)
        last_vc: np.ndarray,       # (N,) bootstrap robust cost value V_C^robust(s_T)
    ):
        """Compute GAE advantages, returns, and MC cost-to-go.
        Call once at the end of rollout (after self.T steps stored).
        """
        assert self.ptr == self.T, f"need full buffer, got ptr={self.ptr}/{self.T}"

        # Reward GAE
        v_with_boot = np.concatenate([self.v_buf, last_v[:, None]], axis=1)  # (N, T+1)
        self.adv_buf = _gae(
            self.rew_buf, v_with_boot, self.done_buf, self.gamma, self.lam,
        )
        self.ret_buf = self.adv_buf + self.v_buf

        # Cost GAE (using scalar robust cost value summary as critic)
        vc_with_boot = np.concatenate([self.vc_buf, last_vc[:, None]], axis=1)
        self.cadv_buf = _gae(
            self.cost_buf, vc_with_boot, self.done_buf, self.cost_gamma, self.cost_lam,
        )
        self.cret_buf = self.cadv_buf + self.vc_buf

        # MC cost-to-go (no bootstrap inside the rollout segment; on the final
        # state we ADD last_vc * gamma^remaining as a soft bootstrap to reduce
        # the truncation bias). For IQN this is a single sample of F_C^{-1}.
        # Standard discrete cumsum reset on done.
        ctg = np.zeros((self.N, self.T), dtype=np.float32)
        running = last_vc.copy().astype(np.float32)  # (N,)
        for t in reversed(range(self.T)):
            not_done = 1.0 - self.done_buf[:, t]
            running = self.cost_buf[:, t] + self.cost_gamma * running * not_done
            ctg[:, t] = running
        self.ctg_buf = ctg

    def get(self) -> dict:
        """Flatten (N, T, ...) -> (N*T, ...) tensors. Reset ptr. Normalize advantages."""
        assert self.ptr == self.T

        adv = self.adv_buf.reshape(-1)
        # Standard PPO advantage norm (per batch)
        adv_mean, adv_std = adv.mean(), adv.std()
        adv_norm = (adv - adv_mean) / (adv_std + 1e-8)

        cadv = self.cadv_buf.reshape(-1)
        # Cost advantage NOT centered: keep sign meaningful (positive = bad)
        # but standardize scale.
        cadv_norm = cadv / (cadv.std() + 1e-8)

        out = dict(
            obs=torch.as_tensor(self.obs_buf.reshape(-1, self.obs_dim), dtype=torch.float32),
            act=torch.as_tensor(self.act_buf.reshape(-1, self.act_dim) if not self.is_discrete else self.act_buf.reshape(-1)),
            rew=torch.as_tensor(self.rew_buf.reshape(-1), dtype=torch.float32),
            cost=torch.as_tensor(self.cost_buf.reshape(-1), dtype=torch.float32),
            logp_old=torch.as_tensor(self.logp_buf.reshape(-1), dtype=torch.float32),
            v_old=torch.as_tensor(self.v_buf.reshape(-1), dtype=torch.float32),
            vc_old=torch.as_tensor(self.vc_buf.reshape(-1), dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf.reshape(-1), dtype=torch.float32),
            cret=torch.as_tensor(self.cret_buf.reshape(-1), dtype=torch.float32),
            adv=torch.as_tensor(adv_norm, dtype=torch.float32),
            cadv=torch.as_tensor(cadv_norm, dtype=torch.float32),
            ctg=torch.as_tensor(self.ctg_buf.reshape(-1), dtype=torch.float32),
            done=torch.as_tensor(self.done_buf.reshape(-1), dtype=torch.float32),
        )
        self.ptr = 0
        return out
