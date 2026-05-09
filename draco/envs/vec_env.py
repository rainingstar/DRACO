"""Vectorized environment wrapper for parallel rollout in DRACO.

Two implementations:

(a) SyncVectorEnv — N envs in a Python loop, batched obs/action tensors.
    Pros: simple, low overhead, no IPC. Best for fast envs (cartpole_stab,
          control envs with cheap dynamics).
    Cons: GIL-bound; doesn't parallelize Python-heavy env code.

(b) AsyncVectorEnv — N worker processes, true parallelism via multiprocessing.
    Pros: real wall-clock speedup for slow envs (MuJoCo, large state spaces).
    Cons: pickle overhead, harder to debug, ~50ms startup cost.

Both expose the same API:
    env.reset(seed=...) -> (obs, info)
    env.step(actions) -> (obs, rew, cost, terminated, truncated, info)

Note: action and reward are batched across N envs; cost is treated as a
separate scalar field (not in info dict, for explicit handling).

Usage:
    venv = SyncVectorEnv(env_fns)  # list of len N, each returns env instance
    obs, _ = venv.reset(seed=42)
    for step in range(T):
        actions = policy(obs)               # (N, act_dim) tensor
        obs, rew, cost, term, trunc, info = venv.step(actions)
        # auto-reset on episode end is handled internally
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import numpy as np


class _BaseVectorEnv:
    """Common interface for sync/async vec envs."""
    n_envs: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        raise NotImplementedError

    def close(self):
        pass


def _extract_cost(info: dict, default: float = 0.0) -> float:
    """Extract scalar cost from info dict. Tries common safe-RL keys."""
    for k in ("cost", "constraint_violation", "violation"):
        if k in info:
            v = info[k]
            return float(v) if not isinstance(v, (list, tuple, np.ndarray)) else float(np.asarray(v).sum())
    return default


class SyncVectorEnv(_BaseVectorEnv):
    """Synchronous vectorized env: N envs in a single process, batched arrays.

    For very fast envs (cartpole_stab ~ 100us/step) this is faster than
    AsyncVectorEnv because IPC overhead dominates the env step cost.

    Args:
        env_fns: list of callables, each returning a constructed env instance.
                 Length determines n_envs.
    """

    def __init__(self, env_fns: List[Callable[[], object]]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        # Probe dimensions from first env's reset
        sample_obs = self._reset_one(0, seed=0)
        self.obs_dim = int(np.asarray(sample_obs).reshape(-1).shape[0])

        # Probe action space
        e0 = self.envs[0]
        if hasattr(e0, "action_space"):
            sp = e0.action_space
            self.action_low = np.asarray(sp.low, dtype=np.float32)
            self.action_high = np.asarray(sp.high, dtype=np.float32)
            self.act_dim = int(np.prod(sp.shape))
        else:
            raise AttributeError("env must expose .action_space")

        # Latest obs cache (for auto-reset)
        self._latest_obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)

    def _reset_one(self, idx: int, seed: Optional[int] = None) -> np.ndarray:
        """Reset env idx. Tolerates both gym (returns obs) and gymnasium (returns (obs, info)) APIs."""
        env = self.envs[idx]
        if seed is not None:
            try:
                out = env.reset(seed=seed)
            except TypeError:
                # Old API: separate seed call
                if hasattr(env, "seed"):
                    env.seed(seed)
                out = env.reset()
        else:
            out = env.reset()
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        for i in range(self.n_envs):
            s = None if seed is None else (seed + i)
            self._latest_obs[i] = self._reset_one(i, seed=s)
        return self._latest_obs.copy(), {}

    def step(self, actions: np.ndarray):
        """Step all envs. Auto-reset on episode end.

        Returns:
            obs:        (N, obs_dim) NEXT obs (post auto-reset for done envs).
            rew:        (N,) rewards.
            cost:       (N,) safe-RL costs (extracted from info).
            terminated: (N,) bool.
            truncated:  (N,) bool.
            info:       list of per-env info dicts (with "final_obs" set for
                        envs that just reset, so the algorithm can compute
                        the bootstrap properly).
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.n_envs, self.act_dim)
        assert actions.shape == (self.n_envs, self.act_dim)

        obs_arr = np.zeros_like(self._latest_obs)
        rew_arr = np.zeros(self.n_envs, dtype=np.float32)
        cost_arr = np.zeros(self.n_envs, dtype=np.float32)
        term_arr = np.zeros(self.n_envs, dtype=bool)
        trunc_arr = np.zeros(self.n_envs, dtype=bool)
        infos: List[dict] = []

        for i, env in enumerate(self.envs):
            out = env.step(actions[i])
            # 4-tuple (gym old) or 5-tuple (gymnasium new)
            if len(out) == 5:
                next_obs, r, term, trunc, info = out
            else:
                next_obs, r, done, info = out
                term, trunc = bool(done), False

            cost = _extract_cost(info)
            done_now = bool(term) or bool(trunc)

            if done_now:
                # Save the true final obs so trainer can bootstrap correctly,
                # then auto-reset.
                final_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                info = {**info, "final_obs": final_obs, "episode_done": True}
                next_obs = self._reset_one(i)
            else:
                info = {**info, "episode_done": False}

            obs_arr[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            rew_arr[i] = float(r)
            cost_arr[i] = float(cost)
            term_arr[i] = bool(term)
            trunc_arr[i] = bool(trunc)
            infos.append(info)

        self._latest_obs = obs_arr
        return obs_arr.copy(), rew_arr, cost_arr, term_arr, trunc_arr, infos

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# AsyncVectorEnv (multiprocessing). Optional; only needed for slow envs.
# ----------------------------------------------------------------------------
def make_async_vector_env(env_fns: List[Callable[[], object]]):
    """Lazy import to avoid forcing multiprocessing dependency.

    Use this for MuJoCo / Robust-Gymnasium tasks where each step is >1ms.
    For cartpole_stab and other fast envs, SyncVectorEnv is faster.
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("AsyncVectorEnv requires gymnasium. Install with `pip install gymnasium`.")
    raise NotImplementedError(
        "Async wrapper deferred to second iteration; use SyncVectorEnv first. "
        "If your env is slow enough to need async, "
        "wrap with gymnasium.vector.AsyncVectorEnv directly."
    )
