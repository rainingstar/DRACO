"""Tiny synthetic safe RL env for smoke testing.

A 4-dim state, 1-dim action linear-quadratic-with-cost env. The dynamics
are a pendulum-like nonlinear system with a state-bound cost.

Used purely for smoke-testing DRACO without depending on safe-control-gym
or robust-gymnasium.
"""
from __future__ import annotations
import numpy as np


class Box:
    """Minimal Box space (api compatible with our needs).
    Named 'Box' for compatibility with `type(space).__name__ == 'Box'` checks.
    """
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class ToySafeEnv:
    """Toy safe RL env: 4-dim state, 1-dim continuous action.

    Dynamics:
        s_{t+1} = A s_t + B a + small noise
    Reward:
        r = -||s||^2 - 0.01||a||^2
    Cost:
        c = 1 if ||s||_inf > bound else 0   (boundary violation)

    Episode terminates if ||s||_inf > bound * 2 or t >= max_steps.
    """

    def __init__(self, max_steps: int = 100, bound: float = 1.0, seed: int = 0):
        self.max_steps = max_steps
        self.bound = bound
        self._rng = np.random.RandomState(seed)
        self.action_space = Box(low=-2.0 * np.ones(1), high=2.0 * np.ones(1), shape=(1,))
        self.observation_space = Box(low=-np.inf * np.ones(4), high=np.inf * np.ones(4), shape=(4,))
        # Linear dynamics
        self.A = np.array([
            [1.0, 0.05, 0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.05],
            [0.0, 0.0,  0.0, 1.0],
        ], dtype=np.float32)
        self.B = np.array([[0.0], [0.05], [0.0], [0.05]], dtype=np.float32)
        self.s = None
        self.t = 0

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.s = self._rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
        self.t = 0
        return self.s.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -2.0, 2.0)
        noise = self._rng.normal(0.0, 0.02, size=4).astype(np.float32)
        self.s = self.A @ self.s + (self.B @ action).reshape(-1) + noise
        self.t += 1
        norm = float(np.linalg.norm(self.s))
        rew = -norm ** 2 - 0.01 * float((action ** 2).sum())
        max_abs = float(np.max(np.abs(self.s)))
        cost = 1.0 if max_abs > self.bound else 0.0
        terminated = max_abs > 2.0 * self.bound
        truncated = self.t >= self.max_steps
        info = {"cost": cost}
        return self.s.copy(), float(rew), bool(terminated), bool(truncated), info

    def close(self):
        pass


def make_toy_env_fn(seed: int = 0, max_steps: int = 100, bound: float = 1.0):
    def _fn():
        return ToySafeEnv(max_steps=max_steps, bound=bound, seed=seed)
    return _fn
