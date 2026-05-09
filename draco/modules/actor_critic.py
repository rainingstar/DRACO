"""Actor + scalar reward critic for DRACO.

Cost critic is intentionally NOT here — it's the IQN-based distributional
critic in iqn_critic.py. We keep reward critic scalar (PPO-standard) for
compute efficiency; only the cost side needs distributional treatment.

Supports both Box (continuous) and Discrete actions, gym + gymnasium.
"""
from __future__ import annotations
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def _is_box(space) -> bool:
    return type(space).__name__ == "Box"


def _is_discrete(space) -> bool:
    return type(space).__name__ == "Discrete"


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module: nn.Module) -> int:
    return int(sum(int(np.prod(p.shape)) for p in module.parameters()))


class _Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(_Actor):
    def __init__(self, obs_dim, act_n, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_n], activation)

    def _distribution(self, obs):
        return Categorical(logits=self.logits_net(obs))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(_Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    """Standard scalar critic. Used only for reward (V_R) in DRACO."""

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class ActorRewardCritic(nn.Module):
    """Actor + scalar reward critic (V_R only).

    The cost critic (IQN) is held separately in the algorithm class to make
    the dependency structure explicit.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: type = nn.Tanh,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if _is_box(action_space):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            self.is_continuous = True
        elif _is_discrete(action_space):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            self.is_continuous = False
        else:
            raise ValueError(f"Unsupported action space {type(action_space).__name__}")
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    @torch.no_grad()
    def step(self, obs: torch.Tensor):
        """One env-step query (BATCHED). obs: (B, obs_dim).
        Returns (a: np[B,...], v: np[B], logp: np[B]). Cost value comes from
        the IQN critic, called separately by the algorithm.
        """
        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return (
            a.cpu().numpy(),
            v.cpu().numpy(),
            logp_a.cpu().numpy(),
        )

    def act_deterministic(self, obs: torch.Tensor):
        with torch.no_grad():
            if self.is_continuous:
                return self.pi.mu_net(obs).cpu().numpy()
            logits = self.pi.logits_net(obs)
            return logits.argmax(dim=-1).cpu().numpy()
