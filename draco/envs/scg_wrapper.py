"""Adapter to use the *real* safe-control-gym environments through a
clean Gymnasium-style API.

Supports all four Fuz-RL paper tasks:
  - cartpole_stab    (4-d obs, 1-d act)
  - cartpole_track   (4-d obs, 1-d act, time-varying reference)
  - quadrotor_stab   (12-d obs, 2-d act, planar quadrotor)
  - quadrotor_track  (12-d obs, 2-d act, time-varying reference)

==============================================================================
Three perturbation channels matching Fuz-RL paper Table 3
==============================================================================

(1) OBSERVATION NOISE -- additive Gaussian, restricted to a subset of dims:
        obs_noise_std        : float, the std (paper uses up to 0.1)
        obs_noise_dims       : list[int] of obs dims to perturb (None = all)
                               Per Table 3:
                               CartPole : [0, 1, 2, 3]            (x, x_dot, theta, theta_dot)
                               Quadrotor: [0, 1, 4, 5, 7, 8]      (x, x_dot, z, z_dot, theta, theta_dot)
                               (the others are kept clean)

(2) DYNAMICS NOISE -- multiplicative Gaussian on inertial parameters,
    sampled ONCE at every episode reset:
        dynamics_noise_std   : float, relative std (paper uses up to 0.1)
        dynamics_noise_params: list[str] of attribute names to perturb.
                               Default per task:
                               CartPole : ['POLE_LENGTH', 'POLE_MASS']
                               Quadrotor: ['MASS', 'IYY']

(3) ACTION NOISE -- two flavors, mutually exclusive:
   (3a) Gaussian per-step (legacy v3 backward-compat):
        act_noise_std        : float
   (3b) Impulse pulse (paper Table 3):
        act_impulse_force    : float, max |F0|; F0 ~ U[-this, +this] sampled
                               once per episode at reset.
        act_impulse_step_offset: int, pulse starts at this step index.
        act_impulse_duration : int, pulse lasts this many steps.
        act_impulse_decay    : float in (0, 1], exponential decay per step
                               so pulse[t] = F0 * decay^(t - offset).

    If both `act_noise_std > 0` and `act_impulse_force > 0`, the impulse
    takes precedence and a warning is printed.

==============================================================================
This file is OPTIONAL: it imports safe_control_gym lazily, so the package
still imports cleanly even when SCG / pybullet is not installed.
==============================================================================
"""

from __future__ import annotations
from typing import Optional, List, Sequence
import warnings

import numpy as np


def _import_scg():
    try:
        from safe_control_gym.utils.registration import make as scg_make
        return scg_make
    except ImportError as e:
        raise ImportError(
            "safe-control-gym is not installed. To use SCGWrapper, run:\n"
            "    pip install pybullet munch dict-deep casadi\n"
            "    pip install -e <path-to-safe-control-gym>\n"
            f"Underlying error: {e}"
        )


# ============================================================================
# Default task configs -- exactly mirror Fuz-RL paper Section 5.1 / Appendix
# C.1 settings (cartpole.yaml / quadrotor.yaml) so results are comparable.
# ============================================================================

def _config_cartpole_stab() -> dict:
    """SafeCartPole-Stab: stabilise the pole upright. theta in [-0.2, 0.2]."""
    return dict(
        env_id='cartpole',
        ctrl_freq=50, pyb_freq=50, physics='pyb', gui=False, verbose=False,
        normalized_rl_action_space=True,
        info_mse_metric_state_weight=[1.0, 0.0, 1.0, 0.0],
        episode_len_sec=5,                    # 250 steps @ 50 Hz
        cost='quadratic',
        task='stabilization',
        task_info=dict(stabilization_goal=[0.0, 0.0],
                       stabilization_goal_tolerance=0.0),
        rew_state_weight=[1.0, 0.1, 1.0, 0.1],
        rew_act_weight=0.1,
        # IMPORTANT: we explicitly set rew_exponential=False at the SCG layer
        # because different SCG versions silently ignore it (cf. user's debug
        # log showing reward=-2.65 even with rew_exponential=True). We handle
        # the exp() transform ourselves in SCGWrapper.step() to ensure
        # FuzRL-paper-compatible positive rewards in [0, 1] across SCG versions.
        rew_exponential=False,
        done_on_violation=False,
        done_on_out_of_bound=True,
        constraints=[dict(
            constraint_form='bounded_constraint',
            constrained_variable='state',
            active_dims=[2],
            lower_bounds=[-0.2], upper_bounds=[0.2],
        )],
    )


def _config_cartpole_track() -> dict:
    """SafeCartPole-Track: track a sinusoidal reference. theta in [-0.2, 0.2]."""
    cfg = _config_cartpole_stab()
    cfg.update(dict(
        task='traj_tracking',
        task_info=dict(
            trajectory_type='sine',
            num_cycles=1,
            trajectory_plane='xz',
            trajectory_position_offset=[0.0, 0.0],
            trajectory_scale=0.1,
            proj_point=[0.0, 0.0, 0.5],
            proj_normal=[0.0, 1.0, 0.0],
        ),
    ))
    return cfg


def _config_quadrotor_stab() -> dict:
    """SafeQuadrotor-Stab (2D planar): hover at z=1.0. z in [-0.5, 1.5]."""
    return dict(
        env_id='quadrotor',
        ctrl_freq=60, pyb_freq=240, physics='pyb', gui=False, verbose=False,
        quad_type=2,                          # 2D planar quadrotor (12-d obs)
        normalized_rl_action_space=False,
        episode_len_sec=5,                    # 300 steps @ 60 Hz
        cost='rl_reward',
        task='stabilization',
        task_info=dict(stabilization_goal=[0.0, 1.0],
                       stabilization_goal_tolerance=0.05),
        rew_state_weight=[1.0, 0.001, 1.0, 0.001, 0.001, 0.001],
        rew_act_weight=0.0001,
        # See _config_cartpole_stab note. We handle exp() in SCGWrapper.step().
        rew_exponential=False,
        done_on_violation=False,
        done_on_out_of_bound=True,
        constraints=[dict(
            constraint_form='bounded_constraint',
            constrained_variable='state',
            active_dims=[2],                  # z position
            lower_bounds=[-0.5], upper_bounds=[1.5],
        )],
        norm_act_scale=0.1,
    )


def _config_quadrotor_track() -> dict:
    """SafeQuadrotor-Track: figure-8 trajectory. z in [-0.5, 1.5]."""
    cfg = _config_quadrotor_stab()
    cfg.update(dict(
        task='traj_tracking',
        task_info=dict(
            trajectory_type='figure8',
            num_cycles=1,
            trajectory_plane='xz',
            trajectory_position_offset=[0.0, 1.0],
            trajectory_scale=0.5,
        ),
    ))
    return cfg


_TASK_CONFIGS = {
    'cartpole_stab': _config_cartpole_stab,
    'cartpole_track': _config_cartpole_track,
    'quadrotor_stab': _config_quadrotor_stab,
    'quadrotor_track': _config_quadrotor_track,
}

# ============================================================================
# Per-paper-Table-3 default dim/param sets for each task family
# ============================================================================
PAPER_OBS_DIMS = {
    # Per Fuz-RL paper Table 3:
    'cartpole_stab':   [0, 1, 2, 3],          # (x, x_dot, theta, theta_dot)
    'cartpole_track':  [0, 1, 2, 3],
    # Quadrotor 2D planar obs is 12-d in SCG. The paper applies obs noise to
    # (x, x_dot, z, z_dot, theta, theta_dot). In SCG quad_type=2 the obs
    # convention is [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    # so the corresponding indices are 0, 1, 4, 5, 7, 10 (theta=index 7,
    # theta_dot=q=index 10). Some SCG versions use a slightly different
    # ordering -- if your paired-state mapping differs, override via
    # `obs_noise_dims` kwarg.
    'quadrotor_stab':  [0, 1, 4, 5, 7, 10],
    'quadrotor_track': [0, 1, 4, 5, 7, 10],
}

PAPER_DYN_PARAMS = {
    # Per Fuz-RL paper Table 3:
    # CartPole: pole_length, pole_mass
    'cartpole_stab':   ['POLE_LENGTH', 'POLE_MASS'],
    'cartpole_track':  ['POLE_LENGTH', 'POLE_MASS'],
    # Quadrotor: quadrotor mass, quadrotor inertia
    'quadrotor_stab':  ['MASS', 'IYY'],
    'quadrotor_track': ['MASS', 'IYY'],
}


def paper_perturbation_kwargs(task_name: str, eps: float) -> dict:
    """Return Fuz-RL Table-3-style perturbation kwargs at intensity eps.

    eps is interpreted as the perturbation magnitude:
        obs noise std       = eps
        dynamics noise std  = eps
        impulse force       = eps * 10        (so eps=0.1 -> Force in [-1, 1])

    Pass the returned dict as **kwargs to SCGWrapper.
    """
    return dict(
        obs_noise_std=float(eps),
        obs_noise_dims=PAPER_OBS_DIMS.get(task_name),
        dynamics_noise_std=float(eps),
        dynamics_noise_params=PAPER_DYN_PARAMS.get(task_name),
        # Impulse: Force ~ U[-eps*10, eps*10], so eps=0.1 reproduces paper's
        # [-1, 1] range exactly.
        act_impulse_force=float(eps) * 10.0,
        act_impulse_step_offset=20,
        act_impulse_duration=80,
        act_impulse_decay=0.9,
    )


# ============================================================================
# The wrapper
# ============================================================================
class SCGWrapper:
    """Universal Gymnasium-style adapter for all four safe-control-gym tasks.

    See module docstring for the full perturbation-channel API. All
    perturbation kwargs default to 0.0 / None (no perturbation).
    """

    def __init__(
        self,
        task_name: str = 'cartpole_stab',
        seed: int = 0,
        # ---- Reward post-processing ---------------------------------------
        # If True, transform reward via exp(reward) so the final reward is in
        # [0, 1] for the typical SCG quadratic-cost case (where SCG returns
        # raw -cost). This matches Fuz-RL paper convention. Disable for envs
        # that already return positive rewards. Default ON.
        rew_post_exp: bool = True,
        # ---- Channel 1: Observation noise ---------------------------------
        obs_noise_std: float = 0.0,
        obs_noise_dims: Optional[Sequence[int]] = None,
        # ---- Channel 2: Action noise (Gaussian OR impulse) ---------------
        act_noise_std: float = 0.0,                        # legacy Gaussian
        act_impulse_force: float = 0.0,                    # paper-style impulse
        act_impulse_step_offset: int = 20,
        act_impulse_duration: int = 80,
        act_impulse_decay: float = 0.9,
        # ---- Channel 3: Dynamics noise -----------------------------------
        param_noise_std: float = 0.0,
        dynamics_noise_std: float = 0.0,
        dynamics_noise_params: Optional[Sequence[str]] = None,
        # ---- SCG-side overrides ------------------------------------------
        config_overrides: Optional[dict] = None,
    ):
        if task_name not in _TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_name!r}. Choose from "
                f"{list(_TASK_CONFIGS.keys())}"
            )

        # Resolve action-noise mutual exclusivity
        if act_noise_std > 0 and act_impulse_force > 0:
            warnings.warn(
                "Both act_noise_std and act_impulse_force were specified. "
                "Impulse takes precedence; Gaussian disabled."
            )
            act_noise_std = 0.0

        scg_make = _import_scg()
        kwargs = _TASK_CONFIGS[task_name]()
        env_id = kwargs.pop('env_id')
        if config_overrides:
            kwargs.update(config_overrides)
        kwargs['seed'] = seed
        self.task_name = task_name
        self.env = scg_make(env_id, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Reward post-processing
        self.rew_post_exp = bool(rew_post_exp)

        # Stash perturbation config
        self.obs_noise_std = float(obs_noise_std)
        if obs_noise_dims is None:
            self.obs_noise_dims = None    # all dims
        else:
            self.obs_noise_dims = np.asarray(list(obs_noise_dims), dtype=np.int64)

        self.act_noise_std = float(act_noise_std)
        self.act_impulse_force = float(act_impulse_force)
        self.act_impulse_step_offset = int(act_impulse_step_offset)
        self.act_impulse_duration = int(act_impulse_duration)
        self.act_impulse_decay = float(act_impulse_decay)

        self.param_noise_std = float(param_noise_std)
        self.dynamics_noise_std = float(dynamics_noise_std)
        self.dynamics_noise_params = list(dynamics_noise_params) if dynamics_noise_params else None

        self.np_random = np.random.RandomState(seed)
        self._step_idx = 0
        self._impulse_F0 = 0.0     # sampled at reset

        # Cache nominal inertial-parameter values (for multiplicative jitter)
        self._nominal_params = self._read_inertial_params()

    # =========================================================================
    # Inertial-parameter handling (dynamics perturbation)
    # =========================================================================
    def _read_inertial_params(self) -> dict:
        """Read current inertial params from the underlying env.

        SCG envs typically store these as instance attributes on the
        unwrapped env (e.g. env.PYB_PARAM, env.cart_mass, env.M, etc.).
        We try common names and silently skip what isn't found.
        """
        out = {}
        # The list of attribute names we care about; covers cartpole + quadrotor
        candidates = [
            'POLE_LENGTH', 'POLE_MASS', 'CART_MASS',          # CartPole
            'pole_length', 'pole_mass', 'cart_mass',
            'MASS', 'IYY', 'IXX', 'IZZ',                       # Quadrotor
            'mass', 'Iyy', 'Ixx', 'Izz', 'm',
        ]
        # Walk through wrapped envs to find the underlying SCG env
        target = self.env
        for attr in ('unwrapped', 'env'):
            if hasattr(target, attr):
                inner = getattr(target, attr)
                # Only descend if the inner has more attrs than we'd expect
                # of a thin wrapper.
                if any(hasattr(inner, c) for c in candidates):
                    target = inner
                    break
        for c in candidates:
            if hasattr(target, c):
                try:
                    val = getattr(target, c)
                    out[c] = float(val) if np.isscalar(val) else val
                except (TypeError, ValueError):
                    pass
        self._inertial_target = target    # remember where to write back
        return out

    def _apply_inertial_jitter(self):
        """Jitter the inertial params on the underlying env.

        This is called at every reset(). If `dynamics_noise_std > 0` we use
        the explicit `dynamics_noise_params` list; otherwise if the legacy
        `param_noise_std > 0` we jitter ALL known params uniformly.
        """
        if self.dynamics_noise_std > 0 and self.dynamics_noise_params:
            param_list = self.dynamics_noise_params
            std = self.dynamics_noise_std
        elif self.param_noise_std > 0:
            param_list = list(self._nominal_params.keys())
            std = self.param_noise_std
        else:
            return

        for p in param_list:
            if p not in self._nominal_params:
                # Skip silently — different SCG versions have different attr names
                continue
            nominal = self._nominal_params[p]
            jitter = 1.0 + std * self.np_random.randn()
            try:
                setattr(self._inertial_target, p, nominal * jitter)
            except AttributeError:
                pass

    def _restore_inertial_params(self):
        for p, v in self._nominal_params.items():
            try:
                setattr(self._inertial_target, p, v)
            except AttributeError:
                pass

    # =========================================================================
    # Per-step perturbations
    # =========================================================================
    def _obs_perturb(self, obs):
        if self.obs_noise_std <= 0:
            return obs
        noise = self.np_random.randn(*obs.shape).astype(obs.dtype) * self.obs_noise_std
        if self.obs_noise_dims is not None:
            mask = np.zeros_like(obs)
            mask[self.obs_noise_dims] = 1.0
            noise = noise * mask
        return obs + noise

    def _act_perturb(self, act):
        # 1. Gaussian (legacy)
        if self.act_noise_std > 0:
            act = act + self.act_noise_std * self.np_random.randn(*act.shape).astype(act.dtype)
        # 2. Impulse (paper Table 3)
        if self.act_impulse_force > 0:
            # Compute pulse value at this step (0 outside [offset, offset+duration))
            t = self._step_idx
            o = self.act_impulse_step_offset
            if o <= t < o + self.act_impulse_duration:
                pulse = self._impulse_F0 * (self.act_impulse_decay ** (t - o))
            else:
                pulse = 0.0
            act = act + np.float32(pulse)
        return act

    # =========================================================================
    # Gymnasium API
    # =========================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Sample new dynamics jitter & impulse F0 for this episode
        self._apply_inertial_jitter()
        if self.act_impulse_force > 0:
            self._impulse_F0 = float(self.np_random.uniform(
                -self.act_impulse_force, self.act_impulse_force))
        else:
            self._impulse_F0 = 0.0
        self._step_idx = 0

        out = self.env.reset()
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        return self._obs_perturb(np.asarray(obs, dtype=np.float32)), info

    def step(self, action):
        action = self._act_perturb(np.asarray(action, dtype=np.float32))
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, False

        # Reward post-processing: SCG with cost='quadratic' returns
        # reward = -(quadratic_cost). We exp() this to recover the
        # FuzRL paper's [0, 1] reward in case the SCG version's
        # rew_exponential flag was silently ignored. Safe to apply since
        # we forced rew_exponential=False in the SCG config above.
        if self.rew_post_exp:
            reward = float(np.exp(reward))

        # Normalise constraint-violation reporting
        if 'constraint_violation' in info:
            cv = int(bool(info['constraint_violation']))
        elif 'constraint_values' in info:
            cv = int(any(v < 0 for v in info['constraint_values']))
        else:
            cv = 0
        info['constraint_violation'] = cv
        info['cost'] = float(cv)  # canonical cost field for vec_env extraction

        self._step_idx += 1
        return (
            self._obs_perturb(np.asarray(obs, dtype=np.float32)),
            reward, terminated, truncated, info,
        )

    def close(self):
        # Restore params before closing so the next env created by SCG sees clean state
        self._restore_inertial_params()
        if hasattr(self.env, "close"):
            self.env.close()


SCGCartPoleAdapter = SCGWrapper                  # backwards-compat alias
