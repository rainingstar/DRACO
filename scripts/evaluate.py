"""Three-channel evaluation script for DRACO.

Evaluates a trained model under FuzRL-paper Table-3-style perturbations:
    - obs    : Gaussian observation noise on a subset of dims
    - act    : impulse force at a fixed step offset
    - dyn    : multiplicative jitter on inertial parameters

For each channel, runs N episodes with the perturbation magnitude `--eps`
and reports (avg_return, avg_cost, std). The output is a JSON file matching
the format consumed by plot_results.py.

Usage:
    python scripts/evaluate.py --model-path runs/cartpole_stab/draco/<run>.model.pt \\
        --task cartpole_stab --eps 0.1 --n-eval-episodes 30
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch

from draco.algos.draco import DRACOAlgorithm, MethodConfig
from draco.envs.vec_env import SyncVectorEnv
from draco.envs.toy_env import make_toy_env_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to .model.pt saved by train.py --save-model.")
    p.add_argument("--task", type=str, required=True,
                   choices=["toy", "cartpole_stab", "cartpole_track",
                            "quadrotor_stab", "quadrotor_track"])
    p.add_argument("--n-eval-episodes", type=int, default=30,
                   help="Episodes per channel.")
    p.add_argument("--eps", type=float, default=0.1,
                   help="Perturbation magnitude (paper uses up to 0.1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path. Defaults to <model_path>.eval.json.")
    return p.parse_args()


def make_perturbed_env_fn(task: str, channel: str, eps: float, seed: int):
    """Build env factory with one perturbation channel active."""
    if task == "toy":
        # toy env doesn't support these perturbations; just return clean env
        return make_toy_env_fn(seed=seed)

    from draco.envs.scg_wrapper import (
        SCGWrapper, paper_perturbation_kwargs, PAPER_OBS_DIMS, PAPER_DYN_PARAMS,
    )

    if channel == "clean":
        kwargs = {}
    elif channel == "obs":
        kwargs = dict(
            obs_noise_std=eps,
            obs_noise_dims=PAPER_OBS_DIMS.get(task),
        )
    elif channel == "act":
        kwargs = dict(
            act_impulse_force=eps * 10.0,
            act_impulse_step_offset=20,
            act_impulse_duration=80,
            act_impulse_decay=0.9,
        )
    elif channel == "dyn":
        kwargs = dict(
            dynamics_noise_std=eps,
            dynamics_noise_params=PAPER_DYN_PARAMS.get(task),
        )
    else:
        raise ValueError(f"Unknown channel: {channel}")

    def _fn():
        return SCGWrapper(task_name=task, seed=seed, **kwargs)
    return _fn


def evaluate_channel(algo, task: str, channel: str, eps: float, n_episodes: int, seed: int):
    """Roll out n_episodes with the given perturbation channel; return stats."""
    fn = make_perturbed_env_fn(task, channel, eps, seed)
    env = fn()

    rets, costs, lens = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep) if hasattr(env, "reset") else (env.reset(), {})
        if isinstance(obs, tuple):
            obs = obs[0]
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        done = False
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1),
                                    dtype=torch.float32, device=algo.device)
            # Deterministic action for evaluation
            act = algo.ac.act_deterministic(obs_t)
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            out = env.step(act)
            if len(out) == 5:
                obs, r, term, trunc, info = out
            else:
                obs, r, dn, info = out
                term, trunc = dn, False
            ep_ret += float(r)
            ep_cost += float(info.get("cost", info.get("constraint_violation", 0.0)))
            ep_len += 1
            done = bool(term) or bool(trunc)
        rets.append(ep_ret)
        costs.append(ep_cost)
        lens.append(ep_len)
    if hasattr(env, "close"):
        env.close()

    return dict(
        ret_mean=float(np.mean(rets)), ret_std=float(np.std(rets)),
        cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs)),
        len_mean=float(np.mean(lens)),
        n_episodes=n_episodes,
    )


def load_algo(model_path: str, task: str, device: str):
    """Load model + reconstruct algo skeleton (no training)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    method_name = ckpt["method"]
    args_dict = ckpt["args"]
    n_shells = args_dict.get("n_shells", 4)
    method = MethodConfig.from_name(method_name, n_shells=n_shells)

    # Build a thin venv just to probe spaces
    if task == "toy":
        venv = SyncVectorEnv([make_toy_env_fn(seed=0)])
    else:
        from draco.envs.scg_wrapper import SCGWrapper
        def _fn():
            return SCGWrapper(task_name=task, seed=0)
        venv = SyncVectorEnv([_fn])

    hidden_sizes = tuple(int(x) for x in args_dict.get("hidden_sizes", "64,64").split(","))
    shell_sigmas = tuple(float(x) for x in args_dict.get("shell_sigmas", "0.01,0.05,0.10,0.20").split(","))

    algo = DRACOAlgorithm(
        venv=venv, method=method, cost_limit=args_dict.get("cost_limit", 25.0),
        seed=args_dict.get("seed", 0), device=device,
        log_dir="/tmp/_draco_eval", run_name="_eval",
        hidden_sizes=hidden_sizes, shell_sigmas=shell_sigmas,
        steps_per_epoch=128,  # not used for eval
    )
    algo.ac.load_state_dict(ckpt["ac_state_dict"])
    algo.cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
    if algo.gpd is not None and "gpd_state_dict" in ckpt:
        algo.gpd.load_state_dict(ckpt["gpd_state_dict"])
    if algo.choquet is not None and "choquet_state_dict" in ckpt:
        algo.choquet.load_state_dict(ckpt["choquet_state_dict"])
    venv.close()
    return algo, method_name


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)
    print(f"[evaluate] loading {args.model_path}, device={device}")
    algo, method_name = load_algo(args.model_path, args.task, device)
    algo.ac.eval()
    algo.cost_critic.eval()

    channels = ["clean", "obs", "act", "dyn"] if args.task != "toy" else ["clean"]
    out = {
        "model_path": args.model_path,
        "task": args.task,
        "method": method_name,
        "eps": args.eps,
        "n_eval_episodes": args.n_eval_episodes,
        "channels": {},
    }
    for ch in channels:
        print(f"[evaluate] channel={ch}, eps={args.eps}...")
        t0 = time.time()
        stats = evaluate_channel(algo, args.task, ch, args.eps, args.n_eval_episodes, args.seed)
        stats["wall_time_s"] = time.time() - t0
        out["channels"][ch] = stats
        print(f"  ret={stats['ret_mean']:.3f}±{stats['ret_std']:.3f}  "
              f"cost={stats['cost_mean']:.3f}±{stats['cost_std']:.3f}  "
              f"len={stats['len_mean']:.0f}  ({stats['wall_time_s']:.1f}s)")

    out_path = args.out or args.model_path.replace(".model.pt", ".eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[evaluate] saved -> {out_path}")


if __name__ == "__main__":
    main()
