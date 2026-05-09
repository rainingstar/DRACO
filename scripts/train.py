"""DRACO unified training entry point.

Usage examples:

    # 1. Smoke test on toy env
    python scripts/train.py --task toy --method draco --seed 0 --epochs 10

    # 2. Full run on cartpole_stab (clean training, no perturbation)
    python scripts/train.py --task cartpole_stab --method draco \\
        --seed 0 --total-steps 1000000 --n-envs 8

    # 3. Same on quadrotor_stab
    python scripts/train.py --task quadrotor_stab --method draco \\
        --seed 0 --total-steps 1000000 --n-envs 8

    # 4. Different methods (5-way comparison)
    for METHOD in baseline fuz evo_style dist_only draco; do
        python scripts/train.py --task cartpole_stab --method $METHOD --seed 0
    done

Method aliases:
    baseline    : PPO-Lagrangian (no risk awareness)
    fuz         : Multi-shell Choquet aggregation (no EVT, no IQN)
    evo_style   : Single GPD on scalar V_C  (no IQN, no Choquet)
    dist_only   : IQN cost critic alone (no EVT, no Choquet)
    draco       : Full method
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sure draco is importable when run from project root
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch

from draco.envs.vec_env import SyncVectorEnv
from draco.envs.toy_env import make_toy_env_fn
from draco.algos.draco import DRACOAlgorithm, MethodConfig


def make_env_fn(task: str, seed: int, **task_kwargs):
    """Return a callable that constructs one env instance.

    Available tasks:
        - toy                              : built-in synthetic env (no deps)
        - cartpole_stab / cartpole_track   : safe-control-gym
        - quadrotor_stab / quadrotor_track : safe-control-gym
    """
    if task == "toy":
        return make_toy_env_fn(seed=seed, max_steps=task_kwargs.get("max_steps", 200))

    if task in ("cartpole_stab", "cartpole_track", "quadrotor_stab", "quadrotor_track"):
        # Lazy import so the package works without safe-control-gym installed
        def _fn():
            from draco.envs.scg_wrapper import SCGWrapper
            return SCGWrapper(task_name=task, seed=seed, **task_kwargs)
        return _fn

    raise ValueError(f"Unknown task: {task!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="toy",
                   choices=["toy", "cartpole_stab", "cartpole_track",
                            "quadrotor_stab", "quadrotor_track"])
    p.add_argument("--method", type=str, default="draco",
                   choices=["baseline", "fuz", "evo_style", "dist_only", "draco"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8,
                   help="Number of parallel envs (SyncVectorEnv).")
    p.add_argument("--n-shells", type=int, default=4,
                   help="K in DRACO (multi-source perturbation shells).")
    p.add_argument("--total-steps", type=int, default=200_000,
                   help="Total environment steps across all envs.")
    p.add_argument("--steps-per-epoch", type=int, default=2048)
    p.add_argument("--cost-limit", type=float, default=1.0,
                   help="Episode cost budget. FuzRL paper: 1 for cartpole, 10 "
                        "for quadrotor. Override per task via --cost-limit.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.97)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--v-lr", type=float, default=1e-3)
    p.add_argument("--vc-lr", type=float, default=1e-3)
    p.add_argument("--gpd-lr", type=float, default=1e-3)
    p.add_argument("--penalty-lr", type=float, default=5e-2)
    p.add_argument("--train-pi-iters", type=int, default=40,
                   help="FuzRL paper: 40 for cartpole, 80 for quadrotor.")
    p.add_argument("--train-v-iters", type=int, default=40)
    p.add_argument("--target-kl", type=float, default=0.2,
                   help="FuzRL paper: 0.2 for cartpole, 0.15 for quadrotor. "
                        "Earlier default 0.015 was too tight.")
    p.add_argument("--n-tau-train", type=int, default=32)
    p.add_argument("--n-tau-eval", type=int, default=64)
    p.add_argument("--eta", type=float, default=0.7)
    p.add_argument("--shell-sigmas", type=str, default="0.01,0.05,0.10,0.20",
                   help="Comma-separated shell sigma values; truncated/padded to n-shells.")
    p.add_argument("--hidden-sizes", type=str, default="64,64",
                   help="Comma-separated hidden layer sizes.")
    p.add_argument("--device", type=str, default="auto",
                   help="'cuda', 'cpu', or 'auto' (uses cuda if available).")
    p.add_argument("--log-dir", type=str, default="./runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-model", action="store_true",
                   help="Save trained model state_dicts to <log_dir>/<run_name>/model.pt")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main():
    args = parse_args()
    device = resolve_device(args.device)
    shell_sigmas = tuple(float(x) for x in args.shell_sigmas.split(","))
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))

    method = MethodConfig.from_name(args.method, n_shells=args.n_shells)
    run_name = args.run_name or f"{args.task}_{args.method}_seed{args.seed}_{int(time.time())}"

    log_dir = os.path.join(args.log_dir, args.task, args.method)
    os.makedirs(log_dir, exist_ok=True)

    # Save args for reproducibility
    args_path = os.path.join(log_dir, f"{run_name}.args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build vec env
    print(f"[DRACO train] task={args.task}, method={args.method}, seed={args.seed}, n_envs={args.n_envs}")
    print(f"[DRACO train] device={device}, log_dir={log_dir}, run_name={run_name}")
    env_fns = [make_env_fn(args.task, seed=args.seed + 1000 * i) for i in range(args.n_envs)]
    venv = SyncVectorEnv(env_fns)

    algo = DRACOAlgorithm(
        venv=venv,
        method=method,
        cost_limit=args.cost_limit,
        gamma=args.gamma, lam=args.lam,
        cost_gamma=args.gamma, cost_lam=args.lam,
        steps_per_epoch=args.steps_per_epoch,
        clip_ratio=0.2,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        target_kl=args.target_kl,
        pi_lr=args.pi_lr, v_lr=args.v_lr, vc_lr=args.vc_lr, gpd_lr=args.gpd_lr,
        n_tau_train=args.n_tau_train, n_tau_eval=args.n_tau_eval,
        eta=args.eta,
        shell_sigmas=shell_sigmas,
        penalty_lr=args.penalty_lr,
        seed=args.seed,
        device=device,
        log_dir=log_dir,
        run_name=run_name,
        hidden_sizes=hidden_sizes,
    )

    t0 = time.time()
    algo.learn(total_steps=args.total_steps)
    elapsed = time.time() - t0
    print(f"[DRACO train] DONE in {elapsed:.1f}s ({args.total_steps/elapsed:.0f} env-steps/sec)")

    if args.save_model:
        save_path = os.path.join(log_dir, f"{run_name}.model.pt")
        ckpt = {
            "ac_state_dict": algo.ac.state_dict(),
            "cost_critic_state_dict": algo.cost_critic.state_dict(),
            "args": vars(args),
            "method": args.method,
        }
        if algo.gpd is not None:
            ckpt["gpd_state_dict"] = algo.gpd.state_dict()
        if algo.choquet is not None:
            ckpt["choquet_state_dict"] = algo.choquet.state_dict()
        torch.save(ckpt, save_path)
        print(f"[DRACO train] Model saved -> {save_path}")

    venv.close()


if __name__ == "__main__":
    main()
