"""scripts_v2/train.py — DRACO training with config.yaml + tqdm + quiet logger.

Differences from scripts/train.py:
- Accepts --config <yaml>; CLI args override config; config overrides parser-default
- tqdm progress bar over epochs, with EpRet/EpCost/Lambda postfix
- Logger writes to .txt and .jsonl as before, but no per-epoch stdout spam
- Records start time, elapsed wall time, env-steps/sec at end (one final line)

Usage:
    python scripts_v2/train.py --config configs/quadrotor_stab.yaml \\
        --method draco --seed 0 --save-model

    # Pure CLI also works:
    python scripts_v2/train.py --task quadrotor_stab --method draco \\
        --seed 0 --total-steps 1000000 --save-model
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from tqdm import tqdm

from _common import load_config, merge_args_with_defaults, make_quiet_dump

from draco.envs.vec_env import SyncVectorEnv
from draco.envs.toy_env import make_toy_env_fn
from draco.algos.draco import DRACOAlgorithm, MethodConfig


PARSER_DEFAULTS = {
    "task": "toy", "method": "draco", "seed": 0,
    "n_envs": 8, "n_shells": 4,
    "total_steps": 200_000, "steps_per_epoch": 2048,
    "cost_limit": 1.0,
    "gamma": 0.99, "lam": 0.97,
    "pi_lr": 3e-4, "v_lr": 1e-3, "vc_lr": 1e-3, "gpd_lr": 1e-3,
    "penalty_lr": 5e-2,
    "train_pi_iters": 40, "train_v_iters": 40, "target_kl": 0.2,
    "n_tau_train": 32, "n_tau_eval": 64, "eta": 0.7,
    "shell_sigmas": "0.01,0.05,0.10,0.20",
    "hidden_sizes": "64,64",
    "device": "auto",
    "log_dir": "./runs", "run_name": None,
    "save_model": False,
}


def make_env_fn(task, seed, **task_kwargs):
    if task == "toy":
        return make_toy_env_fn(seed=seed, max_steps=task_kwargs.get("max_steps", 200))
    if task in ("cartpole_stab", "cartpole_track", "quadrotor_stab", "quadrotor_track"):
        def _fn():
            from draco.envs.scg_wrapper import SCGWrapper
            return SCGWrapper(task_name=task, seed=seed, **task_kwargs)
        return _fn
    raise ValueError(f"Unknown task: {task!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--task", type=str, default=PARSER_DEFAULTS["task"],
                   choices=["toy", "cartpole_stab", "cartpole_track",
                            "quadrotor_stab", "quadrotor_track"])
    p.add_argument("--method", type=str, default=PARSER_DEFAULTS["method"],
                   choices=["baseline", "fuz", "evo_style", "dist_only", "draco"])
    p.add_argument("--seed", type=int, default=PARSER_DEFAULTS["seed"])
    p.add_argument("--n-envs", type=int, default=PARSER_DEFAULTS["n_envs"], dest="n_envs")
    p.add_argument("--n-shells", type=int, default=PARSER_DEFAULTS["n_shells"], dest="n_shells")
    p.add_argument("--total-steps", type=int, default=PARSER_DEFAULTS["total_steps"], dest="total_steps")
    p.add_argument("--steps-per-epoch", type=int, default=PARSER_DEFAULTS["steps_per_epoch"], dest="steps_per_epoch")
    p.add_argument("--cost-limit", type=float, default=PARSER_DEFAULTS["cost_limit"], dest="cost_limit")
    p.add_argument("--gamma", type=float, default=PARSER_DEFAULTS["gamma"])
    p.add_argument("--lam", type=float, default=PARSER_DEFAULTS["lam"])
    p.add_argument("--pi-lr", type=float, default=PARSER_DEFAULTS["pi_lr"], dest="pi_lr")
    p.add_argument("--v-lr", type=float, default=PARSER_DEFAULTS["v_lr"], dest="v_lr")
    p.add_argument("--vc-lr", type=float, default=PARSER_DEFAULTS["vc_lr"], dest="vc_lr")
    p.add_argument("--gpd-lr", type=float, default=PARSER_DEFAULTS["gpd_lr"], dest="gpd_lr")
    p.add_argument("--penalty-lr", type=float, default=PARSER_DEFAULTS["penalty_lr"], dest="penalty_lr")
    p.add_argument("--train-pi-iters", type=int, default=PARSER_DEFAULTS["train_pi_iters"], dest="train_pi_iters")
    p.add_argument("--train-v-iters", type=int, default=PARSER_DEFAULTS["train_v_iters"], dest="train_v_iters")
    p.add_argument("--target-kl", type=float, default=PARSER_DEFAULTS["target_kl"], dest="target_kl")
    p.add_argument("--n-tau-train", type=int, default=PARSER_DEFAULTS["n_tau_train"], dest="n_tau_train")
    p.add_argument("--n-tau-eval", type=int, default=PARSER_DEFAULTS["n_tau_eval"], dest="n_tau_eval")
    p.add_argument("--eta", type=float, default=PARSER_DEFAULTS["eta"])
    p.add_argument("--shell-sigmas", type=str, default=PARSER_DEFAULTS["shell_sigmas"], dest="shell_sigmas")
    p.add_argument("--hidden-sizes", type=str, default=PARSER_DEFAULTS["hidden_sizes"], dest="hidden_sizes")
    p.add_argument("--device", type=str, default=PARSER_DEFAULTS["device"])
    p.add_argument("--log-dir", type=str, default=PARSER_DEFAULTS["log_dir"], dest="log_dir")
    p.add_argument("--run-name", type=str, default=PARSER_DEFAULTS["run_name"], dest="run_name")
    p.add_argument("--save-model", action="store_true", dest="save_model")
    return p.parse_args()


def normalize_list_arg(v, cast):
    """Accept either a CSV string or already-a-list (from YAML)."""
    if isinstance(v, str):
        return tuple(cast(x) for x in v.split(","))
    if isinstance(v, (list, tuple)):
        return tuple(cast(x) for x in v)
    raise TypeError(f"Cannot coerce {v!r} to list")


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    args = merge_args_with_defaults(args, PARSER_DEFAULTS, cfg)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)
    shell_sigmas = normalize_list_arg(args.shell_sigmas, float)
    hidden_sizes = normalize_list_arg(args.hidden_sizes, int)

    method = MethodConfig.from_name(args.method, n_shells=args.n_shells)
    run_name = args.run_name or f"{args.task}_{args.method}_seed{args.seed}_{int(time.time())}"
    log_dir = os.path.join(args.log_dir, args.task, args.method)
    os.makedirs(log_dir, exist_ok=True)

    args_path = os.path.join(log_dir, f"{run_name}.args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # One-line start banner. Then quiet.
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[train_v2] start={start_ts} task={args.task} method={args.method} "
          f"seed={args.seed} steps={args.total_steps} run={run_name}")

    env_fns = [make_env_fn(args.task, seed=args.seed + 1000 * i) for i in range(args.n_envs)]
    venv = SyncVectorEnv(env_fns)
    algo = DRACOAlgorithm(
        venv=venv, method=method,
        cost_limit=args.cost_limit,
        gamma=args.gamma, lam=args.lam,
        cost_gamma=args.gamma, cost_lam=args.lam,
        steps_per_epoch=args.steps_per_epoch, clip_ratio=0.2,
        train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters,
        target_kl=args.target_kl,
        pi_lr=args.pi_lr, v_lr=args.v_lr, vc_lr=args.vc_lr, gpd_lr=args.gpd_lr,
        n_tau_train=args.n_tau_train, n_tau_eval=args.n_tau_eval,
        eta=args.eta, shell_sigmas=shell_sigmas,
        penalty_lr=args.penalty_lr, seed=args.seed, device=device,
        log_dir=log_dir, run_name=run_name, hidden_sizes=hidden_sizes,
    )

    # Hook the logger: write to file as usual, no stdout, update tqdm pbar.
    total_epochs = max(1, args.total_steps // args.steps_per_epoch)
    pbar = tqdm(total=total_epochs, desc=f"{args.method}/seed{args.seed}",
                ncols=110, dynamic_ncols=False, file=sys.stderr)
    from draco.utils.logger import Logger
    Logger.dump = make_quiet_dump(pbar=pbar)

    t0 = time.time()
    algo.learn(total_steps=args.total_steps)
    elapsed = time.time() - t0
    pbar.close()

    if args.save_model:
        save_path = os.path.join(log_dir, f"{run_name}.model.pt")
        ckpt = {
            "ac_state_dict": algo.ac.state_dict(),
            "cost_critic_state_dict": algo.cost_critic.state_dict(),
            "args": vars(args), "method": args.method,
        }
        if algo.gpd is not None:
            ckpt["gpd_state_dict"] = algo.gpd.state_dict()
        if algo.choquet is not None:
            ckpt["choquet_state_dict"] = algo.choquet.state_dict()
        torch.save(ckpt, save_path)

    end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sps = args.total_steps / elapsed if elapsed > 0 else 0
    print(f"[train_v2] DONE end={end_ts} elapsed={elapsed:.0f}s "
          f"steps/sec={sps:.0f} model={'saved' if args.save_model else 'not_saved'}")
    venv.close()


if __name__ == "__main__":
    main()
