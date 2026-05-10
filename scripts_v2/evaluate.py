"""scripts_v2/evaluate.py — Multi-eps × multi-channel evaluation with config.yaml.

Differences from scripts/evaluate.py:
- Sweeps multiple eps values per channel in one call (FuzRL Table 3 + ret-vs-eps plot)
- Accepts --config <yaml> with eps_list, channels, n_eval_episodes, seed
- Suppresses per-episode stdout; uses tqdm for progress
- Output JSON is structured: {channels: {ch: {eps_list: [...], ret_mean: [...], ...}}}
  i.e., paired arrays so plot_results can directly draw ret-vs-eps lines.

Usage:
    python scripts_v2/evaluate.py --config configs/eval_default.yaml \\
        --model-path runs/quadrotor_stab/draco/quadrotor_stab_draco_seed0.model.pt \\
        --task quadrotor_stab

    # CLI args override config:
    python scripts_v2/evaluate.py --config configs/eval_default.yaml \\
        --model-path ... --eps-list 0.0,0.1,0.2 --n-eval-episodes 10
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from _common import load_config, merge_args_with_defaults, silence_stdout

from draco.algos.draco import DRACOAlgorithm, MethodConfig
from draco.envs.vec_env import SyncVectorEnv
from draco.envs.toy_env import make_toy_env_fn


PARSER_DEFAULTS = {
    "n_eval_episodes": 30,
    "eps_list": "0.0,0.05,0.10,0.15,0.20",
    "channels": "clean,obs,act,dyn",
    "seed": 1000,
    "device": "auto",
    "out": None,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=["toy", "cartpole_stab", "cartpole_track",
                            "quadrotor_stab", "quadrotor_track"])
    p.add_argument("--n-eval-episodes", type=int, default=PARSER_DEFAULTS["n_eval_episodes"],
                   dest="n_eval_episodes")
    p.add_argument("--eps-list", type=str, default=PARSER_DEFAULTS["eps_list"], dest="eps_list",
                   help="Comma-separated eps values (e.g., '0.0,0.05,0.10').")
    p.add_argument("--channels", type=str, default=PARSER_DEFAULTS["channels"],
                   help="Comma-separated channel names.")
    p.add_argument("--seed", type=int, default=PARSER_DEFAULTS["seed"])
    p.add_argument("--device", type=str, default=PARSER_DEFAULTS["device"])
    p.add_argument("--out", type=str, default=PARSER_DEFAULTS["out"])
    return p.parse_args()


def _safe_obs_dims(task, dims):
    """PAPER_OBS_DIMS for quadrotor_stab in current scg_wrapper has indices up to 10
    but the env actually exposes 6 dims. Filter to in-bounds indices.
    Done inline to avoid modifying scg_wrapper.py (per user instruction)."""
    if dims is None:
        return None
    from draco.envs.scg_wrapper import SCGWrapper
    probe = SCGWrapper(task_name=task, seed=0)
    obs_dim = probe.observation_space.shape[0]
    probe.close() if hasattr(probe, "close") else None
    safe = [d for d in dims if 0 <= d < obs_dim]
    return safe if safe else None  # None = all dims (in scg_wrapper convention)


def make_perturbed_env_fn(task, channel, eps, seed, _safe_dims_cache={}):
    if task == "toy":
        return make_toy_env_fn(seed=seed)

    from draco.envs.scg_wrapper import (
        SCGWrapper, paper_perturbation_kwargs, PAPER_OBS_DIMS, PAPER_DYN_PARAMS,
    )

    if channel == "clean" or eps <= 0.0:
        kwargs = {}
    elif channel == "obs":
        if task not in _safe_dims_cache:
            _safe_dims_cache[task] = _safe_obs_dims(task, PAPER_OBS_DIMS.get(task))
        kwargs = dict(obs_noise_std=eps, obs_noise_dims=_safe_dims_cache[task])
    elif channel == "act":
        kwargs = dict(
            act_impulse_force=eps * 10.0,
            act_impulse_step_offset=20,
            act_impulse_duration=80,
            act_impulse_decay=0.9,
        )
    elif channel == "dyn":
        kwargs = dict(dynamics_noise_std=eps, dynamics_noise_params=PAPER_DYN_PARAMS.get(task))
    else:
        raise ValueError(f"Unknown channel: {channel}")

    def _fn():
        return SCGWrapper(task_name=task, seed=seed, **kwargs)
    return _fn


def evaluate_one(algo, task, channel, eps, n_episodes, seed):
    fn = make_perturbed_env_fn(task, channel, eps, seed)
    env = fn()
    rets, costs, lens = [], [], []
    for ep in range(n_episodes):
        out0 = env.reset(seed=seed + ep) if hasattr(env, "reset") else (env.reset(), {})
        obs = out0[0] if isinstance(out0, tuple) else out0
        if isinstance(obs, tuple):
            obs = obs[0]
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        done = False
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1),
                                    dtype=torch.float32, device=algo.device)
            act = algo.ac.act_deterministic(obs_t)
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            step_out = env.step(act)
            if len(step_out) == 5:
                obs, r, term, trunc, info = step_out
            else:
                obs, r, dn, info = step_out
                term, trunc = dn, False
            ep_ret += float(r)
            ep_cost += float(info.get("cost", info.get("constraint_violation", 0.0)))
            ep_len += 1
            done = bool(term) or bool(trunc)
        rets.append(ep_ret); costs.append(ep_cost); lens.append(ep_len)
    if hasattr(env, "close"):
        env.close()
    return dict(
        ret_mean=float(np.mean(rets)), ret_std=float(np.std(rets)),
        cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs)),
        len_mean=float(np.mean(lens)),
        n_episodes=n_episodes,
    )


def load_algo(model_path, task, device):
    with silence_stdout():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        method_name = ckpt["method"]
        args_dict = ckpt["args"]
        n_shells = args_dict.get("n_shells", 4)
        method = MethodConfig.from_name(method_name, n_shells=n_shells)

        if task == "toy":
            venv = SyncVectorEnv([make_toy_env_fn(seed=0)])
        else:
            from draco.envs.scg_wrapper import SCGWrapper
            def _fn():
                return SCGWrapper(task_name=task, seed=0)
            venv = SyncVectorEnv([_fn])

        def _norm_list(v, cast):
            if isinstance(v, str):
                return tuple(cast(x) for x in v.split(","))
            if isinstance(v, (list, tuple)):
                return tuple(cast(x) for x in v)
            raise TypeError(f"cannot coerce {v!r} to list")
        hidden_sizes = _norm_list(args_dict.get("hidden_sizes", "64,64"), int)
        shell_sigmas = _norm_list(args_dict.get("shell_sigmas", "0.01,0.05,0.10,0.20"), float)

        algo = DRACOAlgorithm(
            venv=venv, method=method, cost_limit=args_dict.get("cost_limit", 25.0),
            seed=args_dict.get("seed", 0), device=device,
            log_dir="/tmp/_draco_eval", run_name="_eval",
            hidden_sizes=hidden_sizes, shell_sigmas=shell_sigmas,
            steps_per_epoch=128,
        )
        algo.ac.load_state_dict(ckpt["ac_state_dict"])
        algo.cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
        if algo.gpd is not None and "gpd_state_dict" in ckpt:
            algo.gpd.load_state_dict(ckpt["gpd_state_dict"])
        if algo.choquet is not None and "choquet_state_dict" in ckpt:
            algo.choquet.load_state_dict(ckpt["choquet_state_dict"])
        venv.close()
    return algo, method_name


def parse_csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    args = merge_args_with_defaults(args, PARSER_DEFAULTS, cfg)

    eps_list = (args.eps_list if isinstance(args.eps_list, list)
                else parse_csv_floats(args.eps_list))
    channels = (args.channels if isinstance(args.channels, list)
                else parse_csv_strs(args.channels))
    if args.task == "toy":
        channels = ["clean"]

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)

    algo, method_name = load_algo(args.model_path, args.task, device)
    algo.ac.eval(); algo.cost_critic.eval()

    out = {
        "model_path": args.model_path, "task": args.task, "method": method_name,
        "n_eval_episodes": args.n_eval_episodes, "seed": args.seed,
        "eps_list": eps_list, "channel_list": channels,
        "channels": {},
    }
    out["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    t_total = time.time()

    total_evals = sum(1 if (ch == "clean" and 0.0 in eps_list) else len(eps_list) for ch in channels)
    # Simpler: just channels × eps_list, but skip non-zero eps for clean (it's the same as eps=0)
    pbar = tqdm(total=len(channels) * len(eps_list), desc=f"eval {method_name}", ncols=100)
    for ch in channels:
        ch_data = {"eps": [], "ret_mean": [], "ret_std": [],
                   "cost_mean": [], "cost_std": [], "len_mean": []}
        for eps in eps_list:
            stats = evaluate_one(algo, args.task, ch, eps, args.n_eval_episodes, args.seed)
            ch_data["eps"].append(eps)
            ch_data["ret_mean"].append(stats["ret_mean"])
            ch_data["ret_std"].append(stats["ret_std"])
            ch_data["cost_mean"].append(stats["cost_mean"])
            ch_data["cost_std"].append(stats["cost_std"])
            ch_data["len_mean"].append(stats["len_mean"])
            pbar.set_postfix({"ch": ch, "eps": f"{eps:.2f}",
                              "ret": f"{stats['ret_mean']:.1f}",
                              "cost": f"{stats['cost_mean']:.2f}"})
            pbar.update(1)
        out["channels"][ch] = ch_data
    pbar.close()
    out["wall_time_s"] = time.time() - t_total

    out_path = args.out or args.model_path.replace(".model.pt", ".eval_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # Single-line summary that's worth printing (one model, one line)
    print(f"[eval_v2] {method_name} <- {os.path.basename(args.model_path)}: "
          f"{len(channels)}ch × {len(eps_list)}eps × {args.n_eval_episodes}ep "
          f"in {out['wall_time_s']:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
