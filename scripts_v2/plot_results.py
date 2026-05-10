"""scripts_v2/plot_results.py — training curves + eval-sweep plots, config.yaml-driven.

Two modes (selectable via --mode or YAML key 'mode'):
- learning_curves : EpRet/EpCost/Lambda vs env steps (existing behavior)
- eval_sweep      : ret-vs-eps and cost-vs-eps per channel, all methods overlaid
                    (the answer to project-status §5 P2 "ret-vs-eps 答辩金牌图")

Usage:
    python scripts_v2/plot_results.py --config configs/plot_default.yaml \\
        --mode learning_curves --runs-dir runs --out figs/curves.png

    python scripts_v2/plot_results.py --config configs/plot_default.yaml \\
        --mode eval_sweep --runs-dir runs --out figs/eval_sweep.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed.", file=sys.stderr)
    sys.exit(1)

from _common import load_config, merge_args_with_defaults


METHOD_ORDER = ["baseline", "fuz", "evo_style", "dist_only", "draco"]
METHOD_COLORS = {
    "baseline": "#7f7f7f", "fuz": "#1f77b4", "evo_style": "#2ca02c",
    "dist_only": "#ff7f0e", "draco": "#d62728",
}
METHOD_LABELS = {
    "baseline": "PPO-Lag (baseline)", "fuz": "Fuz-PPO-Lag",
    "evo_style": "EVO-style", "dist_only": "Dist-Only (IQN)",
    "draco": "DRACO (ours)",
}


PARSER_DEFAULTS = {
    "mode": "learning_curves",
    "runs_dir": "./runs",
    "tasks": None,
    "methods": None,
    "metrics": "EpRet/mean,EpCost/mean,Lambda/mean",
    "x_key": "TotalEnvSteps/mean",
    "out": "draco_curves.png",
    "smooth": 5,
    "max_points": 200,
    "eval_glob": "*.eval_v2.json",
    "channels": "clean,obs,act,dyn",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default=PARSER_DEFAULTS["mode"],
                   choices=["learning_curves", "eval_sweep"])
    p.add_argument("--runs-dir", type=str, default=PARSER_DEFAULTS["runs_dir"], dest="runs_dir")
    p.add_argument("--tasks", type=str, default=PARSER_DEFAULTS["tasks"])
    p.add_argument("--methods", type=str, default=PARSER_DEFAULTS["methods"])
    p.add_argument("--metrics", type=str, default=PARSER_DEFAULTS["metrics"])
    p.add_argument("--x-key", type=str, default=PARSER_DEFAULTS["x_key"], dest="x_key")
    p.add_argument("--out", type=str, default=PARSER_DEFAULTS["out"])
    p.add_argument("--smooth", type=int, default=PARSER_DEFAULTS["smooth"])
    p.add_argument("--max-points", type=int, default=PARSER_DEFAULTS["max_points"], dest="max_points")
    p.add_argument("--eval-glob", type=str, default=PARSER_DEFAULTS["eval_glob"], dest="eval_glob")
    p.add_argument("--channels", type=str, default=PARSER_DEFAULTS["channels"])
    return p.parse_args()


def to_list(v, cast=str):
    if v is None or v == "None":
        return None
    if isinstance(v, list):
        return [cast(x) for x in v]
    return [cast(x.strip()) for x in str(v).split(",") if x.strip()]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def discover_train_runs(runs_dir, tasks=None, methods=None):
    out = defaultdict(list)
    if not os.path.isdir(runs_dir):
        return out
    for task in sorted(os.listdir(runs_dir)):
        tp = os.path.join(runs_dir, task)
        if not os.path.isdir(tp): continue
        if tasks and task not in tasks: continue
        for m in sorted(os.listdir(tp)):
            mp = os.path.join(tp, m)
            if not os.path.isdir(mp): continue
            if methods and m not in methods: continue
            jsonls = sorted(glob.glob(os.path.join(mp, "*.jsonl")))
            if jsonls:
                out[(task, m)] = jsonls
    return out


def discover_eval_files(runs_dir, eval_glob, tasks=None, methods=None):
    """Return {task: {method: [list of eval json files]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for task in sorted(os.listdir(runs_dir) if os.path.isdir(runs_dir) else []):
        tp = os.path.join(runs_dir, task)
        if not os.path.isdir(tp): continue
        if tasks and task not in tasks: continue
        for m in sorted(os.listdir(tp)):
            mp = os.path.join(tp, m)
            if not os.path.isdir(mp): continue
            if methods and m not in methods: continue
            files = sorted(glob.glob(os.path.join(mp, eval_glob)))
            if files:
                out[task][m] = files
    return out


def smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[: len(x)]


def collect_curves(jsonl_paths, x_key, y_key):
    xs_list, ys_list = [], []
    for p in jsonl_paths:
        rows = load_jsonl(p)
        keep = [(r.get(x_key), r.get(y_key)) for r in rows
                if r.get(x_key) is not None and r.get(y_key) is not None]
        if len(keep) < 2: continue
        xs_list.append(np.array([k[0] for k in keep], dtype=np.float64))
        ys_list.append(np.array([k[1] for k in keep], dtype=np.float64))
    if not xs_list:
        return np.array([]), np.array([])
    x_max = min(x.max() for x in xs_list)
    x_grid = np.linspace(min(x.min() for x in xs_list), x_max, 100)
    y_curves = []
    for xs, ys in zip(xs_list, ys_list):
        idx = np.argsort(xs); xs, ys = xs[idx], ys[idx]
        y_curves.append(np.interp(x_grid, xs, ys, left=ys[0], right=ys[-1]))
    return x_grid, np.array(y_curves)


def plot_learning_curves(args):
    runs = discover_train_runs(args.runs_dir, tasks=to_list(args.tasks),
                                methods=to_list(args.methods))
    if not runs:
        print(f"No training runs in {args.runs_dir}.", file=sys.stderr)
        return 1
    metrics = to_list(args.metrics)
    per_task = defaultdict(dict)
    for (task, m), paths in runs.items():
        per_task[task][m] = paths
    tasks = sorted(per_task.keys())
    n_metrics, n_tasks = len(metrics), len(tasks)
    fig, axes = plt.subplots(n_metrics, n_tasks,
                             figsize=(5 * n_tasks, 3.2 * n_metrics),
                             squeeze=False, sharex=True)
    fig.suptitle("DRACO learning curves", fontsize=14)

    for j, task in enumerate(tasks):
        method_data = per_task[task]
        sorted_m = [m for m in METHOD_ORDER if m in method_data]
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for m in sorted_m:
                paths = method_data[m]
                x, ys = collect_curves(paths, args.x_key, metric)
                if len(x) == 0: continue
                ys_s = np.array([smooth(y, args.smooth) for y in ys])
                mean, std = ys_s.mean(0), ys_s.std(0)
                color = METHOD_COLORS.get(m, "C0")
                label = METHOD_LABELS.get(m, m)
                if len(paths) > 1: label = f"{label} (n={len(paths)})"
                ax.plot(x, mean, color=color, label=label,
                        linewidth=2 if m == "draco" else 1.5)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            if i == 0: ax.set_title(task, fontsize=10)
            ax.set_ylabel(metric.replace("/mean", ""), fontsize=10)
            if i == n_metrics - 1: ax.set_xlabel("env steps", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 0 and j == n_tasks - 1:
                ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_v2] learning_curves -> {args.out}")
    return 0


def plot_eval_sweep(args):
    """Plot ret-vs-eps and cost-vs-eps per channel × method.

    Layout: 2 rows (return, cost) × N columns (channels). One method = one line.
    Multiple seeds = mean ± std across seeds at each eps.
    """
    eval_files = discover_eval_files(args.runs_dir, args.eval_glob,
                                     tasks=to_list(args.tasks),
                                     methods=to_list(args.methods))
    if not eval_files:
        print(f"No eval files matching {args.eval_glob} in {args.runs_dir}.", file=sys.stderr)
        return 1
    channels = to_list(args.channels)
    tasks = sorted(eval_files.keys())

    for task in tasks:
        method_files = eval_files[task]
        sorted_m = [m for m in METHOD_ORDER if m in method_files]
        if not sorted_m:
            continue

        # Collect: data[m][ch] = (eps_array, ret_means_seeds, cost_means_seeds)
        data = {}
        for m in sorted_m:
            data[m] = {}
            seed_evals = [json.load(open(f)) for f in method_files[m]]
            if not seed_evals:
                continue
            for ch in channels:
                ret_seeds, cost_seeds, eps_arr = [], [], None
                for ev in seed_evals:
                    if ch not in ev.get("channels", {}):
                        continue
                    cd = ev["channels"][ch]
                    if eps_arr is None:
                        eps_arr = np.array(cd["eps"], dtype=float)
                    ret_seeds.append(np.array(cd["ret_mean"], dtype=float))
                    cost_seeds.append(np.array(cd["cost_mean"], dtype=float))
                if eps_arr is not None and ret_seeds:
                    data[m][ch] = (eps_arr, np.array(ret_seeds), np.array(cost_seeds))

        n_ch = len(channels)
        fig, axes = plt.subplots(2, n_ch, figsize=(4 * n_ch, 6), squeeze=False)
        fig.suptitle(f"DRACO eval sweep — {task}", fontsize=13)

        for j, ch in enumerate(channels):
            for m in sorted_m:
                if ch not in data.get(m, {}):
                    continue
                eps, rets, costs = data[m][ch]
                color = METHOD_COLORS.get(m, "C0")
                label = METHOD_LABELS.get(m, m)
                n_s = rets.shape[0]
                if n_s > 1:
                    label = f"{label} (n={n_s})"
                # ret
                rm, rs = rets.mean(0), rets.std(0)
                axes[0, j].plot(eps, rm, color=color, label=label,
                                linewidth=2 if m == "draco" else 1.5, marker="o")
                axes[0, j].fill_between(eps, rm - rs, rm + rs, color=color, alpha=0.18)
                # cost
                cm, cs = costs.mean(0), costs.std(0)
                axes[1, j].plot(eps, cm, color=color,
                                linewidth=2 if m == "draco" else 1.5, marker="o")
                axes[1, j].fill_between(eps, cm - cs, cm + cs, color=color, alpha=0.18)

            axes[0, j].set_title(f"channel: {ch}", fontsize=11)
            axes[0, j].set_ylabel("Return", fontsize=10)
            axes[0, j].grid(alpha=0.3)
            axes[1, j].set_xlabel("eps (perturbation magnitude)", fontsize=10)
            axes[1, j].set_ylabel("Cost", fontsize=10)
            axes[1, j].grid(alpha=0.3)

        axes[0, -1].legend(fontsize=8, loc="best")
        fig.tight_layout()

        out_path = args.out.replace(".png", f"_{task}.png") if len(tasks) > 1 else args.out
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_v2] eval_sweep[{task}] -> {out_path}")
    return 0


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    args = merge_args_with_defaults(args, PARSER_DEFAULTS, cfg)

    if args.mode == "learning_curves":
        return plot_learning_curves(args)
    elif args.mode == "eval_sweep":
        return plot_eval_sweep(args)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
