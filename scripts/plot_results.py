"""Plot training curves (shaded bands across seeds) for DRACO methods.

Reads JSONL logs produced by train.py and renders learning curves with
mean ± std bands. Designed for the midterm-defense slide (Return + Cost
+ Lambda × 5 methods on the same x axis).

Usage:
    # Plot all methods on cartpole_stab (auto-discovers seeds from runs/)
    python scripts/plot_results.py \\
        --runs-dir runs/cartpole_stab \\
        --out cartpole_stab.png

    # Compare just two methods, both tasks side by side
    python scripts/plot_results.py \\
        --runs-dir runs --tasks cartpole_stab quadrotor_stab \\
        --methods baseline draco --out compare.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


METHOD_ORDER = ["baseline", "fuz", "evo_style", "dist_only", "draco"]
METHOD_COLORS = {
    "baseline":  "#7f7f7f",  # gray
    "fuz":       "#1f77b4",  # blue
    "evo_style": "#2ca02c",  # green
    "dist_only": "#ff7f0e",  # orange
    "draco":     "#d62728",  # red (highlight)
}
METHOD_LABELS = {
    "baseline":  "PPO-Lag (baseline)",
    "fuz":       "Fuz-PPO-Lag (sigmoid)",
    "evo_style": "EVO-style",
    "dist_only": "Dist-Only (IQN)",
    "draco":     "DRACO (ours)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="./runs",
                   help="Top-level directory containing <task>/<method>/*.jsonl files.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Restrict to these tasks (default: auto-discover).")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Restrict to these methods (default: all 5).")
    p.add_argument("--metrics", nargs="+",
                   default=["EpRet/mean", "EpCost/mean", "Lambda/mean"],
                   help="JSONL keys to plot (one subplot each).")
    p.add_argument("--x-key", type=str, default="TotalEnvSteps/mean",
                   help="JSONL key for x axis.")
    p.add_argument("--out", type=str, default="draco_curves.png")
    p.add_argument("--smooth", type=int, default=5,
                   help="Moving-average window for smoothing curves.")
    p.add_argument("--max-points", type=int, default=200,
                   help="Downsample to at most this many points per curve.")
    return p.parse_args()


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def discover_runs(runs_dir: str, tasks=None, methods=None) -> Dict[Tuple[str, str], List[str]]:
    """Returns {(task, method): [list of jsonl paths]}."""
    out = defaultdict(list)
    if not os.path.isdir(runs_dir):
        return out
    for task_dir in sorted(os.listdir(runs_dir)):
        task_path = os.path.join(runs_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        if tasks and task_dir not in tasks:
            continue
        for method_dir in sorted(os.listdir(task_path)):
            method_path = os.path.join(task_path, method_dir)
            if not os.path.isdir(method_path):
                continue
            if methods and method_dir not in methods:
                continue
            jsonls = sorted(glob.glob(os.path.join(method_path, "*.jsonl")))
            if jsonls:
                out[(task_dir, method_dir)] = jsonls
    return out


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    pad = window // 2
    x_pad = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")[: len(x)]


def collect_curves(jsonl_paths: List[str], x_key: str, y_key: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, y_curves) where y_curves: (n_seeds, n_points) interpolated to common x grid."""
    xs_list, ys_list = [], []
    for p in jsonl_paths:
        rows = load_jsonl(p)
        xs = [r.get(x_key) for r in rows]
        ys = [r.get(y_key) for r in rows]
        # filter Nones
        keep = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(keep) < 2:
            continue
        xs_arr = np.array([k[0] for k in keep], dtype=np.float64)
        ys_arr = np.array([k[1] for k in keep], dtype=np.float64)
        xs_list.append(xs_arr); ys_list.append(ys_arr)
    if not xs_list:
        return np.array([]), np.array([])

    # Common x grid: union of all sample points, capped at max range
    x_max = min(x.max() for x in xs_list)
    x_grid = np.linspace(min(x.min() for x in xs_list), x_max, 100)
    y_curves = []
    for xs_arr, ys_arr in zip(xs_list, ys_list):
        # Sort by x (jsonl should be in order but be safe)
        idx = np.argsort(xs_arr)
        xs_arr, ys_arr = xs_arr[idx], ys_arr[idx]
        y_interp = np.interp(x_grid, xs_arr, ys_arr,
                              left=ys_arr[0], right=ys_arr[-1])
        y_curves.append(y_interp)
    return x_grid, np.array(y_curves)


def main():
    args = parse_args()
    runs = discover_runs(args.runs_dir, tasks=args.tasks, methods=args.methods)
    if not runs:
        print(f"No runs found in {args.runs_dir}. Check the directory.")
        return 1

    # Group by task -> method
    per_task = defaultdict(dict)
    for (task, method), paths in runs.items():
        per_task[task][method] = paths
    tasks = sorted(per_task.keys())

    n_metrics = len(args.metrics)
    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        n_metrics, n_tasks,
        figsize=(5 * n_tasks, 3.2 * n_metrics),
        squeeze=False, sharex=True,
    )
    fig.suptitle("DRACO learning curves", fontsize=14)

    for j, task in enumerate(tasks):
        method_data = per_task[task]
        sorted_methods = [m for m in METHOD_ORDER if m in method_data]
        for i, metric in enumerate(args.metrics):
            ax = axes[i, j]
            for m in sorted_methods:
                paths = method_data[m]
                x, ys = collect_curves(paths, args.x_key, metric)
                if len(x) == 0:
                    continue
                # Smooth each seed curve
                ys_smooth = np.array([smooth(y, args.smooth) for y in ys])
                mean = ys_smooth.mean(axis=0)
                std = ys_smooth.std(axis=0)
                color = METHOD_COLORS.get(m, "C0")
                label = METHOD_LABELS.get(m, m)
                if len(paths) > 1:
                    label = f"{label} (n={len(paths)})"
                ax.plot(x, mean, color=color, label=label,
                        linewidth=2 if m == "draco" else 1.5)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            ax.set_title(f"{task}", fontsize=10) if i == 0 else None
            ylabel = metric.replace("/mean", "")
            ax.set_ylabel(ylabel, fontsize=10)
            if i == n_metrics - 1:
                ax.set_xlabel("env steps", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 0 and j == n_tasks - 1:
                ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_path = args.out
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
