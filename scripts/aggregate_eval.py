"""Aggregate eval JSONs across (task, method, seed) into a CSV.

Reads runs/<task>/<method>/*.eval.json files and emits a wide table:
    task, method, seed, channel, ret_mean, ret_std, cost_mean, cost_std

Plus a method-level aggregated summary (mean across seeds) suitable for the
midterm-defense slide / paper Table 3.

Usage:
    python scripts/aggregate_eval.py --runs-dir runs --out eval_table.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="./runs")
    p.add_argument("--out", type=str, default="eval_table.csv")
    return p.parse_args()


def find_eval_jsons(runs_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(runs_dir, "*", "*", "*.eval.json")))


def main():
    args = parse_args()
    paths = find_eval_jsons(args.runs_dir)
    if not paths:
        print(f"No .eval.json files found under {args.runs_dir}.")
        return 1

    # Detailed (per-seed) rows + aggregated rows
    detailed_rows = []
    grouped = defaultdict(list)  # (task, method, channel) -> list of (ret_mean, cost_mean)

    for p in paths:
        with open(p) as f:
            d = json.load(f)
        task, method = d["task"], d["method"]
        # Try to get seed from file name: <task>_<method>_seed<S>.eval.json
        base = os.path.basename(p)
        seed = "?"
        if "seed" in base:
            try:
                seed = base.split("seed")[1].split(".")[0].split("_")[0]
            except Exception:
                pass
        for ch, stats in d["channels"].items():
            row = dict(
                task=task, method=method, seed=seed, channel=ch,
                eps=d["eps"],
                ret_mean=stats["ret_mean"], ret_std=stats["ret_std"],
                cost_mean=stats["cost_mean"], cost_std=stats["cost_std"],
                len_mean=stats.get("len_mean", float("nan")),
                n_episodes=stats.get("n_episodes"),
            )
            detailed_rows.append(row)
            grouped[(task, method, ch)].append((stats["ret_mean"], stats["cost_mean"]))

    # Write detailed CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
        w.writeheader()
        w.writerows(detailed_rows)
    print(f"[aggregate] detailed rows -> {args.out} ({len(detailed_rows)} rows)")

    # Print aggregated summary table to stdout
    print()
    print(f"{'Task':<18} {'Method':<12} {'Channel':<8} {'AvgRet':>16} {'AvgCost':>16}")
    print("-" * 76)
    import numpy as np
    summary_rows = []
    for (task, method, ch), vals in sorted(grouped.items()):
        rets = np.array([v[0] for v in vals])
        costs = np.array([v[1] for v in vals])
        ret_mean, ret_std = float(rets.mean()), float(rets.std())
        cost_mean, cost_std = float(costs.mean()), float(costs.std())
        print(f"{task:<18} {method:<12} {ch:<8} "
              f"{ret_mean:>+8.2f}±{ret_std:<6.2f} "
              f"{cost_mean:>+8.3f}±{cost_std:<6.3f}")
        summary_rows.append(dict(
            task=task, method=method, channel=ch,
            ret_mean_over_seeds=ret_mean, ret_std_over_seeds=ret_std,
            cost_mean_over_seeds=cost_mean, cost_std_over_seeds=cost_std,
            n_seeds=len(vals),
        ))
    # Save summary too
    summary_path = args.out.replace(".csv", "_summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[aggregate] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
