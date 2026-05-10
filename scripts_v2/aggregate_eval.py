"""scripts_v2/aggregate_eval.py — turn eval JSONs into a long-format CSV.

Output columns: task, method, seed, channel, eps, ret_mean, ret_std, cost_mean, cost_std,
                len_mean, n_episodes

Handles both old format ({"channels": {"ch": {ret_mean: scalar}}}) and new v2 format
({"channels": {"ch": {eps: [...], ret_mean: [...]}}}).
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import sys


SEED_RE = re.compile(r"seed(\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="./runs")
    p.add_argument("--out", type=str, default="eval_table.csv")
    p.add_argument("--glob", type=str, default="*.eval_v2.json",
                   help="Filename pattern under runs/<task>/<method>/ (default new v2 format)")
    return p.parse_args()


def extract_seed(path):
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    args = parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "*", "*", args.glob))):
        try:
            with open(f) as fp:
                ev = json.load(fp)
        except Exception as e:
            print(f"[aggregate] skip {f}: {e}", file=sys.stderr)
            continue
        task = ev.get("task")
        method = ev.get("method")
        seed = extract_seed(f)
        channels = ev.get("channels", {})

        for ch, cd in channels.items():
            # New format: dict of arrays keyed eps
            if isinstance(cd.get("ret_mean"), list):
                eps_arr = cd.get("eps", [])
                rets = cd.get("ret_mean", [])
                rsts = cd.get("ret_std", [None] * len(rets))
                costs = cd.get("cost_mean", [None] * len(rets))
                csts = cd.get("cost_std", [None] * len(rets))
                lens = cd.get("len_mean", [None] * len(rets))
                for i in range(len(eps_arr)):
                    rows.append({
                        "task": task, "method": method, "seed": seed, "channel": ch,
                        "eps": eps_arr[i],
                        "ret_mean": rets[i] if i < len(rets) else None,
                        "ret_std": rsts[i] if i < len(rsts) else None,
                        "cost_mean": costs[i] if i < len(costs) else None,
                        "cost_std": csts[i] if i < len(csts) else None,
                        "len_mean": lens[i] if i < len(lens) else None,
                        "n_episodes": ev.get("n_eval_episodes"),
                    })
            else:
                # Old format: scalar at single eps
                rows.append({
                    "task": task, "method": method, "seed": seed, "channel": ch,
                    "eps": ev.get("eps"),
                    "ret_mean": cd.get("ret_mean"),
                    "ret_std": cd.get("ret_std"),
                    "cost_mean": cd.get("cost_mean"),
                    "cost_std": cd.get("cost_std"),
                    "len_mean": cd.get("len_mean"),
                    "n_episodes": cd.get("n_episodes"),
                })

    fieldnames = ["task", "method", "seed", "channel", "eps",
                  "ret_mean", "ret_std", "cost_mean", "cost_std", "len_mean", "n_episodes"]
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
