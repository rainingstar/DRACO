"""scripts_v2/summarize_eval.py — produce a Markdown summary table from eval_v2 JSONs.

Used by the overnight finalize step to populate reviews/seed0-analysis.md.

Reads ~/draco_safe_rl/runs/<task>/<method>/*.eval_v2.json and produces:
- Per-channel ret table at given eps (mean ± std across seeds)
- Per-channel cost table at same eps
- Δ vs baseline row
- GO/No-go judgement

Usage:
    python scripts_v2/summarize_eval.py --runs-dir runs --eps 0.10 --out summary.md
    python scripts_v2/summarize_eval.py --runs-dir runs --eps 0.10 --json   # JSON output
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


METHOD_ORDER = ["baseline", "fuz", "evo_style", "dist_only", "draco"]
SEED_RE = re.compile(r"seed(\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="./runs")
    p.add_argument("--task", default="quadrotor_stab")
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--channels", default="clean,obs,act,dyn")
    p.add_argument("--out", default=None, help="If set, write Markdown to this path")
    p.add_argument("--json", action="store_true", help="Also print JSON dump to stdout")
    return p.parse_args()


def collect(runs_dir, task, channels, eps):
    """Returns {method: {channel: {ret_means: [seed-list], cost_means: [...]}}}"""
    pattern = os.path.join(runs_dir, task, "*", "*.eval_v2.json")
    out = defaultdict(lambda: defaultdict(lambda: {"ret": [], "cost": [], "seeds": []}))
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                ev = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        method = ev.get("method", "?")
        seed_m = SEED_RE.search(os.path.basename(path))
        seed = int(seed_m.group(1)) if seed_m else -1
        for ch in channels:
            cd = ev.get("channels", {}).get(ch)
            if not cd:
                continue
            eps_arr = cd.get("eps", [])
            try:
                idx = next(i for i, e in enumerate(eps_arr) if abs(float(e) - eps) < 1e-9)
            except StopIteration:
                continue
            out[method][ch]["ret"].append(cd["ret_mean"][idx])
            out[method][ch]["cost"].append(cd["cost_mean"][idx])
            out[method][ch]["seeds"].append(seed)
    return out


def stat(vals):
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, var ** 0.5


def fmt(m, s):
    if m is None:
        return "—"
    if s is None:
        return f"{m:.1f}"
    return f"{m:.1f}±{s:.1f}"


def make_md(data, channels, eps, task):
    lines = []
    lines.append(f"## Eval @ eps={eps:.2f} on {task}\n")
    lines.append(f"Mean ± std across seeds. n=#seeds with valid eval data.\n")

    # Return table
    lines.append("### Return\n")
    head = "| method (n) | " + " | ".join(channels) + " |"
    sep = "|---|" + "---|" * len(channels)
    lines.append(head)
    lines.append(sep)
    base_ret = {}
    for m in METHOD_ORDER:
        if m not in data:
            continue
        ch_means = []
        n_seeds = max(len(data[m][ch]["ret"]) for ch in channels) if any(ch in data[m] for ch in channels) else 0
        row = [f"{m} (n={n_seeds})"]
        for ch in channels:
            vals = data[m][ch]["ret"] if ch in data[m] else []
            mean, std = stat(vals)
            if m == "baseline":
                base_ret[ch] = mean
            row.append(fmt(mean, std))
        lines.append("| " + " | ".join(row) + " |")

    # Delta row
    if "draco" in data and "baseline" in data:
        delta_row = ["**Δ(draco - baseline)**"]
        for ch in channels:
            d_vals = data["draco"][ch]["ret"] if ch in data["draco"] else []
            b_vals = data["baseline"][ch]["ret"] if ch in data["baseline"] else []
            d_m, _ = stat(d_vals); b_m, _ = stat(b_vals)
            if d_m is not None and b_m is not None:
                delta = d_m - b_m
                delta_row.append(f"{delta:+.1f}")
            else:
                delta_row.append("—")
        lines.append("| " + " | ".join(delta_row) + " |")
    lines.append("")

    # Cost table
    lines.append("### Cost\n")
    lines.append(head); lines.append(sep)
    for m in METHOD_ORDER:
        if m not in data:
            continue
        n_seeds = max(len(data[m][ch]["cost"]) for ch in channels) if any(ch in data[m] for ch in channels) else 0
        row = [f"{m} (n={n_seeds})"]
        for ch in channels:
            vals = data[m][ch]["cost"] if ch in data[m] else []
            mean, std = stat(vals)
            row.append(fmt(mean, std))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # GO/No-go judgement
    lines.append("### GO/No-go signal\n")
    if "draco" in data and "baseline" in data:
        wins = 0; ties = 0; total = 0
        for ch in channels:
            if ch == "clean":
                continue
            d_vals = data["draco"][ch]["ret"] if ch in data["draco"] else []
            b_vals = data["baseline"][ch]["ret"] if ch in data["baseline"] else []
            d_m, d_s = stat(d_vals); b_m, b_s = stat(b_vals)
            if d_m is None or b_m is None:
                continue
            total += 1
            margin = (d_s or 0) + (b_s or 0)
            if d_m > b_m + margin:
                wins += 1
                lines.append(f"- ✅ **{ch}**: DRACO {d_m:.1f}±{d_s:.1f} > baseline {b_m:.1f}±{b_s:.1f} "
                             f"(Δ={d_m-b_m:+.1f}, margin {margin:.1f}) — STAT SIG WIN")
            elif d_m > b_m:
                ties += 1
                lines.append(f"- ⚖️ **{ch}**: DRACO {d_m:.1f}±{d_s:.1f} ≳ baseline {b_m:.1f}±{b_s:.1f} "
                             f"(Δ={d_m-b_m:+.1f}, margin {margin:.1f}) — soft win, within margin")
            else:
                lines.append(f"- ❌ **{ch}**: DRACO {d_m:.1f}±{d_s:.1f} ≤ baseline {b_m:.1f}±{b_s:.1f} "
                             f"(Δ={d_m-b_m:+.1f}) — LOSS")
        if wins >= 2:
            lines.append(f"\n**Verdict: GO** (DRACO statsig wins {wins}/{total} perturbation channels)")
        elif wins + ties >= 2:
            lines.append(f"\n**Verdict: WEAK GO** ({wins} statsig + {ties} soft wins of {total}) — "
                         "consider reporting ret-vs-eps curves rather than single-eps tables")
        else:
            lines.append(f"\n**Verdict: PIVOT to P1 (obs-side augmentation)** — "
                         f"only {wins} stat-sig wins of {total} channels")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    data = collect(args.runs_dir, args.task, channels, args.eps)
    md = make_md(data, channels, args.eps, args.task)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"[summarize] wrote {args.out}")
    else:
        print(md)
    if args.json:
        print(json.dumps(
            {m: {ch: {**vals, "ret_mean": stat(vals["ret"])[0],
                                "cost_mean": stat(vals["cost"])[0]}
                 for ch, vals in data[m].items()}
             for m in data}, indent=2, default=str))


if __name__ == "__main__":
    main()
