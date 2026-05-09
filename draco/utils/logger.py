"""Lightweight logger for DRACO experiments. Writes to stdout + JSONL file.

Tracks running statistics per epoch. Designed to be drop-in: very few methods,
no external deps.
"""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict
from typing import Any


class Logger:
    def __init__(self, log_dir: str, run_name: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.run_name = run_name
        self.start_time = time.time()
        self.jsonl_path = os.path.join(log_dir, f"{run_name}.jsonl")
        self.txt_path = os.path.join(log_dir, f"{run_name}.txt")
        # Per-epoch buffer: list of values to be aggregated.
        self.cur: dict[str, list[float]] = defaultdict(list)
        self.epoch_count = 0
        with open(self.jsonl_path, "w") as f:
            pass  # truncate
        with open(self.txt_path, "w") as f:
            f.write(f"# DRACO run: {run_name} started {time.ctime(self.start_time)}\n")

    def store(self, **kvs):
        """Add one or more scalars to the current epoch buffer."""
        for k, v in kvs.items():
            try:
                self.cur[k].append(float(v))
            except (TypeError, ValueError):
                pass  # ignore non-scalar

    def get_stats(self, key: str):
        """Return mean of values for the key in current epoch (None if empty)."""
        v = self.cur.get(key, [])
        if not v:
            return None
        return sum(v) / len(v)

    def dump(self, epoch: int, extra: dict[str, Any] | None = None) -> dict[str, float]:
        """Aggregate current epoch buffer (mean per key), append to JSONL, print summary."""
        agg: dict[str, float] = {"epoch": epoch, "wall_time": time.time() - self.start_time}
        for k, vs in self.cur.items():
            if not vs:
                continue
            agg[f"{k}/mean"] = sum(vs) / len(vs)
            agg[f"{k}/min"] = min(vs)
            agg[f"{k}/max"] = max(vs)
        if extra:
            agg.update(extra)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(agg) + "\n")

        # Pretty print to stdout: a few key fields only
        msg = f"[ep {epoch:>4d} | t {agg['wall_time']:6.0f}s]"
        for k in ("EpRet/mean", "EpCost/mean", "EpLen/mean", "Lambda/mean",
                  "LossPi/mean", "LossV/mean", "LossVc/mean", "LossGPD/mean"):
            if k in agg:
                msg += f" {k.split('/')[0]}={agg[k]:.3f}"
        print(msg)
        with open(self.txt_path, "a") as f:
            f.write(msg + "\n")

        self.cur.clear()
        self.epoch_count = epoch
        return agg
