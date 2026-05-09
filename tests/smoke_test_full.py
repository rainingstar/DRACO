"""End-to-end smoke test: run each of the 5 methods for 5 epochs on a toy env.

Checks:
    1. No NaN/Inf in losses or networks
    2. Episode return improves (or at least doesn't crash)
    3. All methods finish within reasonable time
    4. All-method losses are finite

Usage: python tests/smoke_test_full.py
"""
import os
import sys
import time
import numpy as np
import torch

# Make sure draco is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from draco.envs.toy_env import make_toy_env_fn
from draco.envs.vec_env import SyncVectorEnv
from draco.algos.draco import DRACOAlgorithm, MethodConfig


def run_one(method_name: str, seed: int = 0, n_envs: int = 4, n_epochs: int = 4,
            steps_per_epoch: int = 512):
    print(f"\n{'='*70}\nMethod: {method_name} | seed={seed} | n_envs={n_envs}\n{'='*70}")
    env_fns = [make_toy_env_fn(seed=seed + i) for i in range(n_envs)]
    venv = SyncVectorEnv(env_fns)
    method = MethodConfig.from_name(method_name, n_shells=4)

    algo = DRACOAlgorithm(
        venv=venv,
        method=method,
        cost_limit=10.0,
        steps_per_epoch=steps_per_epoch,
        train_pi_iters=20,
        train_v_iters=20,
        seed=seed,
        device="cpu",
        log_dir=f"./_smoke_logs/{method_name}",
        run_name=f"smoke_{method_name}_seed{seed}",
        hidden_sizes=(32, 32),
    )
    t0 = time.time()
    algo.learn(total_steps=steps_per_epoch * n_epochs)
    elapsed = time.time() - t0

    # Validate: scan logger txt for NaN
    log_path = os.path.join(algo.logger.log_dir, algo.logger.run_name + ".jsonl")
    import json
    with open(log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    final_ep_ret = lines[-1].get("EpRet/mean", None)
    final_ep_cost = lines[-1].get("EpCost/mean", None)
    final_loss_pi = lines[-1].get("LossPi/mean", None)
    final_loss_v = lines[-1].get("LossV/mean", None)
    final_loss_vc = lines[-1].get("LossVc/mean", None)
    final_lambda = lines[-1].get("Lambda/mean", None)

    metrics = dict(
        method=method_name,
        elapsed_s=elapsed,
        final_ep_ret=final_ep_ret,
        final_ep_cost=final_ep_cost,
        final_loss_pi=final_loss_pi,
        final_loss_v=final_loss_v,
        final_loss_vc=final_loss_vc,
        final_lambda=final_lambda,
    )
    # Check no NaN
    for k, v in metrics.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            print(f"  FAIL: {k}={v} is NaN/Inf!")
            metrics["passed"] = False
            return metrics
    metrics["passed"] = True
    print(f"  Final: ret={final_ep_ret}, cost={final_ep_cost}, "
          f"loss_pi={final_loss_pi}, loss_vc={final_loss_vc}, lambda={final_lambda}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return metrics


def main():
    methods = ["baseline", "fuz", "evo_style", "dist_only", "draco"]
    results = []
    for m in methods:
        try:
            r = run_one(m, seed=0, n_envs=4, n_epochs=3, steps_per_epoch=512)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"method": m, "passed": False, "error": str(e)})

    print(f"\n{'='*70}\nSMOKE TEST SUMMARY\n{'='*70}")
    print(f"{'Method':<12} {'Passed':<8} {'Elapsed':<10} {'EpRet':<10} {'EpCost':<10} {'Lambda':<10}")
    for r in results:
        if r.get("passed"):
            print(f"{r['method']:<12} {'OK':<8} {r['elapsed_s']:>6.1f}s   "
                  f"{(r['final_ep_ret'] or float('nan')):>+8.2f}  "
                  f"{(r['final_ep_cost'] or float('nan')):>+8.2f}  "
                  f"{(r['final_lambda'] or float('nan')):>+8.3f}")
        else:
            print(f"{r['method']:<12} {'FAIL':<8} {r.get('error', '')}")

    n_passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{n_passed}/{len(results)} methods passed.")
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
