# DRACO: Distributional Risk-Aware Choquet-Ordered safe RL

> 论文标题候选：**风险感知的值分布鲁棒安全强化学习**
> Algorithm name placeholder: `ALGO_NAME = "DRACO"` in `draco/algos/draco.py` — change in one place to rename.

DRACO is a PPO-Lagrangian-based safe RL algorithm that combines:

1. **D**istributional cost critic via IQN — instead of `V_C(s)` (scalar), we maintain `Z_C(s, τ)` (full quantile function);
2. **R**isk-aware tail estimation via per-state GPD — POT-fit on IQN tail samples (peaks-over-threshold) gives state-conditional CVaR with extrapolation;
3. **A**ggregation across K perturbation shells — each shell is a "risk source"; each has its own GPD head;
4. **C**hoquet integral with sigmoid+reg capacity — non-additive aggregation that captures source synergy;
5. **O**rdered safe RL — Lagrangian penalty + cost budget constraint replaces `E[C] ≤ d` with `Choquet-CVaR(C) ≤ d`.

See `ALGORITHM.md` for the math and `docs/derivation.md` for derivations.

---

## TL;DR

- **5 baselines** all in this codebase (single `--method` flag): `baseline / fuz / evo_style / dist_only / draco`
- **Multi-env parallel rollout** — 8 envs gives ~3× wall-clock speedup vs serial
- **Validated end-to-end** on a synthetic env: all 5 methods train without NaN, GPD MLE recovers true ξ/β, Choquet stays non-additive
- **First-class `safe-control-gym`** support (cartpole_stab, quadrotor_stab) plus a built-in toy env for dependency-free smoke testing
- **Three-channel evaluation** (obs / act / dyn perturbations, FuzRL Table-3 style) with full reproducibility

---

## Repository layout

```
draco_safe_rl/
├── README.md          # ← you are here
├── SETUP.md           # detailed env setup (start here if new)
├── ALGORITHM.md       # math and pseudocode
├── EXPERIMENTS.md     # replication protocol for the midterm defense
├── requirements.txt
├── configs/           # YAML configs per task
├── draco/
│   ├── algos/draco.py       # main algorithm + 5-method dispatch
│   ├── modules/
│   │   ├── iqn_critic.py        # IQN distributional cost critic
│   │   ├── multi_source_gpd.py  # K-source GPD heads + closed-form CVaR
│   │   ├── fuzzy_net.py         # sigmoid + reg Choquet aggregator (B1 fixed)
│   │   ├── actor_critic.py      # Gaussian/Categorical actor + scalar reward critic
│   │   └── buffer.py            # vectorized rollout buffer with cost-to-go targets
│   ├── envs/
│   │   ├── vec_env.py            # SyncVectorEnv (multi-env parallel rollout)
│   │   ├── toy_env.py            # 4-d synthetic safe RL env (no deps)
│   │   └── scg_wrapper.py        # safe-control-gym adapter (cartpole, quadrotor)
│   └── utils/{logger.py, seeds.py}
├── scripts/
│   ├── train.py                  # ⭐ unified training entry
│   ├── evaluate.py               # 3-channel perturbation eval
│   ├── plot_results.py           # shaded-band learning curves
│   ├── aggregate_eval.py         # collect *.eval.json → CSV
│   └── run_midterm_experiments.sh # one-button launcher
└── tests/
    ├── smoke_test_full.py    # all-5-methods integration test
    └── ...                   # individual module tests
```

---

## Quickstart (3 commands)

> **Prerequisite**: complete `SETUP.md` first.

```bash
# 1. Smoke test (no extra deps; ~30s)
python tests/smoke_test_full.py

# 2. Train DRACO on toy task (5 min on CPU)
python scripts/train.py --task toy --method draco --seed 0 \
    --total-steps 50000 --n-envs 8

# 3. Plot the curves
python scripts/plot_results.py --runs-dir runs --out my_first_run.png
open my_first_run.png   # or `xdg-open` on Linux
```

If you have safe-control-gym set up:

```bash
python scripts/train.py --task cartpole_stab --method draco --seed 0 \
    --total-steps 1000000 --n-envs 8 --save-model
```

---

## The five methods at a glance

|Method|IQN cost critic|EVT/GPD|Multi-shell|Choquet|What it tells us|
|---|---|---|---|---|---|
|`baseline`|❌|❌|❌|❌|PPO-Lag floor|
|`fuz`|❌|❌|✓ K=4|✓|Choquet alone helps?|
|`evo_style`|❌|✓ single|❌|❌|EVT alone helps? (≈ EVO)|
|`dist_only`|✓|❌|❌|❌|Distributional cost critic alone helps?|
|**`draco`**|✓|✓ multi|✓ K=4|✓|**The proposed method**|

The 4-cell ablation matrix (excluding baseline) lets you isolate each contribution:

- `fuz` vs `baseline`: marginal effect of multi-shell + Choquet alone
- `evo_style` vs `baseline`: marginal effect of EVT alone
- `dist_only` vs `baseline`: marginal effect of distributional critic alone
- `draco` vs each of the above: incremental gain of combining

This 4-cell matrix is the experimental backbone of the paper / defense.

---

## Key CLI flags (`scripts/train.py`)

|flag|default|notes|
|---|---|---|
|`--task`|`toy`|`toy / cartpole_stab / cartpole_track / quadrotor_stab / quadrotor_track`|
|`--method`|`draco`|see table above|
|`--seed`|`0`||
|`--n-envs`|`8`|parallel envs (linear speedup up to ~16 for fast envs)|
|`--total-steps`|`200_000`|environment steps total|
|`--cost-limit`|`25.0`|episode cost budget; tune per task|
|`--n-shells`|`4`|K in DRACO; 4 is a sweet spot (Choquet capacity has 2^K-2 free params)|
|`--shell-sigmas`|`0.01,0.05,0.10,0.20`|stratified perturbation magnitudes|
|`--n-tau-train`|`32`|IQN quantile samples per gradient step|
|`--eta`|`0.7`|GPD threshold quantile (POT)|

Run `python scripts/train.py --help` for the full list.

---

## Multi-env parallelism (training-time speedup)

DRACO uses `SyncVectorEnv` (Python-loop batched envs in a single process) by default. Empirically:

- **N=1 env**: 245 env-steps/sec (CPU, toy env)
- **N=4 envs**: 505 env-steps/sec (2.06×)
- **N=8 envs**: 708 env-steps/sec (2.89×)

For slow envs (MuJoCo, complex quadrotor dynamics), `AsyncVectorEnv` (multiprocessing) gives nearly linear scaling. Wrapping is one line — see `draco/envs/vec_env.py`.

---

## Reproducing the midterm defense results

```bash
bash scripts/run_midterm_experiments.sh
# Expected wall time on 3080: ~30-50 GPU-h
# Output:
#   runs/<task>/<method>/<run>.jsonl       (training logs)
#   runs/<task>/<method>/<run>.eval.json   (3-channel eval)
#   midterm_curves.png                     (learning curves)
#   midterm_eval_table.csv                 (eval results)
```

See `EXPERIMENTS.md` for the full protocol.

---

## Citation / changelog

This codebase is derived from the `evt_otp_fuz_v5_ablation` PRISM repo:

- **Removed** (per ablation results showing OT is harmful): all OT-perturbation modules (`ot_perturb.py`)
- **Removed**: legacy diff scripts, soft-max-Choquet variants, multiple v3/v4 changelogs
- **Added**: IQN cost critic, multi-source GPD heads, Choquet aggregation pipeline
- **Refactored**: vectorized rollout, cleaner method dispatch via `MethodConfig`

For the experimental results that motivate this design, see `PROJECT_HANDOFF_5_9.md` (kept in `docs/` for record).
