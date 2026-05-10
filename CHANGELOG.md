# DRACO Codebase Changelog

> Records non-trivial changes to scripts, configs, and infra on the bitahub server
> (`~/draco_safe_rl/`). Core algorithm code (`draco/algos/`, `draco/modules/`,
> `draco/envs/`) is intentionally NOT modified — workarounds live in `scripts_v2/`.
>
> Format: reverse chronological. Each entry: date, time-local (CST), commit-style
> headline, then bullets. Skip trivial reformat / typo fixes.
>
> Append new entries via `scripts_v2/update_changelog.py` (auto-stamps date/time
> and lists `git status --porcelain` for changed files).

## 2026-05-10

### 08:50 — chore(changelog): bootstrap helper script

- self-test entry by update_changelog.py to verify it works
- should appear at top of 2026-05-10 section

### 08:00 — feat(train_warmup): two-phase warmup training to bypass Lagrangian cold-start attractor

- New `scripts_v2/train_warmup.py`. Adds `--warmup-steps N` flag.
  - Phase 1: build algo with `method=baseline`, train for `N` steps.
  - Phase 2: rebuild algo with `--method` (target, e.g., draco), transfer
    `ac.state_dict()` (actor + reward critic, same architecture across all methods).
    Cost critic / GPD / Choquet re-init from scratch (different shapes).
  - Same venv across phases. Single contiguous tqdm pbar.
  - If `--warmup-steps == 0` or `--method == baseline`, behaves like train.py.
- New `~/run_warmup_draco.sh`. Launches `train_warmup.py` for draco × {0, 1, 2, 15, 42}
  with warmup_steps=100K (50 epochs of baseline) + 900K of DRACO. Auto CPU eval each.
- Smoke test (seed 99, 50K + 50K budget): final ret=271, EpCost=0, Lambda=0.013 — healthy.
- Hypothesis: warmup phase produces "safe enough" actor before risk-aware machinery
  kicks in, so phase 2 sees no early cost violation → no Lagrangian attractor trigger.
- Status: 5-seed batch running, ETA ~12:30 CST.

### 02:01 — feat(scripts): launch n=5 extra seeds (15, 42) and bug fixes

- New `~/run_extra_seeds.sh`. 5 method × {15, 42} = 10 train+eval runs using
  `scripts_v2/train.py`. Eval backgrounded (CPU, no GPU contention with next train).
  `set -uo pipefail` (no `-e`) so single failure doesn't kill batch.
- Triggered by user's autonomous-fallback rule: "if no other instructions by 02:00,
  run 2 more seeds (15, 42)".
- bugfix `scripts_v2/evaluate.py:_norm_list()` — handles `hidden_sizes` /
  `shell_sigmas` from new model checkpoints where they are stored as YAML lists,
  not CSV strings. Old code did `.split(",")` blindly → `AttributeError`.
- Retro-eval baseline_seed15/42 manually after fix.

### 03:05 — bugfix(evaluate): list-vs-string args in load_algo

- `scripts_v2/evaluate.py` `_norm_list(v, cast)` now accepts list/tuple OR CSV string.
- Affects models trained by `train.py` with `--config <yaml>` (YAML serializes lists).

## 2026-05-09

### 23:00 — refactor(scripts_v2): config.yaml + tqdm + clean stdout

- New directory `~/draco_safe_rl/scripts_v2/` (cp from `scripts/`, then rewritten).
- New `scripts_v2/_common.py`:
  - `load_config(path)` — YAML loader (PyYAML).
  - `merge_args_with_defaults(args, parser_defaults, cfg)` — CLI > config >
    parser-default precedence.
  - `make_quiet_dump(pbar)` — monkey-patches `draco.utils.logger.Logger.dump`
    to silence print() while preserving .txt/.jsonl writes; updates tqdm postfix.
- New `scripts_v2/train.py`:
  - Accepts `--config <yaml>`. CLI overrides config.
  - tqdm pbar over epochs with `EpRet/EpCost/Lambda` postfix.
  - Logger writes to .txt/.jsonl; no per-epoch stdout spam.
  - Records start/end timestamps + steps/sec at end.
- New `scripts_v2/evaluate.py`:
  - Multi-eps × multi-channel sweep in one call (not just one eps like old).
  - `--config configs/eval_default.yaml` with `eps_list`, `channels`, `n_eval_episodes`.
  - tqdm progress over `(channels × eps)`. No per-episode stdout.
  - Output JSON shape: `{channels: {ch: {eps: [...], ret_mean: [...], ...}}}` —
    paired arrays for direct ret-vs-eps plotting.
- New `scripts_v2/plot_results.py`:
  - Mode `learning_curves` (existing): EpRet/EpCost/Lambda vs steps.
  - Mode `eval_sweep` (NEW): ret-vs-eps and cost-vs-eps per channel × method.
- New `scripts_v2/aggregate_eval.py` — long-format CSV (handles both old and new
  eval JSON shapes).
- New `scripts_v2/summarize_eval.py` — produces Markdown summary table at given
  eps with mean ± std across seeds and rule-based GO/No-go judgement.
- New `configs/eval_default.yaml` and `configs/plot_default.yaml`.
- Helper shell scripts in `~/`:
  - `eval_all_cpu.sh` — batch CPU eval of pre-trained models.
  - `eval_draco_watcher.sh` — watch for draco model files and CPU-eval as they appear.
  - `finalize.sh` — aggregate + plot in one call.
- Constraint: did NOT modify `scripts/` (the runner was actively using it for
  the in-flight training batch). All new code in `scripts_v2/`.

### 22:55 — bugfix(evaluate): work around scg_wrapper PAPER_OBS_DIMS bug

- `scg_wrapper.py:178` declares `PAPER_OBS_DIMS["quadrotor_stab"] = [0, 1, 4, 5, 7, 10]`,
  but actual quadrotor_stab obs is 6-d (indices 0-5). Indices 7, 10 cause
  `IndexError: index 7 is out of bounds for axis 0 with size 6` during obs-channel eval.
- Per user instruction "do not modify core algorithm/env code", workaround in
  `scripts_v2/evaluate.py:_safe_obs_dims()`:
  - Probes `SCGWrapper.observation_space.shape[0]` once.
  - Filters PAPER_OBS_DIMS to in-bounds indices.
  - Result for quadrotor_stab: dims `[0, 1, 4, 5]` instead of `[0, 1, 4, 5, 7, 10]`.
  - Cached per task to avoid repeated env probes.

### 22:30 — infra(robust-gymnasium): clone repo and create isolated venv (feasibility prep)

- Cloned `https://github.com/SafeRL-Lab/Robust-Gymnasium` to `~/rg_repo/`.
- Created isolated venv `~/rg_venv/` (`python -m venv`) to avoid polluting main env.
- `pip install robust-gymnasium` failed: transitive `pybullet_svl` wheel
  build error on Python 3.11 (`error: invalid command 'bdist_wheel'`).
- Workaround documented in `reviews/robust-gymnasium-feasibility.md`. Install
  succeeded later (08:00) via `pip install --no-deps -e ~/rg_repo`.

### 22:30 — fix(claude-md): correct project path

- `CLAUDE.md` line 14: `~/draco_safe_rl_v3/` → `~/draco_safe_rl/` (actual path).

---

## Conventions

- **Date headers** are local CST. Sub-entries have `HH:MM — type(scope): headline`.
- **Types**: `feat`, `bugfix`, `refactor`, `infra`, `fix`, `docs`, `chore`, `test`.
- **Append via helper**: `python3 scripts_v2/update_changelog.py "type(scope): headline" --details "bullet 1" "bullet 2"` from `~/draco_safe_rl/`.
- **Don't log**: trivial reformat, typo fixes, generated outputs (`*.eval_v2.json`,
  `*.png`, `*.csv`), partial intermediate states (only commit-worthy units).
- **Mirror**: server-side `~/draco_safe_rl/CHANGELOG.md` is canonical;
  local `C:\Users\Delia\Desktop\Draco\CHANGELOG.md` is a mirror, scp'd when changes settle.
