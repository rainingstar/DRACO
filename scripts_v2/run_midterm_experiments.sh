#!/usr/bin/env bash
# DRACO midterm-defense experiment launcher
#
# Runs all 5 methods × 2 tasks × 3 seeds = 30 runs sequentially, plus
# evaluation under 3 perturbation channels for each. Estimated wall-time
# on 3080: ~30-50 GPU-h (cartpole_stab fast, quadrotor_stab moderate).
#
# Usage:
#     bash scripts/run_midterm_experiments.sh                 # full grid
#     bash scripts/run_midterm_experiments.sh cartpole_stab   # single task
#     SEEDS="0 1" bash scripts/run_midterm_experiments.sh     # restrict seeds
#
# Override env vars:
#     METHODS, SEEDS, TOTAL_STEPS, N_ENVS, LOG_DIR, EVAL_EPS, EVAL_EPISODES
#
# After completion, plot with:
#     python scripts/plot_results.py --runs-dir runs --out midterm_curves.png

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
METHODS=${METHODS:-"baseline fuz evo_style dist_only draco"}
SEEDS=${SEEDS:-"0 1 2"}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
N_ENVS=${N_ENVS:-8}
LOG_DIR=${LOG_DIR:-runs}
EVAL_EPS=${EVAL_EPS:-0.10}
EVAL_EPISODES=${EVAL_EPISODES:-30}

# Default tasks if no arg
if [ "$#" -eq 0 ]; then
    TASKS="cartpole_stab quadrotor_stab"
else
    TASKS="$@"
fi

mkdir -p "$LOG_DIR"

# ============================================================================
# Print plan
# ============================================================================
echo "===================================================================="
echo "DRACO midterm experiment plan"
echo "===================================================================="
echo "Tasks      : $TASKS"
echo "Methods    : $METHODS"
echo "Seeds      : $SEEDS"
echo "Total steps: $TOTAL_STEPS"
echo "N envs     : $N_ENVS"
echo "Eval eps   : $EVAL_EPS, episodes: $EVAL_EPISODES"
echo "Log dir    : $LOG_DIR"
N_TASKS=$(echo $TASKS | wc -w)
N_METHODS=$(echo $METHODS | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_TASKS * N_METHODS * N_SEEDS))
echo "Total runs : $N_RUNS"
echo "===================================================================="
echo ""

# ============================================================================
# Train all combinations
# ============================================================================
RUN_IDX=0
for TASK in $TASKS; do
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            RUN_IDX=$((RUN_IDX + 1))
            RUN_NAME="${TASK}_${METHOD}_seed${SEED}"
            echo "------- [$RUN_IDX/$N_RUNS] TRAIN $RUN_NAME -------"
            python scripts/train.py \
                --task "$TASK" \
                --method "$METHOD" \
                --seed "$SEED" \
                --total-steps "$TOTAL_STEPS" \
                --n-envs "$N_ENVS" \
                --log-dir "$LOG_DIR" \
                --run-name "$RUN_NAME" \
                --save-model
        done
    done
done

# ============================================================================
# Evaluate (only for SCG tasks; toy doesn't support perturbations)
# ============================================================================
echo ""
echo "===================================================================="
echo "EVALUATION PHASE"
echo "===================================================================="
RUN_IDX=0
for TASK in $TASKS; do
    if [ "$TASK" = "toy" ]; then
        echo "Skipping eval for toy task."
        continue
    fi
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            RUN_IDX=$((RUN_IDX + 1))
            RUN_NAME="${TASK}_${METHOD}_seed${SEED}"
            MODEL_PATH="$LOG_DIR/$TASK/$METHOD/${RUN_NAME}.model.pt"
            if [ ! -f "$MODEL_PATH" ]; then
                echo "  SKIP (no model): $MODEL_PATH"
                continue
            fi
            echo "------- [$RUN_IDX] EVAL $RUN_NAME -------"
            python scripts/evaluate.py \
                --model-path "$MODEL_PATH" \
                --task "$TASK" \
                --eps "$EVAL_EPS" \
                --n-eval-episodes "$EVAL_EPISODES" \
                --seed 1000
        done
    done
done

# ============================================================================
# Plot
# ============================================================================
echo ""
echo "===================================================================="
echo "PLOTTING"
echo "===================================================================="
python scripts/plot_results.py \
    --runs-dir "$LOG_DIR" \
    --tasks $TASKS \
    --out midterm_curves.png

# ============================================================================
# Aggregate eval table
# ============================================================================
python scripts/aggregate_eval.py --runs-dir "$LOG_DIR" --out midterm_eval_table.csv

echo ""
echo "===================================================================="
echo "ALL DONE. See:"
echo "  - $LOG_DIR/<task>/<method>/<run>.jsonl     (training logs)"
echo "  - $LOG_DIR/<task>/<method>/<run>.eval.json (eval results)"
echo "  - midterm_curves.png                        (training curves)"
echo "  - midterm_eval_table.csv                    (aggregated eval table)"
echo "===================================================================="
