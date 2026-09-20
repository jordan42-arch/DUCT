#!/bin/bash
# Baselines on PEMS under the PEMS protocol (lookback 48, horizons 12-48),
# which is what makes them comparable with the DUCT PEMS cells.
set -u

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${PYTHON:-/mnt/traffic/home/dingkuiye/.conda/envs/time/bin/python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}

DATASET=$1
GPU=$2

LOOKBACK=${LOOKBACK:-48}
HORIZONS=${HORIZONS:-"12 24 36 48"}
MODELS=${MODELS:-"Naive DLinear iTransformer PatchTST"}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-2}
LR=${LR:-1e-4}
SEED=${SEED:-2024}

LOG_DIR="$ROOT/logs/comparison/by_model/_driver"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pems_rerun_${DATASET}${TAG:-}.log"

cd "$ROOT" || exit 1
echo "START $DATASET gpu=$GPU lookback=$LOOKBACK horizons=$HORIZONS $(date)" > "$LOG"

for horizon in $HORIZONS; do
    for model in $MODELS; do
        run_dir="$RESULT_ROOT/$model/all_datasets"
        mkdir -p "$run_dir"
        echo "=== $model $DATASET pl$horizon $(date) ===" >> "$LOG"
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" finetune.py \
            --data_dir "$DATA_ROOT" \
            --model "$model" \
            --dataset "$DATASET" \
            --lookback "$LOOKBACK" \
            --pred_len "$horizon" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --lr "$LR" \
            --seed "$SEED" \
            --results_dir "$run_dir" >> "$LOG" 2>&1
        status=$?
        [ $status -ne 0 ] && echo "FAILED ($status): $model $DATASET pl$horizon" >> "$LOG"
    done
done

echo "DONE $DATASET $(date)" >> "$LOG"
