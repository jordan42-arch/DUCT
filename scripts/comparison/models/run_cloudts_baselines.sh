#!/bin/bash
# Baselines on the cloud workload datasets, same protocol as run_cloudts.sh
# so the rows are comparable. Usage: run_cloudts_baselines.sh [GPU]
set -u

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

LOOKBACK=${LOOKBACK:-144}
HORIZON=${HORIZON:-10}
MODELS=${MODELS:-"Naive DLinear iTransformer PatchTST"}
CLOUDTS_DATASETS=${CLOUDTS_DATASETS:-"FaaS_Small FaaS_Medium FaaS_Large IaaS_Small IaaS_Medium IaaS_Large"}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-16}
LR=${LR:-1e-4}
SEED=${SEED:-2024}

LOG_DIR="$ROOT/logs/comparison/by_model/_driver"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cloudts_baselines_gpu${GPU}.log"

cd "$ROOT" || exit 1
echo "START cloudts baselines gpu=$GPU lookback=$LOOKBACK horizon=$HORIZON $(date)" > "$LOG"

for dataset in $CLOUDTS_DATASETS; do
    for model in $MODELS; do
        run_dir="$RESULT_ROOT/$model/all_datasets"
        mkdir -p "$run_dir"
        metrics="$run_dir/cells/${dataset}_pl${HORIZON}__${model}/metrics.json"
        if [ "${FORCE_FINETUNE:-0}" != "1" ] && [ -s "$metrics" ]; then
            echo "=== SKIP $model $dataset pl$HORIZON metrics exists $(date) ===" >> "$LOG"
            continue
        fi
        echo "=== $model $dataset pl$HORIZON $(date) ===" >> "$LOG"
        "$PYTHON" finetune.py \
            --data_dir "$DATA_ROOT" \
            --model "$model" \
            --dataset "$dataset" \
            --lookback "$LOOKBACK" \
            --pred_len "$HORIZON" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --lr "$LR" \
            --seed "$SEED" \
            --results_dir "$run_dir" >> "$LOG" 2>&1
        [ $? -ne 0 ] && echo "FAILED: $model $dataset pl$HORIZON" >> "$LOG"
    done
done

echo "DONE cloudts baselines $(date)" >> "$LOG"
