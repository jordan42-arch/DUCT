#!/bin/bash
# Shared runner for comparison experiments. Source this file from dataset-specific
# scripts in this directory.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_dataset}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/checkpoints/comparison/by_dataset}

LOOKBACK=${LOOKBACK:-96}
HORIZONS=${HORIZONS:-"96 192 336 720"}
MODELS=${MODELS:-"Naive DLinear iTransformer PatchTST"}
EPOCHS=${EPOCHS:-30}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-100}
BATCH=${BATCH:-16}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-$BATCH}
SEED=${SEED:-2024}
LR=${LR:-1e-4}

write_common_config() {
    local config_file=$1
    local experiment_type=$2
    local group_name=$3
    local datasets_csv=$4
    local extra_models=$5
    local checkpoint_path=${6:-}

    cat > "$config_file" <<EOF
experiment_type=$experiment_type
group_name=$group_name
datasets=$datasets_csv
models=$extra_models
horizons=$HORIZONS
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$PRETRAIN_EPOCHS
batch=$BATCH
pretrain_batch=$PRETRAIN_BATCH
lr=$LR
seed=$SEED
data_root=$DATA_ROOT
checkpoint=$checkpoint_path
created_at=$(date -Iseconds)
EOF
}

run_logged() {
    local log_file=$1
    shift
    echo ">>> $*" >> "$log_file"
    "$@" >> "$log_file" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "FAILED ($status): $*" >> "$log_file"
    fi
    return 0
}

run_comparison_group() {
    local group_name=$1
    local datasets_csv=$2
    local pretrain_datasets_csv=${3:-$datasets_csv}
    local run_dir="$RESULT_ROOT/$group_name"
    local ckpt_dir="$CHECKPOINT_ROOT/$group_name"
    local log_file="$run_dir/run.log"
    local ckpt="$ckpt_dir/pretrained_backbone.pt"

    mkdir -p "$run_dir" "$ckpt_dir"
    cd "$ROOT" || exit 1

    write_common_config "$run_dir/run_config.env" "comparison_by_dataset" "$group_name" "$datasets_csv" "$MODELS DUCT" "$ckpt"
    echo "Started comparison group=$group_name datasets=$datasets_csv seed=$SEED date=$(date)" > "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Metrics: $run_dir/metrics.json" >> "$log_file"
    echo "Cells: $run_dir/cells" >> "$log_file"
    echo "Config: $run_dir/run_config.env" >> "$log_file"
    if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
        echo "=== PRETRAIN $pretrain_datasets_csv $(date) ===" >> "$log_file"
        run_logged "$log_file" "$PYTHON" pretrain.py \
            --data_dir "$DATA_ROOT" \
            --datasets "$pretrain_datasets_csv" \
            --lookback "$LOOKBACK" \
            --epochs "$PRETRAIN_EPOCHS" \
            --batch "$PRETRAIN_BATCH" \
            --lr "$LR" \
            --seed "$SEED" \
            --save_path "$ckpt"
    fi

    local datasets=${datasets_csv//,/ }
    for dataset in $datasets; do
        for horizon in $HORIZONS; do
            for model in $MODELS; do
                echo "=== $model $dataset pl$horizon $(date) ===" >> "$log_file"
                run_logged "$log_file" "$PYTHON" finetune.py \
                    --data_dir "$DATA_ROOT" \
                    --model "$model" \
                    --dataset "$dataset" \
                    --lookback "$LOOKBACK" \
                    --pred_len "$horizon" \
                    --epochs "$EPOCHS" \
                    --batch "$BATCH" \
                    --lr "$LR" \
                    --seed "$SEED" \
                    --results_dir "$run_dir"
            done

            echo "=== DUCT $dataset pl$horizon $(date) ===" >> "$log_file"
            run_logged "$log_file" "$PYTHON" finetune.py \
                --data_dir "$DATA_ROOT" \
                --model DUCT \
                --dataset "$dataset" \
                --lookback "$LOOKBACK" \
                --pred_len "$horizon" \
                --pretrained_path "$ckpt" \
                --epochs "$EPOCHS" \
                --batch "$BATCH" \
                --lr "$LR" \
                --seed "$SEED" \
                --results_dir "$run_dir"
        done
    done

    echo "DONE comparison group=$group_name date=$(date)" >> "$log_file"
}

run_single_model_group() {
    local model_name=$1
    local group_name=$2
    local datasets_csv=$3
    local run_dir="$RESULT_ROOT/$model_name/$group_name"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir"
    cd "$ROOT" || exit 1

    write_common_config "$run_dir/run_config.env" "comparison_by_model" "$group_name" "$datasets_csv" "$model_name" ""
    echo "Started single-model comparison model=$model_name group=$group_name datasets=$datasets_csv seed=$SEED date=$(date)" > "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Metrics: $run_dir/metrics.json" >> "$log_file"
    echo "Cells: $run_dir/cells" >> "$log_file"
    echo "Config: $run_dir/run_config.env" >> "$log_file"

    local datasets=${datasets_csv//,/ }
    for dataset in $datasets; do
        for horizon in $HORIZONS; do
            local metrics="$run_dir/cells/${dataset}_pl${horizon}__${model_name}/metrics.json"
            if [ "${FORCE_FINETUNE:-0}" != "1" ] && [ -s "$metrics" ]; then
                echo "=== SKIP $model_name $dataset pl$horizon metrics exists $(date) ===" >> "$log_file"
                continue
            fi
            echo "=== $model_name $dataset pl$horizon $(date) ===" >> "$log_file"
            run_logged "$log_file" "$PYTHON" finetune.py \
                --data_dir "$DATA_ROOT" \
                --model "$model_name" \
                --dataset "$dataset" \
                --lookback "$LOOKBACK" \
                --pred_len "$horizon" \
                --epochs "$EPOCHS" \
                --batch "$BATCH" \
                --lr "$LR" \
                --seed "$SEED" \
                --results_dir "$run_dir"
        done
    done

    echo "DONE single-model comparison model=$model_name group=$group_name date=$(date)" >> "$log_file"
}
