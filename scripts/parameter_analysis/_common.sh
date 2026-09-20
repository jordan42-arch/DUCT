#!/bin/bash
# Shared helpers for DUCT parameter analysis suites.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/scripts/comparison/_common.sh"

PARAM_MODEL=${PARAM_MODEL:-DUCT}
PARAM_EVAL_DATASETS=${PARAM_EVAL_DATASETS:-"ETTm1 ETTh2"}
PARAM_HORIZONS=${PARAM_HORIZONS:-"96 336 720"}
PARAM_PRETRAIN_DATASETS=${PARAM_PRETRAIN_DATASETS:-"ETTm1"}
PARAM_GROUP=${PARAM_GROUP:-ett_regime}

PARAM_RESULT_ROOT=${PARAM_RESULT_ROOT:-$ROOT/logs/parameter_analysis}
PARAM_CHECKPOINT_ROOT=${PARAM_CHECKPOINT_ROOT:-$ROOT/checkpoints/parameter_analysis}

tag_value() {
    echo "$1" | sed -e 's/[^A-Za-z0-9_.-]/_/g' -e 's/\./p/g'
}

run_parameter_value() {
    local suite=$1
    local parameter=$2
    local value=$3
    local pretrain_datasets_csv=$4
    local eval_datasets=$5
    local horizons=$6
    local pretrain_flags=$7
    local pretrain_epochs=${8:-$PRETRAIN_EPOCHS}
    local pretrain_batch=${9:-$PRETRAIN_BATCH}
    local finetune_batch=${10:-$BATCH}
    local finetune_flags=${11:-}
    local model_name=${12:-$PARAM_MODEL}

    local value_tag
    value_tag=$(tag_value "$value")
    local run_dir="$PARAM_RESULT_ROOT/$suite/seed_$SEED/$value_tag"
    local ckpt_dir="$PARAM_CHECKPOINT_ROOT/$suite/seed_$SEED/$value_tag"
    local ckpt="$ckpt_dir/pretrained_backbone.pt"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir" "$ckpt_dir"
    cd "$ROOT" || exit 1

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=parameter_analysis
group_name=$PARAM_GROUP
model=$model_name
suite=$suite
parameter=$parameter
value=$value
pretrain_datasets=$pretrain_datasets_csv
eval_datasets=$eval_datasets
horizons=$horizons
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$pretrain_epochs
pretrain_batch=$pretrain_batch
finetune_batch=$finetune_batch
lr=$LR
seed=$SEED
data_root=$DATA_ROOT
checkpoint=$ckpt
pretrain_flags=$pretrain_flags
finetune_flags=$finetune_flags
created_at=$(date -Iseconds)
EOF

    echo "Started parameter suite=$suite value=$value date=$(date)" > "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Metrics: $run_dir/metrics.json" >> "$log_file"
    echo "Cells: $run_dir/cells" >> "$log_file"
    echo "Config: $run_dir/run_config.env" >> "$log_file"

    read -r -a flag_array <<< "$pretrain_flags"
    echo "=== PRETRAIN suite=$suite value=$value datasets=$pretrain_datasets_csv epochs=$pretrain_epochs $(date) ===" >> "$log_file"
    run_logged "$log_file" "$PYTHON" pretrain.py \
        --model "$model_name" \
        --data_dir "$DATA_ROOT" \
        --datasets "$pretrain_datasets_csv" \
        --lookback "$LOOKBACK" \
        --epochs "$pretrain_epochs" \
        --batch "$pretrain_batch" \
        --lr "$LR" \
        --seed "$SEED" \
        --save_path "$ckpt" \
        "${flag_array[@]}"

    for dataset in $eval_datasets; do
        for horizon in $horizons; do
            echo "=== FINETUNE suite=$suite value=$value dataset=$dataset horizon=$horizon $(date) ===" >> "$log_file"
            run_logged "$log_file" "$PYTHON" finetune.py \
                --data_dir "$DATA_ROOT" \
                --model "$model_name" \
                --dataset "$dataset" \
                --lookback "$LOOKBACK" \
                --pred_len "$horizon" \
                --pretrained_path "$ckpt" \
                --epochs "$EPOCHS" \
                --batch "$finetune_batch" \
                --lr "$LR" \
                --seed "$SEED" \
                $finetune_flags \
                --results_dir "$run_dir"
        done
    done

    echo "DONE parameter suite=$suite value=$value date=$(date)" >> "$log_file"
}
