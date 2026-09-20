#!/bin/bash
# DUCT across selected datasets. Each dataset gets its own pretraining
# checkpoint so ETT granularities are never mixed into one backbone.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/checkpoints/comparison/by_model}

source "$ROOT/scripts/comparison/_common.sh"

MODEL_DATASETS=${MODEL_DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar PEMS03 PEMS04 PEMS07 PEMS08"}
GROUP_NAME=${GROUP_NAME:-all_datasets}
BRANCH_LOSS_WEIGHT=${BRANCH_LOSS_WEIGHT:-0.2}

run_dir="$RESULT_ROOT/DUCT/$GROUP_NAME"
ckpt_root="$CHECKPOINT_ROOT/DUCT/$GROUP_NAME"
log_file="$run_dir/run.log"
mkdir -p "$run_dir" "$ckpt_root"
cd "$ROOT" || exit 1

cat > "$run_dir/run_config.env" <<EOF
experiment_type=comparison_by_model
model=DUCT
group_name=$GROUP_NAME
datasets=$MODEL_DATASETS
horizons=$HORIZONS
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$PRETRAIN_EPOCHS
lr=$LR
seed=$SEED
branch_loss_weight=$BRANCH_LOSS_WEIGHT
data_root=$DATA_ROOT
checkpoint_root=$ckpt_root
created_at=$(date -Iseconds)
EOF

echo "Started DUCT comparison group=$GROUP_NAME datasets=$MODEL_DATASETS seed=$SEED date=$(date)" > "$log_file"
echo "Run dir: $run_dir" >> "$log_file"
echo "Metrics: $run_dir/metrics.json" >> "$log_file"
echo "Cells: $run_dir/cells" >> "$log_file"
echo "Config: $run_dir/run_config.env" >> "$log_file"

contains_dataset() {
    local needle=$1
    for item in $MODEL_DATASETS; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

dataset_batch() {
    case "$1" in
        traffic|PEMS03|PEMS04|PEMS07|PEMS08) echo "${HIGH_VAR_BATCH:-2}" ;;
        electricity) echo "${ELECTRICITY_BATCH:-4}" ;;
        solar) echo "${SOLAR_BATCH:-8}" ;;
        *) echo "$BATCH" ;;
    esac
}

run_duct_group() {
    local group=$1
    local datasets_csv=$2
    local pretrain_csv=$3
    local batch_value=$4
    local ckpt_dir="$ckpt_root/$group"
    local ckpt="$ckpt_dir/pretrained_backbone.pt"
    mkdir -p "$ckpt_dir"

    if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
        echo "=== PRETRAIN DUCT group=$group datasets=$pretrain_csv batch=$batch_value $(date) ===" >> "$log_file"
        run_logged "$log_file" "$PYTHON" pretrain.py \
            --model DUCT \
            --data_dir "$DATA_ROOT" \
            --datasets "$pretrain_csv" \
            --lookback "$LOOKBACK" \
            --epochs "$PRETRAIN_EPOCHS" \
            --batch "$batch_value" \
            --lr "$LR" \
            --seed "$SEED" \
            --save_path "$ckpt"
    fi

    local datasets=${datasets_csv//,/ }
    for dataset in $datasets; do
        local finetune_batch
        finetune_batch=$(dataset_batch "$dataset")
        for horizon in $HORIZONS; do
            echo "=== DUCT $dataset pl$horizon batch=$finetune_batch $(date) ===" >> "$log_file"
            run_logged "$log_file" "$PYTHON" finetune.py \
                --data_dir "$DATA_ROOT" \
                --model DUCT \
                --dataset "$dataset" \
                --lookback "$LOOKBACK" \
                --pred_len "$horizon" \
                --pretrained_path "$ckpt" \
                --epochs "$EPOCHS" \
                --batch "$finetune_batch" \
                --lr "$LR" \
                --branch_loss_weight "$BRANCH_LOSS_WEIGHT" \
                --seed "$SEED" \
                --results_dir "$run_dir"
        done
    done
}

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar PEMS03 PEMS04 PEMS07 PEMS08; do
    if contains_dataset "$dataset"; then
        batch_value=$(dataset_batch "$dataset")
        run_duct_group "$dataset" "$dataset" "$dataset" "$batch_value"
    fi
done

echo "DONE DUCT comparison group=$GROUP_NAME date=$(date)" >> "$log_file"
