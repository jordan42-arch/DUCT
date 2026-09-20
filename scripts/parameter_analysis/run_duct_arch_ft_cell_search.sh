#!/bin/bash
# Second-stage search: capacity plus finetune lr/batch, selected per cell.
# Pretraining checkpoints are shared by dataset and architecture tag.
#   CUDA_VISIBLE_DEVICES=1 WORKER_INDEX=0 NUM_WORKERS=3 bash <this script>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/comparison/_common.sh"

MODEL_NAME=${MODEL_NAME:-DUCT}
SEARCH_GROUP=${SEARCH_GROUP:-duct_cell_hp_archft_all}
SEARCH_DATASETS=${SEARCH_DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar PEMS03 PEMS04 PEMS07 PEMS08"}
SEARCH_HORIZONS=${SEARCH_HORIZONS:-"96 192 336 720"}
SEARCH_RESULT_ROOT=${SEARCH_RESULT_ROOT:-$ROOT/logs/parameter_analysis/$SEARCH_GROUP}
SEARCH_CHECKPOINT_ROOT=${SEARCH_CHECKPOINT_ROOT:-$ROOT/checkpoints/parameter_analysis/$SEARCH_GROUP}
SEARCH_CONFIGS_FILE=${SEARCH_CONFIGS_FILE:-}
WORKER_INDEX=${WORKER_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

tag_text() {
    echo "$1" | sed -e 's/[^A-Za-z0-9_.-]/_/g' -e 's/\./p/g'
}

pretrain_defaults() {
    echo "--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all"
}

config_line() {
    local arch_name=$1
    local d_model=$2
    local n_layers=$3
    local d_ff=$4
    local ft_lr=$5
    local ft_batch=$6
    local extra_name=${7:-}
    local pre_flags
    local ft_flags
    local cfg_name
    local lr_tag
    local batch_tag

    lr_tag=$(tag_text "$ft_lr")
    batch_tag=$(tag_text "$ft_batch")
    cfg_name="${arch_name}_lr${lr_tag}_b${batch_tag}${extra_name}"
    pre_flags="--d_model $d_model --n_layers $n_layers --d_ff $d_ff $(pretrain_defaults)"
    ft_flags="--d_model $d_model --n_layers $n_layers --d_ff $d_ff --lr $ft_lr --batch $ft_batch --branch_loss_weight 0.20"
    echo "$cfg_name|$arch_name|$pre_flags|$ft_flags"
}

default_configs() {
    # Architecture sweep with batch fixed at 16 and three finetune learning rates.
    for lr in 5e-5 1e-4 2e-4; do
        config_line d096_l2_ff192 96 2 192 "$lr" 16
        config_line d128_l2_ff256 128 2 256 "$lr" 16
        config_line d128_l3_ff256 128 3 256 "$lr" 16
        config_line d160_l3_ff320 160 3 320 "$lr" 16
        config_line d160_l4_ff320 160 4 320 "$lr" 16
        config_line d192_l3_ff384 192 3 384 "$lr" 16
    done

    # Batch-size probes around the default and a wider architecture.
    config_line d128_l3_ff256 128 3 256 1e-4 8
    config_line d128_l3_ff256 128 3 256 1e-4 32
    config_line d160_l3_ff320 160 3 320 1e-4 8
    config_line d160_l3_ff320 160 3 320 1e-4 32

    # Two interaction probes for lr x batch.
    config_line d128_l3_ff256 128 3 256 5e-5 32 _lr_batch_probe
    config_line d128_l3_ff256 128 3 256 2e-4 8 _lr_batch_probe
}

load_configs() {
    if [ -n "$SEARCH_CONFIGS_FILE" ]; then
        grep -v '^[[:space:]]*#' "$SEARCH_CONFIGS_FILE" | sed '/^[[:space:]]*$/d'
    else
        default_configs
    fi
}

run_cmd() {
    local log_file=$1
    shift
    echo ">>> $*" >> "$log_file"
    "$@" >> "$log_file" 2>&1
}

cell_done() {
    local run_dir=$1
    local dataset=$2
    local horizon=$3
    local metrics="$run_dir/cells/${dataset}_pl${horizon}__${MODEL_NAME}/metrics.json"
    [ -s "$metrics" ]
}

ensure_pretrain() {
    local dataset=$1
    local pretrain_tag=$2
    local pretrain_flags=$3
    local ckpt=$4
    local log_file=$5

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        echo "=== SKIP PRETRAIN dataset=$dataset tag=$pretrain_tag ckpt exists $(date) ===" >> "$log_file"
        return 0
    fi

    mkdir -p "$(dirname "$ckpt")"
    local lock_dir="${ckpt}.lock"
    while ! mkdir "$lock_dir" 2>/dev/null; do
        if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
            echo "=== SKIP PRETRAIN dataset=$dataset tag=$pretrain_tag produced by another worker $(date) ===" >> "$log_file"
            return 0
        fi
        echo "waiting for pretrain lock $lock_dir $(date)" >> "$log_file"
        sleep 30
    done

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        rmdir "$lock_dir"
        echo "=== SKIP PRETRAIN dataset=$dataset tag=$pretrain_tag ckpt exists after lock $(date) ===" >> "$log_file"
        return 0
    fi

    local pre_args=()
    if [ -n "$pretrain_flags" ]; then
        read -r -a pre_args <<< "$pretrain_flags"
    fi

    echo "=== PRETRAIN dataset=$dataset tag=$pretrain_tag epochs=$PRETRAIN_EPOCHS $(date) ===" >> "$log_file"
    if run_cmd "$log_file" "$PYTHON" pretrain.py \
        --model "$MODEL_NAME" \
        --data_dir "$DATA_ROOT" \
        --datasets "$dataset" \
        --lookback "$LOOKBACK" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch "$PRETRAIN_BATCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --save_path "$ckpt" \
        "${pre_args[@]}"; then
        rmdir "$lock_dir"
    else
        local status=$?
        echo "FAILED PRETRAIN dataset=$dataset tag=$pretrain_tag status=$status $(date)" >> "$log_file"
        rmdir "$lock_dir" || true
        return "$status"
    fi
}

run_cell_config() {
    local dataset=$1
    local horizon=$2
    local config_name=$3
    local pretrain_tag=$4
    local pretrain_flags=$5
    local finetune_flags=$6

    local ds_tag
    ds_tag=$(tag_text "$dataset")
    local hz_tag="pl$(tag_text "$horizon")"
    local run_dir="$SEARCH_RESULT_ROOT/$ds_tag/$hz_tag/$config_name"
    local ckpt="$SEARCH_CHECKPOINT_ROOT/$ds_tag/$pretrain_tag/pretrained_backbone.pt"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir"
    cd "$ROOT" || exit 1

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=duct_cell_arch_ft_hyperparameter_search
group_name=$SEARCH_GROUP
model=$MODEL_NAME
dataset=$dataset
horizon=$horizon
config_name=$config_name
pretrain_tag=$pretrain_tag
pretrain_flags=$pretrain_flags
finetune_flags=$finetune_flags
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$PRETRAIN_EPOCHS
batch=$BATCH
pretrain_batch=$PRETRAIN_BATCH
lr=$LR
seed=$SEED
data_root=$DATA_ROOT
checkpoint=$ckpt
worker_index=$WORKER_INDEX
num_workers=$NUM_WORKERS
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}
created_at=$(date -Iseconds)
EOF

    echo "Started arch+ft cell search dataset=$dataset horizon=$horizon config=$config_name date=$(date)" >> "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Checkpoint: $ckpt" >> "$log_file"

    ensure_pretrain "$dataset" "$pretrain_tag" "$pretrain_flags" "$ckpt" "$log_file"

    if [ "${FORCE_FINETUNE:-0}" != "1" ] && cell_done "$run_dir" "$dataset" "$horizon"; then
        echo "=== SKIP FINETUNE dataset=$dataset horizon=$horizon config=$config_name metrics exists $(date) ===" >> "$log_file"
        return 0
    fi

    local ft_args=()
    if [ -n "$finetune_flags" ]; then
        read -r -a ft_args <<< "$finetune_flags"
    fi

    echo "=== FINETUNE dataset=$dataset horizon=$horizon config=$config_name epochs=$EPOCHS $(date) ===" >> "$log_file"
    run_cmd "$log_file" "$PYTHON" finetune.py \
        --data_dir "$DATA_ROOT" \
        --model "$MODEL_NAME" \
        --dataset "$dataset" \
        --lookback "$LOOKBACK" \
        --pred_len "$horizon" \
        --pretrained_path "$ckpt" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --seed "$SEED" \
        "${ft_args[@]}" \
        --results_dir "$run_dir"
}

mkdir -p "$SEARCH_RESULT_ROOT" "$SEARCH_CHECKPOINT_ROOT"
mapfile -t CONFIGS < <(load_configs)

idx=0
for dataset in $SEARCH_DATASETS; do
    for horizon in $SEARCH_HORIZONS; do
        for spec in "${CONFIGS[@]}"; do
            if [ $((idx % NUM_WORKERS)) -ne "$WORKER_INDEX" ]; then
                idx=$((idx + 1))
                continue
            fi
            IFS='|' read -r config_name pretrain_tag pretrain_flags finetune_flags <<< "$spec"
            if run_cell_config "$dataset" "$horizon" "$config_name" "$pretrain_tag" "$pretrain_flags" "$finetune_flags"; then
                :
            else
                status=$?
                mkdir -p "$SEARCH_RESULT_ROOT"
                echo "FAILED dataset=$dataset horizon=$horizon config=$config_name status=$status date=$(date)" \
                    >> "$SEARCH_RESULT_ROOT/failures.log"
            fi
            idx=$((idx + 1))
        done
    done
done

echo "DONE duct arch+ft cell search group=$SEARCH_GROUP worker=$WORKER_INDEX/$NUM_WORKERS date=$(date)"
