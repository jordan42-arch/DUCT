#!/bin/bash
# Per-cell hyperparameter search; WORKER_INDEX/NUM_WORKERS shard across GPUs.
#   CUDA_VISIBLE_DEVICES=1 WORKER_INDEX=0 NUM_WORKERS=3 bash <this script>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/comparison/_common.sh"

MODEL_NAME=${MODEL_NAME:-DUCT}
SEARCH_GROUP=${SEARCH_GROUP:-duct_cell_hp_proxy}
SEARCH_DATASETS=${SEARCH_DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2"}
SEARCH_HORIZONS=${SEARCH_HORIZONS:-"96 192 336 720"}
SEARCH_RESULT_ROOT=${SEARCH_RESULT_ROOT:-$ROOT/logs/parameter_analysis/$SEARCH_GROUP}
SEARCH_CHECKPOINT_ROOT=${SEARCH_CHECKPOINT_ROOT:-$ROOT/checkpoints/parameter_analysis/$SEARCH_GROUP}
SEARCH_CONFIGS_FILE=${SEARCH_CONFIGS_FILE:-}
WORKER_INDEX=${WORKER_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

tag_text() {
    echo "$1" | sed -e 's/[^A-Za-z0-9_.-]/_/g' -e 's/\./p/g'
}

default_configs() {
    cat <<'EOF'
default|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
cw010|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.10 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
cw030|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.30 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
cw070|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.70 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
mask020|--mask_ratio 0.20 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
mask030|--mask_ratio 0.30 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
mask050|--mask_ratio 0.50 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20
temp005|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.05 --augmentation_mode all|--branch_loss_weight 0.20
temp010|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.10 --augmentation_mode all|--branch_loss_weight 0.20
temp020|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.20 --augmentation_mode all|--branch_loss_weight 0.20
blw000|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.00
blw005|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.05
blw010|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.10
blw050|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.50
aug_noise_scaling|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode noise,scaling|--branch_loss_weight 0.20
aug_none|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode none|--branch_loss_weight 0.20
fusion025|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20 --fusion_init_patch_weight 0.25 --freeze_fusion
fusion050|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20 --fusion_init_patch_weight 0.50 --freeze_fusion
fusion075|--mask_ratio 0.40 --mae_weight 1.0 --contrastive_weight 0.50 --temperature 0.07 --augmentation_mode all|--branch_loss_weight 0.20 --fusion_init_patch_weight 0.75 --freeze_fusion
EOF
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
    local config_name=$2
    local pretrain_flags=$3
    local ckpt=$4
    local log_file=$5

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        echo "=== SKIP PRETRAIN dataset=$dataset config=$config_name ckpt exists $(date) ===" >> "$log_file"
        return 0
    fi

    mkdir -p "$(dirname "$ckpt")"
    local lock_dir="${ckpt}.lock"
    while ! mkdir "$lock_dir" 2>/dev/null; do
        if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
            echo "=== SKIP PRETRAIN dataset=$dataset config=$config_name produced by another worker $(date) ===" >> "$log_file"
            return 0
        fi
        echo "waiting for pretrain lock $lock_dir $(date)" >> "$log_file"
        sleep 30
    done

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        rmdir "$lock_dir"
        echo "=== SKIP PRETRAIN dataset=$dataset config=$config_name ckpt exists after lock $(date) ===" >> "$log_file"
        return 0
    fi

    local pre_args=()
    if [ -n "$pretrain_flags" ]; then
        read -r -a pre_args <<< "$pretrain_flags"
    fi

    echo "=== PRETRAIN dataset=$dataset config=$config_name epochs=$PRETRAIN_EPOCHS $(date) ===" >> "$log_file"
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
        echo "FAILED PRETRAIN dataset=$dataset config=$config_name status=$status $(date)" >> "$log_file"
        rmdir "$lock_dir" || true
        return "$status"
    fi
}

run_cell_config() {
    local dataset=$1
    local horizon=$2
    local config_name=$3
    local pretrain_flags=$4
    local finetune_flags=$5

    local ds_tag
    ds_tag=$(tag_text "$dataset")
    local hz_tag="pl$(tag_text "$horizon")"
    local run_dir="$SEARCH_RESULT_ROOT/$ds_tag/$hz_tag/$config_name"
    local ckpt="$SEARCH_CHECKPOINT_ROOT/$ds_tag/$config_name/pretrained_backbone.pt"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir"
    cd "$ROOT" || exit 1

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=duct_cell_hyperparameter_search
group_name=$SEARCH_GROUP
model=$MODEL_NAME
dataset=$dataset
horizon=$horizon
config_name=$config_name
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

    echo "Started cell search dataset=$dataset horizon=$horizon config=$config_name date=$(date)" >> "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Checkpoint: $ckpt" >> "$log_file"

    ensure_pretrain "$dataset" "$config_name" "$pretrain_flags" "$ckpt" "$log_file"

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
        if [ $((idx % NUM_WORKERS)) -ne "$WORKER_INDEX" ]; then
            idx=$((idx + 1))
            continue
        fi
        for spec in "${CONFIGS[@]}"; do
            IFS='|' read -r config_name pretrain_flags finetune_flags <<< "$spec"
            if run_cell_config "$dataset" "$horizon" "$config_name" "$pretrain_flags" "$finetune_flags"; then
                :
            else
                status=$?
                mkdir -p "$SEARCH_RESULT_ROOT"
                echo "FAILED dataset=$dataset horizon=$horizon config=$config_name status=$status date=$(date)" \
                    >> "$SEARCH_RESULT_ROOT/failures.log"
            fi
        done
        idx=$((idx + 1))
    done
done

echo "DONE duct cell search group=$SEARCH_GROUP worker=$WORKER_INDEX/$NUM_WORKERS date=$(date)"
