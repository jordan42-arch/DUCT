#!/bin/bash
# Focused ETTh1-96 search, expanding around the current winners.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/comparison/_common.sh"

MODEL_NAME=${MODEL_NAME:-DUCT}
SEARCH_GROUP=${SEARCH_GROUP:-duct_etth1_96_detailed}
SEARCH_RESULT_ROOT=${SEARCH_RESULT_ROOT:-$ROOT/logs/parameter_analysis/$SEARCH_GROUP}
SEARCH_CHECKPOINT_ROOT=${SEARCH_CHECKPOINT_ROOT:-$ROOT/checkpoints/parameter_analysis/$SEARCH_GROUP}
TARGET_DATASET=${TARGET_DATASET:-ETTh1}
TARGET_HORIZON=${TARGET_HORIZON:-96}
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
    local patch_len=$5
    local stride=$6
    local dropout=$7
    local ft_lr=$8
    local ft_batch=$9
    local branch_weight=${10}
    local suffix=${11:-}
    local extra_ft=${12:-}

    local ptag="p$(tag_text "$patch_len")s$(tag_text "$stride")"
    local dtag="drop$(tag_text "$dropout")"
    local lr_tag="lr$(tag_text "$ft_lr")"
    local batch_tag="b$(tag_text "$ft_batch")"
    local blw_tag="blw$(tag_text "$branch_weight")"
    local config_name="${arch_name}_${ptag}_${dtag}_${lr_tag}_${batch_tag}_${blw_tag}${suffix}"
    local pretrain_tag="${arch_name}_${ptag}_${dtag}"
    local pre_flags
    local ft_flags

    pre_flags="--d_model $d_model --n_layers $n_layers --d_ff $d_ff --patch_len $patch_len --stride $stride --dropout $dropout $(pretrain_defaults)"
    ft_flags="--d_model $d_model --n_layers $n_layers --d_ff $d_ff --patch_len $patch_len --stride $stride --dropout $dropout --lr $ft_lr --batch $ft_batch --branch_loss_weight $branch_weight $extra_ft"
    echo "$config_name|$pretrain_tag|$pre_flags|$ft_flags"
}

default_configs() {
    local arch_specs=(
        "d096_l2_ff192 96 2 192"
        "d096_l3_ff192 96 3 192"
        "d128_l2_ff256 128 2 256"
        "d128_l3_ff256 128 3 256"
        "d128_l4_ff256 128 4 256"
        "d128_l4_ff384 128 4 384"
        "d160_l3_ff320 160 3 320"
        "d160_l4_ff320 160 4 320"
        "d160_l5_ff320 160 5 320"
        "d160_l4_ff480 160 4 480"
        "d192_l3_ff384 192 3 384"
        "d192_l4_ff384 192 4 384"
        "d192_l4_ff576 192 4 576"
        "d224_l3_ff448 224 3 448"
        "d224_l4_ff448 224 4 448"
    )

    # Main architecture x finetune-lr grid. Ordered by lr then architecture so
    # the three workers start with different pretrain tags.
    for lr in 3e-5 5e-5 7e-5 1e-4 1.5e-4; do
        for spec in "${arch_specs[@]}"; do
            read -r name d_model n_layers d_ff <<< "$spec"
            config_line "$name" "$d_model" "$n_layers" "$d_ff" 16 8 0.10 "$lr" 16 0.20
        done
    done

    # Batch-size probes around promising small, medium, and wide settings.
    for spec in \
        "d096_l2_ff192 96 2 192" \
        "d160_l4_ff320 160 4 320" \
        "d192_l3_ff384 192 3 384"; do
        read -r name d_model n_layers d_ff <<< "$spec"
        for lr in 5e-5 7e-5 1e-4; do
            for batch in 8 32; do
                config_line "$name" "$d_model" "$n_layers" "$d_ff" 16 8 0.10 "$lr" "$batch" 0.20 "_batchprobe"
            done
        done
    done

    # Auxiliary branch loss weight probes.
    for spec in \
        "d096_l2_ff192 96 2 192" \
        "d160_l4_ff320 160 4 320"; do
        read -r name d_model n_layers d_ff <<< "$spec"
        for lr in 5e-5 7e-5; do
            for blw in 0.00 0.05 0.10 0.30 0.50; do
                config_line "$name" "$d_model" "$n_layers" "$d_ff" 16 8 0.10 "$lr" 16 "$blw" "_branchprobe"
            done
        done
    done

    # Fixed fusion probes. Previous coarse search liked fixed patch weight 0.25.
    for lr in 5e-5 7e-5; do
        for w in 0.15 0.25 0.35 0.50 0.65; do
            config_line d160_l4_ff320 160 4 320 16 8 0.10 "$lr" 16 0.20 "_fusion$(tag_text "$w")" "--fusion_init_patch_weight $w --freeze_fusion"
        done
    done

    # Dropout probes with matched pretraining and finetuning dropout.
    for spec in \
        "d096_l2_ff192 96 2 192" \
        "d160_l4_ff320 160 4 320"; do
        read -r name d_model n_layers d_ff <<< "$spec"
        for lr in 5e-5 7e-5; do
            for dropout in 0.05 0.15 0.20; do
                config_line "$name" "$d_model" "$n_layers" "$d_ff" 16 8 "$dropout" "$lr" 16 0.20 "_dropprobe"
            done
        done
    done

    # Patch length / stride probes with matched pretraining and finetuning.
    for spec in \
        "d096_l2_ff192 96 2 192" \
        "d160_l4_ff320 160 4 320"; do
        read -r name d_model n_layers d_ff <<< "$spec"
        for lr in 5e-5 7e-5; do
            for patch_stride in "8 4" "12 6" "24 12"; do
                read -r patch_len stride <<< "$patch_stride"
                config_line "$name" "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" 0.10 "$lr" 16 0.20 "_patchprobe"
            done
        done
    done
}

run_cmd() {
    local log_file=$1
    shift
    echo ">>> $*" >> "$log_file"
    "$@" >> "$log_file" 2>&1
}

cell_done() {
    local run_dir=$1
    local metrics="$run_dir/cells/${TARGET_DATASET}_pl${TARGET_HORIZON}__${MODEL_NAME}/metrics.json"
    [ -s "$metrics" ]
}

ensure_pretrain() {
    local pretrain_tag=$1
    local pretrain_flags=$2
    local ckpt=$3
    local log_file=$4

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        echo "=== SKIP PRETRAIN tag=$pretrain_tag ckpt exists $(date) ===" >> "$log_file"
        return 0
    fi

    mkdir -p "$(dirname "$ckpt")"
    local lock_dir="${ckpt}.lock"
    while ! mkdir "$lock_dir" 2>/dev/null; do
        if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
            echo "=== SKIP PRETRAIN tag=$pretrain_tag produced by another worker $(date) ===" >> "$log_file"
            return 0
        fi
        echo "waiting for pretrain lock $lock_dir $(date)" >> "$log_file"
        sleep 30
    done

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        rmdir "$lock_dir"
        echo "=== SKIP PRETRAIN tag=$pretrain_tag ckpt exists after lock $(date) ===" >> "$log_file"
        return 0
    fi

    local pre_args=()
    read -r -a pre_args <<< "$pretrain_flags"
    echo "=== PRETRAIN tag=$pretrain_tag epochs=$PRETRAIN_EPOCHS $(date) ===" >> "$log_file"
    if run_cmd "$log_file" "$PYTHON" pretrain.py \
        --model "$MODEL_NAME" \
        --data_dir "$DATA_ROOT" \
        --datasets "$TARGET_DATASET" \
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
        echo "FAILED PRETRAIN tag=$pretrain_tag status=$status $(date)" >> "$log_file"
        rmdir "$lock_dir" || true
        return "$status"
    fi
}

run_config() {
    local config_name=$1
    local pretrain_tag=$2
    local pretrain_flags=$3
    local finetune_flags=$4
    local run_dir="$SEARCH_RESULT_ROOT/$TARGET_DATASET/pl${TARGET_HORIZON}/$config_name"
    local ckpt="$SEARCH_CHECKPOINT_ROOT/$TARGET_DATASET/$pretrain_tag/pretrained_backbone.pt"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir"
    cd "$ROOT" || exit 1

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=duct_etth1_96_detailed_search
group_name=$SEARCH_GROUP
model=$MODEL_NAME
dataset=$TARGET_DATASET
horizon=$TARGET_HORIZON
config_name=$config_name
pretrain_tag=$pretrain_tag
pretrain_flags=$pretrain_flags
finetune_flags=$finetune_flags
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$PRETRAIN_EPOCHS
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

    echo "Started ETTh1-96 detailed config=$config_name date=$(date)" >> "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Checkpoint: $ckpt" >> "$log_file"

    ensure_pretrain "$pretrain_tag" "$pretrain_flags" "$ckpt" "$log_file"

    if [ "${FORCE_FINETUNE:-0}" != "1" ] && cell_done "$run_dir"; then
        echo "=== SKIP FINETUNE config=$config_name metrics exists $(date) ===" >> "$log_file"
        return 0
    fi

    local ft_args=()
    read -r -a ft_args <<< "$finetune_flags"
    echo "=== FINETUNE config=$config_name epochs=$EPOCHS $(date) ===" >> "$log_file"
    run_cmd "$log_file" "$PYTHON" finetune.py \
        --data_dir "$DATA_ROOT" \
        --model "$MODEL_NAME" \
        --dataset "$TARGET_DATASET" \
        --lookback "$LOOKBACK" \
        --pred_len "$TARGET_HORIZON" \
        --pretrained_path "$ckpt" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --seed "$SEED" \
        "${ft_args[@]}" \
        --results_dir "$run_dir"
}

mkdir -p "$SEARCH_RESULT_ROOT" "$SEARCH_CHECKPOINT_ROOT"
CONFIGS=()
while IFS= read -r line; do
    CONFIGS+=("$line")
done < <(default_configs)

if [ "${PRINT_CONFIGS:-0}" = "1" ]; then
    printf "%s\n" "${CONFIGS[@]}"
    exit 0
fi

idx=0
for spec in "${CONFIGS[@]}"; do
    if [ $((idx % NUM_WORKERS)) -ne "$WORKER_INDEX" ]; then
        idx=$((idx + 1))
        continue
    fi
    IFS='|' read -r config_name pretrain_tag pretrain_flags finetune_flags <<< "$spec"
    if run_config "$config_name" "$pretrain_tag" "$pretrain_flags" "$finetune_flags"; then
        :
    else
        status=$?
        echo "FAILED config=$config_name status=$status date=$(date)" >> "$SEARCH_RESULT_ROOT/failures.log"
    fi
    idx=$((idx + 1))
done

echo "DONE ETTh1-96 detailed search group=$SEARCH_GROUP worker=$WORKER_INDEX/$NUM_WORKERS date=$(date)"
