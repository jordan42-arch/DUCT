#!/usr/bin/env bash
# Shared helpers for DUCT ablation scripts.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ ! -f "$ROOT/pretrain.py" ] || [ ! -f "$ROOT/finetune.py" ]; then
    echo "Cannot find pretrain.py/finetune.py under $ROOT" >&2
    exit 1
fi

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/ablation/DUCT}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/checkpoints/ablation/DUCT}

SEED=${SEED:-2024}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-20}
PRETRAIN_LR=${PRETRAIN_LR:-1e-4}
MASK_RATIO=${MASK_RATIO:-0.40}
MAE_WEIGHT=${MAE_WEIGHT:-1.0}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0.50}
TEMPERATURE=${TEMPERATURE:-0.07}
AUGMENTATION_MODE=${AUGMENTATION_MODE:-all}

ABLATION_SUITE=${ABLATION_SUITE:-core}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"full mae_only contrastive_only no_pretrain no_branch_aux fixed_equal_fusion patch_only var_only no_augmentation no_time_warp"}

tag_text() {
    echo "$1" | sed -e 's/[^A-Za-z0-9_.-]/_/g' -e 's/\./p/g'
}

pretrain_tag() {
    local d_model=$1
    local n_layers=$2
    local d_ff=$3
    local patch_len=$4
    local stride=$5
    local dropout=$6
    echo "d${d_model}_l${n_layers}_ff${d_ff}_p${patch_len}s${stride}_drop$(tag_text "$dropout")"
}

run_logged() {
    local log_file=$1
    shift
    echo ">>> $*" >> "$log_file"
    "$@" >> "$log_file" 2>&1
}

suite_root() {
    echo "$RESULT_ROOT/$ABLATION_SUITE/seed_$SEED"
}

suite_ckpt_root() {
    echo "$CHECKPOINT_ROOT/$ABLATION_SUITE/seed_$SEED"
}

write_summary_header() {
    local summary_file
    summary_file="$(suite_root)/runs.tsv"
    mkdir -p "$(dirname "$summary_file")"
    if [ ! -f "$summary_file" ]; then
        printf "suite\tsection\tvariant\tdataset\thorizon\tlookback\td_model\tn_layers\td_ff\tpatch_len\tstride\tdropout\tlr\tbatch\tbranch_loss_weight\tepochs\tpatience\tpretrain_label\tpretrain_flags\tfinetune_flags\trun_dir\n" > "$summary_file"
    fi
}

append_summary() {
    local section=$1
    local variant=$2
    local dataset=$3
    local horizon=$4
    local lookback=$5
    local d_model=$6
    local n_layers=$7
    local d_ff=$8
    local patch_len=$9
    local stride=${10}
    local dropout=${11}
    local lr=${12}
    local batch=${13}
    local branch_loss_weight=${14}
    local epochs=${15}
    local patience=${16}
    local pretrain_label=${17}
    local pretrain_flags=${18}
    local finetune_flags=${19}
    local run_dir=${20}
    write_summary_header
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ABLATION_SUITE" "$section" "$variant" "$dataset" "$horizon" "$lookback" \
        "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout" "$lr" "$batch" \
        "$branch_loss_weight" "$epochs" "$patience" "$pretrain_label" "$pretrain_flags" \
        "$finetune_flags" "$run_dir" >> "$(suite_root)/runs.tsv"
}

ensure_pretrain() {
    local pretrain_label=$1
    local dataset=$2
    local lookback=$3
    local d_model=$4
    local n_layers=$5
    local d_ff=$6
    local patch_len=$7
    local stride=$8
    local dropout=$9
    local pretrain_batch=${10}
    local pretrain_extra=${11}

    local tag
    tag=$(pretrain_tag "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout")
    local ckpt_dir
    ckpt_dir="$(suite_ckpt_root)/$dataset/$tag/$pretrain_label"
    local ckpt="$ckpt_dir/pretrained_backbone.pt"
    local log_file="$ckpt_dir/pretrain.log"
    mkdir -p "$ckpt_dir"

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        echo "$ckpt"
        return 0
    fi

    local extra_args=()
    if [ -n "$pretrain_extra" ]; then
        read -r -a extra_args <<< "$pretrain_extra"
    fi

    {
        echo "Started DUCT ablation pretrain label=$pretrain_label dataset=$dataset date=$(date)"
        echo "checkpoint=$ckpt"
        echo "pretrain_extra=$pretrain_extra"
    } > "$log_file"
    run_logged "$log_file" "$PYTHON" pretrain.py \
        --model DUCT \
        --data_dir "$DATA_ROOT" \
        --datasets "$dataset" \
        --lookback "$lookback" \
        --patch_len "$patch_len" \
        --stride "$stride" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --d_ff "$d_ff" \
        --dropout "$dropout" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch "$pretrain_batch" \
        --lr "$PRETRAIN_LR" \
        --seed "$SEED" \
        --mask_ratio "$MASK_RATIO" \
        --mae_weight "$MAE_WEIGHT" \
        --contrastive_weight "$CONTRASTIVE_WEIGHT" \
        --temperature "$TEMPERATURE" \
        --augmentation_mode "$AUGMENTATION_MODE" \
        --save_path "$ckpt" \
        "${extra_args[@]}"
    echo "$ckpt"
}

variant_spec() {
    local variant=$1
    local base_branch_loss=$2
    case "$variant" in
        full)
            echo "reference|full||0|--branch_loss_weight $base_branch_loss"
            ;;
        mae_only)
            echo "pretrain_objective|mae_only|--no_contrastive|0|--branch_loss_weight $base_branch_loss"
            ;;
        contrastive_only)
            echo "pretrain_objective|contrastive_only|--no_mae|0|--branch_loss_weight $base_branch_loss"
            ;;
        no_pretrain)
            echo "pretrain_objective|scratch||1|--branch_loss_weight $base_branch_loss"
            ;;
        no_augmentation)
            echo "augmentation|no_augmentation|--no_augmentation|0|--branch_loss_weight $base_branch_loss"
            ;;
        no_time_warp)
            echo "augmentation|no_time_warp|--augmentation_mode noise,scaling|0|--branch_loss_weight $base_branch_loss"
            ;;
        no_branch_aux)
            echo "branch_fusion|full||0|--branch_loss_weight 0"
            ;;
        fixed_equal_fusion)
            echo "branch_fusion|full||0|--branch_loss_weight $base_branch_loss --fusion_init_patch_weight 0.5 --freeze_fusion"
            ;;
        patch_only)
            echo "branch_fusion|full||0|--branch_loss_weight 0 --fusion_init_patch_weight 1.0 --freeze_fusion"
            ;;
        var_only)
            echo "branch_fusion|full||0|--branch_loss_weight 0 --fusion_init_patch_weight 0.0 --freeze_fusion"
            ;;
        *)
            echo "Unknown ablation variant: $variant" >&2
            return 2
            ;;
    esac
}

run_variant_cell() {
    local variant=$1
    local dataset=$2
    local horizon=$3
    local lookback=$4
    local d_model=$5
    local n_layers=$6
    local d_ff=$7
    local patch_len=$8
    local stride=$9
    local dropout=${10}
    local lr=${11}
    local batch=${12}
    local branch_loss_weight=${13}
    local epochs=${14}
    local patience=${15}
    local pretrain_batch=${16}

    local spec section pretrain_label pretrain_extra scratch finetune_extra
    spec=$(variant_spec "$variant" "$branch_loss_weight")
    IFS='|' read -r section pretrain_label pretrain_extra scratch finetune_extra <<< "$spec"

    local run_dir
    run_dir="$(suite_root)/$section/$variant/$dataset/pl$horizon"
    local log_file="$run_dir/run.log"
    local metrics="$run_dir/cells/${dataset}_pl${horizon}__DUCT/metrics.json"
    mkdir -p "$run_dir"

    append_summary "$section" "$variant" "$dataset" "$horizon" "$lookback" "$d_model" "$n_layers" \
        "$d_ff" "$patch_len" "$stride" "$dropout" "$lr" "$batch" "$branch_loss_weight" \
        "$epochs" "$patience" "$pretrain_label" "$pretrain_extra" "$finetune_extra" "$run_dir"

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=duct_ablation
suite=$ABLATION_SUITE
section=$section
variant=$variant
model=DUCT
dataset=$dataset
horizon=$horizon
lookback=$lookback
d_model=$d_model
n_layers=$n_layers
d_ff=$d_ff
patch_len=$patch_len
stride=$stride
dropout=$dropout
lr=$lr
batch=$batch
branch_loss_weight=$branch_loss_weight
epochs=$epochs
patience=$patience
pretrain_batch=$pretrain_batch
pretrain_epochs=$PRETRAIN_EPOCHS
pretrain_lr=$PRETRAIN_LR
pretrain_label=$pretrain_label
pretrain_flags=$pretrain_extra
finetune_flags=$finetune_extra
scratch=$scratch
seed=$SEED
data_root=$DATA_ROOT
created_at=$(date -Iseconds)
EOF

    if [ "${FORCE_FINETUNE:-0}" != "1" ] && [ -s "$metrics" ]; then
        echo "SKIP existing metrics: $metrics"
        return 0
    fi

    {
        echo "Started DUCT ablation variant=$variant dataset=$dataset horizon=$horizon date=$(date)"
        echo "Run dir: $run_dir"
    } > "$log_file"

    local finetune_args=()
    if [ -n "$finetune_extra" ]; then
        read -r -a finetune_args <<< "$finetune_extra"
    fi

    local ckpt=""
    if [ "$scratch" != "1" ]; then
        ckpt=$(ensure_pretrain "$pretrain_label" "$dataset" "$lookback" "$d_model" "$n_layers" "$d_ff" \
            "$patch_len" "$stride" "$dropout" "$pretrain_batch" "$pretrain_extra")
    fi

    local args=(
        "$PYTHON" finetune.py
        --data_dir "$DATA_ROOT"
        --model DUCT
        --dataset "$dataset"
        --lookback "$lookback"
        --pred_len "$horizon"
        --patch_len "$patch_len"
        --stride "$stride"
        --d_model "$d_model"
        --n_layers "$n_layers"
        --d_ff "$d_ff"
        --dropout "$dropout"
        --epochs "$epochs"
        --patience "$patience"
        --batch "$batch"
        --lr "$lr"
        --seed "$SEED"
        --results_dir "$run_dir"
    )
    if [ "$scratch" != "1" ]; then
        args+=(--pretrained_path "$ckpt")
    fi
    args+=("${finetune_args[@]}")

    run_logged "$log_file" "${args[@]}"
    echo "DONE DUCT ablation variant=$variant dataset=$dataset horizon=$horizon date=$(date)" >> "$log_file"
}

run_duct_ablation_cell() {
    local dataset=$1
    local horizon=$2
    local lookback=$3
    local d_model=$4
    local n_layers=$5
    local d_ff=$6
    local patch_len=$7
    local stride=$8
    local dropout=$9
    local lr=${10}
    local batch=${11}
    local branch_loss_weight=${12}
    local epochs=${13}
    local patience=${14}
    local pretrain_batch=${15:-16}

    for variant in $ABLATION_VARIANTS; do
        run_variant_cell "$variant" "$dataset" "$horizon" "$lookback" "$d_model" "$n_layers" \
            "$d_ff" "$patch_len" "$stride" "$dropout" "$lr" "$batch" "$branch_loss_weight" \
            "$epochs" "$patience" "$pretrain_batch"
    done
}
