set -euo pipefail

if [ ! -f pretrain.py ] || [ ! -f finetune.py ]; then
    echo "Run this script from the repository root, for example: bash scripts/comparison/ETTh1/run_ETTh1.sh" >&2
    exit 1
fi

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-datasets}
RESULT_ROOT=${RESULT_ROOT:-logs/comparison/DUCT}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/comparison/DUCT}

LOOKBACK=${LOOKBACK:-96}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-20}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-16}
PRETRAIN_LR=${PRETRAIN_LR:-1e-4}
SEED=${SEED:-2024}

MASK_RATIO=${MASK_RATIO:-0.40}
MAE_WEIGHT=${MAE_WEIGHT:-1.0}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0.50}
TEMPERATURE=${TEMPERATURE:-0.07}
AUGMENTATION_MODE=${AUGMENTATION_MODE:-all}

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

init_dataset_run() {
    RUN_DATASET=$1
    RUN_DIR="$RESULT_ROOT/$RUN_DATASET"
    CKPT_ROOT="$CHECKPOINT_ROOT/$RUN_DATASET"
    LOG_FILE="$RUN_DIR/run.log"
    PARAMS_FILE="$RUN_DIR/params.tsv"

    mkdir -p "$RUN_DIR" "$CKPT_ROOT"
    {
        echo "dataset=$RUN_DATASET"
        echo "result_dir=$RUN_DIR"
        echo "checkpoint_root=$CKPT_ROOT"
        echo "data_root=$DATA_ROOT"
        echo "lookback=$LOOKBACK"
        echo "pretrain_epochs=$PRETRAIN_EPOCHS"
        echo "pretrain_batch=$PRETRAIN_BATCH"
        echo "pretrain_lr=$PRETRAIN_LR"
        echo "seed=$SEED"
        echo "created_at=$(date -Iseconds)"
    } > "$RUN_DIR/run_config.env"
    echo "Started DUCT comparison run dataset=$RUN_DATASET date=$(date)" > "$LOG_FILE"
    printf "dataset\thorizon\td_model\tn_layers\td_ff\tpatch_len\tstride\tdropout\tlr\tbatch\tbranch_loss_weight\tepochs\tpatience\tpretrain_tag\n" > "$PARAMS_FILE"
}

ensure_pretrain() {
    local dataset=$1
    local d_model=$2
    local n_layers=$3
    local d_ff=$4
    local patch_len=$5
    local stride=$6
    local dropout=$7
    local tag
    tag=$(pretrain_tag "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout")
    local ckpt_dir="$CKPT_ROOT/$tag"
    local ckpt="$ckpt_dir/pretrained_backbone.pt"
    mkdir -p "$ckpt_dir"

    if [ "${FORCE_PRETRAIN:-0}" != "1" ] && [ -s "$ckpt" ]; then
        echo "=== SKIP PRETRAIN dataset=$dataset tag=$tag checkpoint exists $(date) ===" >> "$LOG_FILE"
        echo "$ckpt"
        return 0
    fi

    echo "=== PRETRAIN dataset=$dataset tag=$tag $(date) ===" >> "$LOG_FILE"
    run_logged "$LOG_FILE" "$PYTHON" pretrain.py \
        --model DUCT \
        --data_dir "$DATA_ROOT" \
        --datasets "$dataset" \
        --lookback "$LOOKBACK" \
        --patch_len "$patch_len" \
        --stride "$stride" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --d_ff "$d_ff" \
        --dropout "$dropout" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch "$PRETRAIN_BATCH" \
        --lr "$PRETRAIN_LR" \
        --seed "$SEED" \
        --mask_ratio "$MASK_RATIO" \
        --mae_weight "$MAE_WEIGHT" \
        --contrastive_weight "$CONTRASTIVE_WEIGHT" \
        --temperature "$TEMPERATURE" \
        --augmentation_mode "$AUGMENTATION_MODE" \
        --save_path "$ckpt"
    echo "$ckpt"
}

run_dataset_cell() {
    local dataset=$1
    local horizon=$2
    local d_model=$3
    local n_layers=$4
    local d_ff=$5
    local patch_len=$6
    local stride=$7
    local dropout=$8
    local lr=$9
    local batch=${10}
    local branch_loss_weight=${11}
    local epochs=${12}
    local patience=${13}

    local tag
    tag=$(pretrain_tag "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$horizon" "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout" \
        "$lr" "$batch" "$branch_loss_weight" "$epochs" "$patience" "$tag" >> "$PARAMS_FILE"

    local ckpt
    ckpt=$(ensure_pretrain "$dataset" "$d_model" "$n_layers" "$d_ff" "$patch_len" "$stride" "$dropout")

    local metrics="$RUN_DIR/cells/${dataset}_pl${horizon}__DUCT/metrics.json"
    if [ "${FORCE_FINETUNE:-0}" != "1" ] && [ -s "$metrics" ]; then
        echo "=== SKIP FINETUNE dataset=$dataset pl=$horizon metrics exists $(date) ===" >> "$LOG_FILE"
        return 0
    fi

    echo "=== FINETUNE dataset=$dataset pl=$horizon tag=$tag $(date) ===" >> "$LOG_FILE"
    run_logged "$LOG_FILE" "$PYTHON" finetune.py \
        --data_dir "$DATA_ROOT" \
        --model DUCT \
        --dataset "$dataset" \
        --lookback "$LOOKBACK" \
        --pred_len "$horizon" \
        --patch_len "$patch_len" \
        --stride "$stride" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --d_ff "$d_ff" \
        --dropout "$dropout" \
        --pretrained_path "$ckpt" \
        --epochs "$epochs" \
        --patience "$patience" \
        --batch "$batch" \
        --lr "$lr" \
        --seed "$SEED" \
        --branch_loss_weight "$branch_loss_weight" \
        --results_dir "$RUN_DIR"
}

finish_dataset_run() {
    echo "DONE DUCT comparison run dataset=$RUN_DATASET date=$(date)" >> "$LOG_FILE"
}
