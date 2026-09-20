#!/bin/bash
# Resume missing DUCT cells without rewriting the original run log.
#   GPU_ID=0 TASKS="ETTm1:720 ETTm2:all" bash <this script>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/checkpoints/comparison/by_model}

source "$ROOT/scripts/comparison/_common.sh"

GROUP_NAME=${GROUP_NAME:-all_datasets}
GPU_ID=${GPU_ID:?Set GPU_ID to the physical GPU index, e.g. 0}
TASKS=${TASKS:?Set TASKS, e.g. "ETTm1:720 ETTm2:all traffic:all"}
RUN_LABEL=${RUN_LABEL:-gpu${GPU_ID}}
BRANCH_LOSS_WEIGHT=${BRANCH_LOSS_WEIGHT:-0.2}

run_dir="$RESULT_ROOT/DUCT/$GROUP_NAME"
ckpt_root="$CHECKPOINT_ROOT/DUCT/$GROUP_NAME"
driver_dir="$RESULT_ROOT/_driver"
log_file="$driver_dir/run_duct_resume_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$run_dir" "$ckpt_root" "$driver_dir"
cd "$ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU_ID"

dataset_batch() {
    case "$1" in
        traffic|PEMS03|PEMS04|PEMS07|PEMS08) echo "${HIGH_VAR_BATCH:-2}" ;;
        electricity) echo "${ELECTRICITY_BATCH:-4}" ;;
        solar) echo "${SOLAR_BATCH:-8}" ;;
        *) echo "$BATCH" ;;
    esac
}

run_cmd() {
    echo ">>> $*" | tee -a "$log_file"
    "$@" 2>&1 | tee -a "$log_file"
    local status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "FAILED ($status): $*" | tee -a "$log_file"
        exit "$status"
    fi
}

metric_exists() {
    local dataset=$1
    local horizon=$2
    "$PYTHON" - "$run_dir/metrics.json" "DUCT__${dataset}__pl${horizon}" <<'PY'
import json
import os
import sys

path, key = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    raise SystemExit(1)
try:
    data = json.load(open(path))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if key in data else 1)
PY
}

checkpoint_for_dataset() {
    local dataset=$1
    echo "$ckpt_root/$dataset/pretrained_backbone.pt"
}

checkpoint_is_current() {
    local ckpt=$1
    "$PYTHON" - "$ckpt" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
except Exception:
    raise SystemExit(1)
required = ("patch_branch.", "variate_branch.", "rep_fusion.")
ok = all(any(k.startswith(prefix) for k in sd) for prefix in required)
raise SystemExit(0 if ok else 1)
PY
}

pretrain_for_dataset() {
    local dataset=$1
    local ckpt
    ckpt=$(checkpoint_for_dataset "$dataset")
    if [ -s "$ckpt" ] && checkpoint_is_current "$ckpt"; then
        echo "Using existing checkpoint for $dataset: $ckpt" | tee -a "$log_file"
        return
    fi
    if [ -s "$ckpt" ]; then
        echo "Existing checkpoint is stale/incompatible; re-pretraining $dataset: $ckpt" | tee -a "$log_file"
    fi

    local batch_value
    batch_value=$(dataset_batch "$dataset")
    mkdir -p "$(dirname "$ckpt")"
    echo "=== PRETRAIN DUCT group=$dataset datasets=$dataset batch=$batch_value $(date) ===" | tee -a "$log_file"
    run_cmd "$PYTHON" pretrain.py \
        --model DUCT \
        --data_dir "$DATA_ROOT" \
        --datasets "$dataset" \
        --lookback "$LOOKBACK" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch "$batch_value" \
        --lr "$LR" \
        --seed "$SEED" \
        --save_path "$ckpt"
}

echo "Started DUCT resume label=$RUN_LABEL gpu=$GPU_ID date=$(date)" | tee -a "$log_file"
echo "Host: $(hostname)" | tee -a "$log_file"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$log_file"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}" | tee -a "$log_file"
echo "Python: $(which "$PYTHON")" | tee -a "$log_file"
echo "Tasks: $TASKS" | tee -a "$log_file"
echo "Run dir: $run_dir" | tee -a "$log_file"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$log_file"

cat > "$run_dir/resume_${RUN_LABEL}_config.env" <<EOF
experiment_type=comparison_by_model_resume
model=DUCT
group_name=$GROUP_NAME
gpu_id=$GPU_ID
tasks=$TASKS
horizons=$HORIZONS
lookback=$LOOKBACK
epochs=$EPOCHS
pretrain_epochs=$PRETRAIN_EPOCHS
lr=$LR
seed=$SEED
branch_loss_weight=$BRANCH_LOSS_WEIGHT
data_root=$DATA_ROOT
created_at=$(date -Iseconds)
EOF

for item in $TASKS; do
    dataset=${item%%:*}
    horizon_spec=${item#*:}
    if [ "$horizon_spec" = "$dataset" ] || [ "$horizon_spec" = "all" ]; then
        horizon_list="$HORIZONS"
    else
        horizon_list=${horizon_spec//,/ }
    fi

    pretrain_for_dataset "$dataset"
    ckpt=$(checkpoint_for_dataset "$dataset")
    batch_value=$(dataset_batch "$dataset")
    for horizon in $horizon_list; do
        if metric_exists "$dataset" "$horizon"; then
            echo "SKIP existing aggregate metric: DUCT__${dataset}__pl${horizon}" | tee -a "$log_file"
            continue
        fi
        echo "=== DUCT $dataset pl$horizon batch=$batch_value $(date) ===" | tee -a "$log_file"
        run_cmd "$PYTHON" finetune.py \
            --data_dir "$DATA_ROOT" \
            --model DUCT \
            --dataset "$dataset" \
            --lookback "$LOOKBACK" \
            --pred_len "$horizon" \
            --pretrained_path "$ckpt" \
            --epochs "$EPOCHS" \
            --batch "$batch_value" \
            --lr "$LR" \
            --branch_loss_weight "$BRANCH_LOSS_WEIGHT" \
            --seed "$SEED" \
            --results_dir "$run_dir"
    done
done

echo "DONE DUCT resume label=$RUN_LABEL gpu=$GPU_ID date=$(date)" | tee -a "$log_file"
