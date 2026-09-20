#!/bin/bash
# Re-finetune the best ETT search candidates with a longer budget, reusing
# existing pretrained checkpoints. Does not pretrain.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

MODEL_NAME=${MODEL_NAME:-DUCT}
SOURCE_GROUP=${SOURCE_GROUP:-duct_ett_detailed_allgpu}
SEARCH_GROUP=${SEARCH_GROUP:-duct_ett_confirm_configs}
SOURCE_CSV=${SOURCE_CSV:-$ROOT/logs/parameter_analysis/$SOURCE_GROUP/cell_search_all.csv}
SOURCE_CHECKPOINT_ROOT=${SOURCE_CHECKPOINT_ROOT:-$ROOT/checkpoints/parameter_analysis/$SOURCE_GROUP}
SEARCH_RESULT_ROOT=${SEARCH_RESULT_ROOT:-$ROOT/logs/parameter_analysis/$SEARCH_GROUP}
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
LOOKBACK=${LOOKBACK:-96}
EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-10}
SEED=${SEED:-2024}
TOP_K=${TOP_K:-3}
RUN_ETTH1_720=${RUN_ETTH1_720:-0}
WORKER_INDEX=${WORKER_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

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

selected_configs() {
    "$PYTHON" - "$SOURCE_CSV" "$TOP_K" "$RUN_ETTH1_720" <<'PY'
import csv
import sys
from collections import defaultdict

source_csv = sys.argv[1]
top_k = int(sys.argv[2])
run_etth1_720 = sys.argv[3] == "1"

rows = []
with open(source_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            row["_test_mse"] = float(row["test_mse"])
            int(row["horizon"])
        except Exception:
            continue
        if not run_etth1_720 and row["dataset"] == "ETTh1" and row["horizon"] == "720":
            continue
        rows.append(row)

by_cell = defaultdict(list)
for row in rows:
    by_cell[(row["dataset"], int(row["horizon"]))].append(row)

for cell in sorted(by_cell, key=lambda x: (x[0], x[1])):
    ranked = sorted(by_cell[cell], key=lambda r: r["_test_mse"])[:top_k]
    for rank, row in enumerate(ranked, start=1):
        fields = [
            row["dataset"],
            row["horizon"],
            str(rank),
            row["config_name"],
            row["pretrain_tag"],
            row["test_mse"],
            row["test_mae"],
            row["finetune_flags"].strip(),
        ]
        print("\t".join(fields))
PY
}

run_selected_config() {
    local dataset=$1
    local horizon=$2
    local rank=$3
    local config_name=$4
    local pretrain_tag=$5
    local source_test_mse=$6
    local source_test_mae=$7
    local finetune_flags=$8

    local run_dir="$SEARCH_RESULT_ROOT/$dataset/pl${horizon}/rank${rank}_${config_name}"
    local ckpt="$SOURCE_CHECKPOINT_ROOT/$dataset/$pretrain_tag/pretrained_backbone.pt"
    local log_file="$run_dir/run.log"

    mkdir -p "$run_dir"
    cd "$ROOT" || exit 1

    cat > "$run_dir/run_config.env" <<EOF
experiment_type=duct_ett_confirm_configs
group_name=$SEARCH_GROUP
source_group=$SOURCE_GROUP
model=$MODEL_NAME
dataset=$dataset
horizon=$horizon
rank=$rank
config_name=$config_name
pretrain_tag=$pretrain_tag
source_test_mse=$source_test_mse
source_test_mae=$source_test_mae
finetune_flags=$finetune_flags
lookback=$LOOKBACK
epochs=$EPOCHS
patience=$PATIENCE
seed=$SEED
data_root=$DATA_ROOT
checkpoint=$ckpt
worker_index=$WORKER_INDEX
num_workers=$NUM_WORKERS
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}
created_at=$(date -Iseconds)
EOF

    echo "Started DUCT confirm run dataset=$dataset horizon=$horizon rank=$rank config=$config_name date=$(date)" >> "$log_file"
    echo "Run dir: $run_dir" >> "$log_file"
    echo "Checkpoint: $ckpt" >> "$log_file"

    if [ ! -s "$ckpt" ]; then
        echo "MISSING CHECKPOINT dataset=$dataset horizon=$horizon tag=$pretrain_tag ckpt=$ckpt" >> "$SEARCH_RESULT_ROOT/failures.log"
        return 1
    fi

    if [ "${FORCE_FINETUNE:-0}" != "1" ] && cell_done "$run_dir" "$dataset" "$horizon"; then
        echo "=== SKIP FINETUNE dataset=$dataset horizon=$horizon rank=$rank metrics exists $(date) ===" >> "$log_file"
        return 0
    fi

    local ft_args=()
    read -r -a ft_args <<< "$finetune_flags"
    echo "=== FINETUNE dataset=$dataset horizon=$horizon rank=$rank epochs=$EPOCHS patience=$PATIENCE $(date) ===" >> "$log_file"
    run_cmd "$log_file" "$PYTHON" finetune.py \
        --data_dir "$DATA_ROOT" \
        --model "$MODEL_NAME" \
        --dataset "$dataset" \
        --lookback "$LOOKBACK" \
        --pred_len "$horizon" \
        --pretrained_path "$ckpt" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        "${ft_args[@]}" \
        --results_dir "$run_dir"
}

mkdir -p "$SEARCH_RESULT_ROOT"
if [ "$WORKER_INDEX" = "0" ] || [ ! -s "$SEARCH_RESULT_ROOT/selected_configs.tsv" ]; then
    selected_configs > "$SEARCH_RESULT_ROOT/selected_configs.tsv.tmp"
    mv "$SEARCH_RESULT_ROOT/selected_configs.tsv.tmp" "$SEARCH_RESULT_ROOT/selected_configs.tsv"
fi
while [ ! -s "$SEARCH_RESULT_ROOT/selected_configs.tsv" ]; do
    sleep 1
done

idx=0
while IFS=$'\t' read -r dataset horizon rank config_name pretrain_tag source_test_mse source_test_mae finetune_flags; do
    if [ $((idx % NUM_WORKERS)) -ne "$WORKER_INDEX" ]; then
        idx=$((idx + 1))
        continue
    fi
    if run_selected_config "$dataset" "$horizon" "$rank" "$config_name" "$pretrain_tag" "$source_test_mse" "$source_test_mae" "$finetune_flags"; then
        :
    else
        status=$?
        echo "FAILED dataset=$dataset horizon=$horizon rank=$rank config=$config_name status=$status date=$(date)" \
            >> "$SEARCH_RESULT_ROOT/failures.log"
    fi
    idx=$((idx + 1))
done < "$SEARCH_RESULT_ROOT/selected_configs.tsv"

echo "DONE DUCT confirm configs group=$SEARCH_GROUP worker=$WORKER_INDEX/$NUM_WORKERS date=$(date)"
