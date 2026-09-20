#!/bin/bash
# Run recent TSLib baselines under the exact DUCT data/evaluation protocol.
# Every completed cell is skipped on restart.
set -u

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}
MODELS=${MODELS:-"TimeMixer PAttn WPMixer FreTS TSMixer"}
EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-6}
LR=${LR:-1e-4}
SEED=${SEED:-2024}
FORCE_FINETUNE=${FORCE_FINETUNE:-0}
MODEL_COUNT=$(echo "$MODELS" | wc -w)
TARGET_CELLS=$((MODEL_COUNT * 58))

LONG_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar"
PEMS_DATASETS="PEMS03 PEMS04 PEMS07 PEMS08"
CLOUD_DATASETS="FaaS_Small FaaS_Medium FaaS_Large IaaS_Small IaaS_Medium IaaS_Large"

DRIVER_DIR="$RESULT_ROOT/_driver"
mkdir -p "$DRIVER_DIR"
DRIVER_LOG="$DRIVER_DIR/recent_tslib.log"
STATUS_LOG="$DRIVER_DIR/recent_tslib.status.log"

completed_count() {
    local total=0
    local model
    for model in $MODELS; do
        local dir="$RESULT_ROOT/$model/all_datasets/cells"
        if [ -d "$dir" ]; then
            local count
            count=$(find "$dir" -mindepth 2 -maxdepth 2 -name metrics.json -type f | wc -l)
            total=$((total + count))
        fi
    done
    echo "$total"
}

batch_for() {
    case "$1" in
        FaaS_Small|FaaS_Medium|FaaS_Large|IaaS_Small|IaaS_Medium|IaaS_Large) echo 16 ;;
        *) echo 2 ;;
    esac
}

run_cell() {
    local model=$1
    local dataset=$2
    local lookback=$3
    local horizon=$4
    local batch
    batch=$(batch_for "$dataset")

    local run_dir="$RESULT_ROOT/$model/all_datasets"
    local cell_dir="$run_dir/cells/${dataset}_pl${horizon}__${model}"
    local metrics="$cell_dir/metrics.json"
    mkdir -p "$run_dir"

    if [ "$FORCE_FINETUNE" != "1" ] && [ -s "$metrics" ]; then
        echo "=== SKIP $model $dataset pl$horizon metrics exists $(date) ===" >> "$DRIVER_LOG"
        return
    fi

    echo "=== START $model $dataset lb$lookback pl$horizon batch$batch $(date) ===" \
        | tee -a "$DRIVER_LOG"
    if "$PYTHON" "$ROOT/finetune.py" \
        --data_dir "$DATA_ROOT" \
        --model "$model" \
        --dataset "$dataset" \
        --lookback "$lookback" \
        --pred_len "$horizon" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch "$batch" \
        --lr "$LR" \
        --seed "$SEED" \
        --results_dir "$run_dir" >> "$DRIVER_LOG" 2>&1; then
        echo "=== DONE $model $dataset pl$horizon progress=$(completed_count)/$TARGET_CELLS $(date) ===" \
            | tee -a "$DRIVER_LOG"
    else
        local status=$?
        echo "=== FAILED status=$status $model $dataset pl$horizon $(date) ===" \
            | tee -a "$DRIVER_LOG"
    fi
}

cd "$ROOT" || exit 1

if echo " $MODELS " | grep -q " WPMixer "; then
    if ! "$PYTHON" -c "import pywt" >/dev/null 2>&1; then
        echo "Missing dependency: PyWavelets. Run: $PYTHON -m pip install PyWavelets" >&2
        exit 2
    fi
fi

for model in $MODELS; do
    run_dir="$RESULT_ROOT/$model/all_datasets"
    mkdir -p "$run_dir"
    cat > "$run_dir/run_config.env" <<EOF
experiment_type=recent_tslib_same_protocol
model=$model
datasets_long=ETTh1,ETTh2,ETTm1,ETTm2,electricity,traffic,weather,exchange_rate,solar
lookback_long=96
horizons_long=96,192,336,720
datasets_pems=PEMS03,PEMS04,PEMS07,PEMS08
lookback_pems=48
horizons_pems=12,24,36,48
datasets_cloud=FaaS_Small,FaaS_Medium,FaaS_Large,IaaS_Small,IaaS_Medium,IaaS_Large
lookback_cloud=144
horizons_cloud=10
epochs=$EPOCHS
patience=$PATIENCE
batch_standard_and_pems=2
batch_cloud=16
lr=$LR
seed=$SEED
optimizer=AdamW
scheduler=CosineAnnealingLR
normalization=train_split_per_variable
created_at=$(date -Iseconds)
EOF
done

echo "$(date) recent TSLib run started models=[$MODELS] target=$TARGET_CELLS" \
    | tee "$STATUS_LOG" >> "$DRIVER_LOG"

for model in $MODELS; do
    for dataset in $LONG_DATASETS; do
        for horizon in 96 192 336 720; do
            run_cell "$model" "$dataset" 96 "$horizon"
        done
    done
    for dataset in $PEMS_DATASETS; do
        for horizon in 12 24 36 48; do
            run_cell "$model" "$dataset" 48 "$horizon"
        done
    done
    for dataset in $CLOUD_DATASETS; do
        run_cell "$model" "$dataset" 144 10
    done
done

"$PYTHON" "$ROOT/scripts/comparison/make_comparison_results.py" \
    --root "$ROOT/logs/comparison" \
    --out "$ROOT/logs/comparison_results.md" >> "$DRIVER_LOG" 2>&1 || true

echo "$(date) recent TSLib run finished progress=$(completed_count)/$TARGET_CELLS" \
    | tee -a "$STATUS_LOG" >> "$DRIVER_LOG"
