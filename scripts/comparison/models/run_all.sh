#!/bin/bash
# Run all comparison/baseline models independently.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON=${PYTHON:-python}

LONG_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar"
PEMS_DATASETS="PEMS03 PEMS04 PEMS07 PEMS08"

for runner in run_naive.sh run_dlinear.sh run_itransformer.sh run_patchtst.sh; do
  MODEL_DATASETS="$LONG_DATASETS" LOOKBACK=96 HORIZONS="96 192 336 720" \
    GROUP_NAME=all_datasets bash "$SCRIPT_DIR/$runner"
  MODEL_DATASETS="$PEMS_DATASETS" LOOKBACK=48 HORIZONS="12 24 36 48" \
    GROUP_NAME=all_datasets bash "$SCRIPT_DIR/$runner"
done

# DUCT uses the released per-dataset/per-horizon configurations.
bash "$ROOT/scripts/comparison/run_all.sh"
bash "$SCRIPT_DIR/run_cloudts_baselines.sh"

"$PYTHON" "$ROOT/scripts/comparison/make_comparison_results.py" \
  --root "$ROOT/logs/comparison" \
  --out "$ROOT/logs/comparison_results.md"
