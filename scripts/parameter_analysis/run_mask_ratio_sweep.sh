#!/bin/bash
# MAE patch mask-ratio sweep. Default includes the paper setting 0.4.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

MASK_RATIO_VALUES=${MASK_RATIO_VALUES:-"0.20 0.30 0.40 0.50 0.60"}

for value in $MASK_RATIO_VALUES; do
    run_parameter_value \
        "mask_ratio" \
        "mask_ratio" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "--mask_ratio $value"
done
