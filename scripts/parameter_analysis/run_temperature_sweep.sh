#!/bin/bash
# InfoNCE temperature sweep. Lower values sharpen positives; higher values make
# negatives softer. Default includes the paper setting 0.07.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

TEMPERATURE_VALUES=${TEMPERATURE_VALUES:-"0.03 0.05 0.07 0.10 0.20"}

for value in $TEMPERATURE_VALUES; do
    run_parameter_value \
        "temperature" \
        "temperature" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "--temperature $value"
done
