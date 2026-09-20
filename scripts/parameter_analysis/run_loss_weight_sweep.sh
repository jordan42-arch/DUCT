#!/bin/bash
# Contrastive loss weight sweep with MAE weight fixed at 1.0.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

CONTRASTIVE_WEIGHT_VALUES=${CONTRASTIVE_WEIGHT_VALUES:-"0.10 0.30 0.50 0.70 1.00"}

for value in $CONTRASTIVE_WEIGHT_VALUES; do
    run_parameter_value \
        "loss_weight" \
        "contrastive_weight" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "--mae_weight 1.0 --contrastive_weight $value"
done
