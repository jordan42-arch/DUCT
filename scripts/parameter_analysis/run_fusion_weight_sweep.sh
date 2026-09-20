#!/bin/bash
# Sweep fixed fusion weights (value = patch weight, variate gets 1-value).
# Fusion is frozen, isolating each branch's contribution.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

FUSION_PATCH_WEIGHT_VALUES=${FUSION_PATCH_WEIGHT_VALUES:-"0.00 0.25 0.50 0.75 1.00"}
BRANCH_LOSS_WEIGHT=${BRANCH_LOSS_WEIGHT:-0.2}

for value in $FUSION_PATCH_WEIGHT_VALUES; do
    run_parameter_value \
        "fusion_weight" \
        "fusion_patch_weight" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "" \
        "$PRETRAIN_EPOCHS" \
        "$PRETRAIN_BATCH" \
        "$BATCH" \
        "--branch_loss_weight $BRANCH_LOSS_WEIGHT --fusion_init_patch_weight $value --freeze_fusion"
done
