#!/bin/bash
# Sweep the auxiliary branch-supervision weight used during DUCT finetuning:
# loss = fused + w * (patch_branch + variate_branch).
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

BRANCH_LOSS_WEIGHT_VALUES=${BRANCH_LOSS_WEIGHT_VALUES:-"0 0.05 0.10 0.20 0.50 1.00"}

for value in $BRANCH_LOSS_WEIGHT_VALUES; do
    run_parameter_value \
        "branch_loss_weight" \
        "branch_loss_weight" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "" \
        "$PRETRAIN_EPOCHS" \
        "$PRETRAIN_BATCH" \
        "$BATCH" \
        "--branch_loss_weight $value"
done
