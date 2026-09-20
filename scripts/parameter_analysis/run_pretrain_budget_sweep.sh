#!/bin/bash
# Pretraining budget sweep. This tests whether the default pretrain length is
# undertrained, saturated, or over-regularized.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

PRETRAIN_BUDGET_VALUES=${PRETRAIN_BUDGET_VALUES:-"10 30 60 100"}

for value in $PRETRAIN_BUDGET_VALUES; do
    run_parameter_value \
        "pretrain_budget" \
        "pretrain_epochs" \
        "$value" \
        "$PARAM_PRETRAIN_DATASETS" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        "" \
        "$value"
done
