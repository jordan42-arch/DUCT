#!/bin/bash
# Pretraining pool sweep: ETT variants share a channel count, so one backbone
# can span them. Tests which pool drives representation quality.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/scripts/parameter_analysis/_common.sh"

POOL_SPECS=(
    "ETTh1_only|ETTh1"
    "ETTh2_only|ETTh2"
    "ETTm1_only|ETTm1"
    "hourly_ETT|ETTh1,ETTh2"
    "paper_pool|ETTh1,ETTh2,ETTm1"
    "all_ETT|ETTh1,ETTh2,ETTm1,ETTm2"
)

for spec in "${POOL_SPECS[@]}"; do
    name=${spec%%|*}
    datasets=${spec#*|}
    run_parameter_value \
        "pretrain_pool" \
        "pretrain_datasets" \
        "$name" \
        "$datasets" \
        "$PARAM_EVAL_DATASETS" \
        "$PARAM_HORIZONS" \
        ""
done
