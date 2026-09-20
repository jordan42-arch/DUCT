#!/bin/bash
# Comparison sweeps on PEMS03/04/07/08. Each PEMS dataset gets its own
# pretraining checkpoint because the channel counts differ.
set -e

PEMS_DATASETS=${PEMS_DATASETS:-"PEMS03 PEMS04 PEMS07 PEMS08"}
LOOKBACK=${LOOKBACK:-48}
HORIZONS=${HORIZONS:-"12 24 36 48"}
BATCH=${BATCH:-2}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-2}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

for dataset in $PEMS_DATASETS; do
    run_comparison_group "$dataset" "$dataset" "$dataset"
done
