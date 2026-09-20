#!/bin/bash
# Comparison sweep on the ETT family.
set -e

ETT_DATASETS=${ETT_DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2"}
BATCH=${BATCH:-16}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-16}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
for dataset in $ETT_DATASETS; do
    run_comparison_group "$dataset" "$dataset" "$dataset"
done
