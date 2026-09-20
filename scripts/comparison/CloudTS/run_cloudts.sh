#!/bin/bash
# DUCT on the six ByteDance cloud workload datasets.
# lookback 144 = 1440 minutes, matching the E3Former lookback in wall-clock
# time; horizon 10 = 100 minutes. The release is 10-minute granularity.
set -euo pipefail

LOOKBACK=${LOOKBACK:-144}
HORIZON=${HORIZON:-10}
CLOUDTS_DATASETS=${CLOUDTS_DATASETS:-"FaaS_Small FaaS_Medium FaaS_Large IaaS_Small IaaS_Medium IaaS_Large"}

source scripts/comparison/_duct_common.sh

for dataset in $CLOUDTS_DATASETS; do
    init_dataset_run "$dataset"
    #                 ds        h         d_model n_layers d_ff patch stride drop lr    batch blw  ep patience
    run_dataset_cell "$dataset" "$HORIZON" 128     2        256  16    8      0.10 1e-4  16    0.20 30 10
    finish_dataset_run
done
