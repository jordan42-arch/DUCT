#!/bin/bash
# Run only the Naive persistence baseline across selected datasets.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$ROOT/logs/comparison/by_model}
BATCH=${BATCH:-64}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-0}

source "$ROOT/scripts/comparison/_common.sh"

MODEL_DATASETS=${MODEL_DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic weather exchange_rate solar PEMS03 PEMS04 PEMS07 PEMS08"}
GROUP_NAME=${GROUP_NAME:-all_datasets}

run_single_model_group "Naive" "$GROUP_NAME" "$(echo "$MODEL_DATASETS" | tr ' ' ',')"
