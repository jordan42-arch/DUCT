set -euo pipefail

LOOKBACK=${LOOKBACK:-96}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-2}
source scripts/comparison/_duct_common.sh

init_dataset_run traffic

run_dataset_cell traffic 96  160 3 320 16 8 0.10 7e-5 2 0.20 30 10
run_dataset_cell traffic 192 160 3 320 16 8 0.10 7e-5 2 0.20 30 10
run_dataset_cell traffic 336 160 3 320 16 8 0.10 7e-5 2 0.20 30 10
run_dataset_cell traffic 720 160 3 320 16 8 0.10 7e-5 2 0.20 30 10

finish_dataset_run
