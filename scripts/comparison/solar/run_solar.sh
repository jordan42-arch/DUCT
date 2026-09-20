set -euo pipefail

LOOKBACK=${LOOKBACK:-96}
source scripts/comparison/_duct_common.sh

init_dataset_run solar

run_dataset_cell solar 96  160 4 320 16 8 0.05 7e-5   16 0.20 12 6
run_dataset_cell solar 192 192 4 384 16 8 0.10 1.5e-4 16 0.20 12 6
run_dataset_cell solar 336 224 4 448 16 8 0.10 1e-4   16 0.20 12 6
run_dataset_cell solar 720 192 4 384 16 8 0.10 1.5e-4 16 0.20 12 6

finish_dataset_run
