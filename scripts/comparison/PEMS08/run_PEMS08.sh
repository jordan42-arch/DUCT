set -euo pipefail

LOOKBACK=${LOOKBACK:-48}
source scripts/comparison/_duct_common.sh

init_dataset_run PEMS08

run_dataset_cell PEMS08 12  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS08 24  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS08 36  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS08 48  192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10

finish_dataset_run
