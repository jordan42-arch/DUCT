set -euo pipefail

LOOKBACK=${LOOKBACK:-48}
source scripts/comparison/_duct_common.sh

init_dataset_run PEMS03

run_dataset_cell PEMS03 12  192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS03 24  192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS03 36  192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS03 48  192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10

finish_dataset_run
