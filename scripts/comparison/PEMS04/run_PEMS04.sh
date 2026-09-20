set -euo pipefail

LOOKBACK=${LOOKBACK:-48}
source scripts/comparison/_duct_common.sh

init_dataset_run PEMS04

run_dataset_cell PEMS04 12  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS04 24  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS04 36  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell PEMS04 48  224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10

finish_dataset_run
