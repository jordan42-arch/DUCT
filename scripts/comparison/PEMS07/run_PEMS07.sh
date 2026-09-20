set -euo pipefail

LOOKBACK=${LOOKBACK:-48}
source scripts/comparison/_duct_common.sh

init_dataset_run PEMS07

run_dataset_cell PEMS07 12  224 4 448 16 8 0.10 1.5e-4 16 0.20 12 6
run_dataset_cell PEMS07 24  224 4 448 16 8 0.10 1.5e-4 16 0.20 12 6
run_dataset_cell PEMS07 36  224 4 448 16 8 0.10 3e-5   16 0.20 12 6
run_dataset_cell PEMS07 48  96  2 192 12 6 0.10 7e-5   16 0.20 12 6

finish_dataset_run
