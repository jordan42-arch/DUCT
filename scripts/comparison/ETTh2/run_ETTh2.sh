set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run ETTh2

run_dataset_cell ETTh2 96  96  2 192 16 8 0.10 1e-4 8  0.20 12 6
run_dataset_cell ETTh2 192 96  2 192 16 8 0.10 5e-5 8  0.20 30 10
run_dataset_cell ETTh2 336 128 2 256 16 8 0.10 1e-4 16 0.20 30 10
run_dataset_cell ETTh2 720 96  2 192 16 8 0.10 1e-4 8  0.20 12 6

finish_dataset_run
