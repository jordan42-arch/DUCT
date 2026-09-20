set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run ETTh1

run_dataset_cell ETTh1 96  96  2 192 8  4  0.10 5e-5 16 0.20 12 6
run_dataset_cell ETTh1 192 96  2 192 8  4  0.10 5e-5 16 0.20 30 10
run_dataset_cell ETTh1 336 160 5 320 16 8  0.10 1e-4 16 0.20 30 10
run_dataset_cell ETTh1 720 160 4 320 8  4  0.10 5e-5 16 0.20 30 10

finish_dataset_run
