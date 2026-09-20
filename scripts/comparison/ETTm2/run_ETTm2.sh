set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run ETTm2

run_dataset_cell ETTm2 96  224 3 448 16 8  0.10 7e-5 16 0.20 12 6
run_dataset_cell ETTm2 192 96  2 192 24 12 0.10 7e-5 16 0.20 12 6
run_dataset_cell ETTm2 336 96  2 192 24 12 0.10 7e-5 16 0.20 12 6
run_dataset_cell ETTm2 720 96  2 192 16 8  0.10 5e-5 16 0.50 12 6

finish_dataset_run
