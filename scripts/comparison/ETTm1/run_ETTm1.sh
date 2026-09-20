set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run ETTm1

run_dataset_cell ETTm1 96  192 4 576 16 8 0.10 3e-5 16 0.20 12 6
run_dataset_cell ETTm1 192 160 5 320 16 8 0.10 7e-5 16 0.20 12 6
run_dataset_cell ETTm1 336 192 4 576 16 8 0.10 3e-5 16 0.20 12 6
run_dataset_cell ETTm1 720 160 4 320 12 6 0.10 5e-5 16 0.20 12 6

finish_dataset_run
