set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run exchange_rate

run_dataset_cell exchange_rate 96  128 2 256 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell exchange_rate 192 192 3 384 16 8 0.10 7e-5   8  0.20 30 10
run_dataset_cell exchange_rate 336 160 4 480 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell exchange_rate 720 160 5 320 16 8 0.10 1e-4   16 0.20 30 10

finish_dataset_run
