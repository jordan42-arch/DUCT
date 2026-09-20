set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run weather

run_dataset_cell weather 96  192 3 384 16 8 0.10 5e-5   32 0.20 30 10
run_dataset_cell weather 192 192 3 384 16 8 0.10 5e-5   32 0.20 30 10
run_dataset_cell weather 336 160 4 320 8  4 0.10 5e-5   16 0.20 30 10
run_dataset_cell weather 720 160 4 320 8  4 0.10 5e-5   16 0.20 30 10

finish_dataset_run
