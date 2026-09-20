set -euo pipefail

source scripts/comparison/_duct_common.sh

init_dataset_run electricity

run_dataset_cell electricity 96  192 4 384 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell electricity 192 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell electricity 336 224 3 448 16 8 0.10 1.5e-4 16 0.20 30 10
run_dataset_cell electricity 720 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10

finish_dataset_run
