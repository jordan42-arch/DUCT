#!/usr/bin/env bash
# Main paper ablation for DUCT on representative ETT cells.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ABLATION_SUITE=${ABLATION_SUITE:-core}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"full mae_only contrastive_only no_pretrain no_branch_aux fixed_equal_fusion patch_only var_only no_augmentation no_time_warp"}

source "$ROOT/scripts/ablation/_common.sh"
source "$ROOT/scripts/ablation/configs.sh"

# Short and long horizons cover the two most common failure modes:
# local short-term fit and long-horizon robustness.
CORE_CELLS=${CORE_CELLS:-"ETTh1:96 ETTh1:720 ETTh2:96 ETTh2:720 ETTm1:96 ETTm1:720 ETTm2:96 ETTm2:720"}

run_cell_list "$CORE_CELLS"
