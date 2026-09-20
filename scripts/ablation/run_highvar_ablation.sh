#!/usr/bin/env bash
# High-variate ablation: emphasize the variate branch and learned fusion.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ABLATION_SUITE=${ABLATION_SUITE:-highvar}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"full no_pretrain no_branch_aux fixed_equal_fusion patch_only var_only mae_only"}

source "$ROOT/scripts/ablation/_common.sh"
source "$ROOT/scripts/ablation/configs.sh"

HIGHVAR_CELLS=${HIGHVAR_CELLS:-"electricity:96 electricity:720 traffic:96 traffic:720 PEMS03:12 PEMS03:48 PEMS04:12 PEMS04:48 PEMS08:12 PEMS08:48 PEMS07:12 PEMS07:48"}

run_cell_list "$HIGHVAR_CELLS"
