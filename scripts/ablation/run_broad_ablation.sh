#!/usr/bin/env bash
# Broad DUCT ablation across dataset families.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ABLATION_SUITE=${ABLATION_SUITE:-broad}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"full mae_only no_pretrain no_branch_aux fixed_equal_fusion patch_only var_only"}

source "$ROOT/scripts/ablation/_common.sh"
source "$ROOT/scripts/ablation/configs.sh"

# Keep short and long horizons for each family. This is intentionally broader
# than the main-table core ablation because different datasets may expose
# different parts of the model story.
BROAD_CELLS=${BROAD_CELLS:-"ETTh1:96 ETTh1:720 ETTh2:96 ETTh2:720 ETTm1:96 ETTm1:720 ETTm2:96 ETTm2:720 electricity:96 electricity:720 weather:96 weather:720 exchange_rate:96 exchange_rate:720 solar:96 solar:720 traffic:96 traffic:720 PEMS03:12 PEMS03:48 PEMS04:12 PEMS04:48 PEMS07:12 PEMS07:48 PEMS08:12 PEMS08:48"}

run_cell_list "$BROAD_CELLS"
