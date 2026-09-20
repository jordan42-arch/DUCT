#!/usr/bin/env bash
# Pretraining-objective and augmentation ablation for DUCT.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ABLATION_SUITE=${ABLATION_SUITE:-objective}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"full mae_only contrastive_only no_pretrain no_augmentation no_time_warp"}

source "$ROOT/scripts/ablation/_common.sh"
source "$ROOT/scripts/ablation/configs.sh"

OBJECTIVE_CELLS=${OBJECTIVE_CELLS:-"ETTh1:96 ETTh1:720 ETTm1:96 ETTm1:720 weather:96 weather:720 exchange_rate:96 exchange_rate:720 solar:96 solar:720"}

run_cell_list "$OBJECTIVE_CELLS"
