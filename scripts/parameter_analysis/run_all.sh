#!/bin/bash
# Run the complete parameter-analysis plan.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

bash "$ROOT/scripts/parameter_analysis/run_temperature_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_loss_weight_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_branch_loss_weight_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_fusion_weight_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_mask_ratio_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_pretrain_budget_sweep.sh"
bash "$ROOT/scripts/parameter_analysis/run_pretrain_pool_sweep.sh"
