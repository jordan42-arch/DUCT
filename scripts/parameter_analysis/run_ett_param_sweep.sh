#!/bin/bash
# Legacy entrypoint. Prefer the explicit run_*_sweep.sh scripts.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SWEEP=${SWEEP:-temperature}

case "$SWEEP" in
    temperature)
        if [ -n "${VALUES:-}" ]; then
            export TEMPERATURE_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_temperature_sweep.sh"
        ;;
    contrastive_weight|loss_weight)
        if [ -n "${VALUES:-}" ]; then
            export CONTRASTIVE_WEIGHT_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_loss_weight_sweep.sh"
        ;;
    mask_ratio)
        if [ -n "${VALUES:-}" ]; then
            export MASK_RATIO_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_mask_ratio_sweep.sh"
        ;;
    branch_loss_weight|branch_weight)
        if [ -n "${VALUES:-}" ]; then
            export BRANCH_LOSS_WEIGHT_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_branch_loss_weight_sweep.sh"
        ;;
    fusion_weight|fusion_patch_weight)
        if [ -n "${VALUES:-}" ]; then
            export FUSION_PATCH_WEIGHT_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_fusion_weight_sweep.sh"
        ;;
    pretrain_epochs|pretrain_budget)
        if [ -n "${VALUES:-}" ]; then
            export PRETRAIN_BUDGET_VALUES="$VALUES"
        fi
        bash "$ROOT/scripts/parameter_analysis/run_pretrain_budget_sweep.sh"
        ;;
    pretrain_pool)
        bash "$ROOT/scripts/parameter_analysis/run_pretrain_pool_sweep.sh"
        ;;
    all)
        bash "$ROOT/scripts/parameter_analysis/run_all.sh"
        ;;
    *)
        echo "Unknown SWEEP=$SWEEP" >&2
        exit 2
        ;;
esac
