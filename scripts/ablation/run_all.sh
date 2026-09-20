#!/usr/bin/env bash
# Run the complete DUCT ablation suite.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

bash "$ROOT/scripts/ablation/run_core_ablation.sh"
bash "$ROOT/scripts/ablation/run_highvar_ablation.sh"
bash "$ROOT/scripts/ablation/run_objective_ablation.sh"
bash "$ROOT/scripts/ablation/run_broad_ablation.sh"
