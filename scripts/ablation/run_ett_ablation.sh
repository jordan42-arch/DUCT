#!/usr/bin/env bash
# Backward-compatible entrypoint for the main ETT ablation.
set -euo pipefail

bash "$(cd "$(dirname "$0")" && pwd)/run_core_ablation.sh"
