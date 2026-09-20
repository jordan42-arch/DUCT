#!/bin/bash
# Comparison sweep on Solar.
set -e

BATCH=${BATCH:-8}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-8}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
run_comparison_group "solar" "solar" "solar"
