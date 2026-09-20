#!/bin/bash
# Comparison sweep on Electricity (321 variables).
set -e

BATCH=${BATCH:-4}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-4}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
run_comparison_group "electricity" "electricity" "electricity"
