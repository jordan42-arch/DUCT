#!/bin/bash
# Comparison sweep on Exchange Rate.
set -e

BATCH=${BATCH:-16}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-16}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
run_comparison_group "exchange_rate" "exchange_rate" "exchange_rate"
