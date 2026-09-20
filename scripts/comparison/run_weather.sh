#!/bin/bash
# Comparison sweep on Weather.
set -e

BATCH=${BATCH:-16}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-16}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
run_comparison_group "weather" "weather" "weather"
