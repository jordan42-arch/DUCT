#!/bin/bash
# Comparison sweep on Traffic (862 variables). Defaults use a small batch because
# both iTransformer attention and DUCT contrastive pretraining scale with V.
set -e

BATCH=${BATCH:-2}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-2}

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
run_comparison_group "traffic" "traffic" "traffic"
