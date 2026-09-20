#!/bin/bash
# Run all comparison sweeps. Override EPOCHS/PRETRAIN_EPOCHS/HORIZONS from the
# environment for smoke tests or shorter server jobs.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

bash "$SCRIPT_DIR/run_ett.sh"
bash "$SCRIPT_DIR/run_electricity.sh"
bash "$SCRIPT_DIR/run_traffic.sh"
bash "$SCRIPT_DIR/run_weather.sh"
bash "$SCRIPT_DIR/run_exchange_rate.sh"
bash "$SCRIPT_DIR/run_solar.sh"
bash "$SCRIPT_DIR/run_pems.sh"
bash "$SCRIPT_DIR/CloudTS/run_cloudts.sh"
