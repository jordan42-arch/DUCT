#!/usr/bin/env bash
# Fixed DUCT cell parameters from scripts/comparison/<dataset>/run_<dataset>.sh.

run_configured_cell() {
    local dataset=$1
    local horizon=$2
    case "${dataset}:${horizon}" in
        ETTh1:96) run_duct_ablation_cell ETTh1 96 96 96 2 192 8 4 0.10 5e-5 16 0.20 12 6 16 ;;
        ETTh1:192) run_duct_ablation_cell ETTh1 192 96 96 2 192 8 4 0.10 5e-5 16 0.20 30 10 16 ;;
        ETTh1:336) run_duct_ablation_cell ETTh1 336 96 160 5 320 16 8 0.10 1e-4 16 0.20 30 10 16 ;;
        ETTh1:720) run_duct_ablation_cell ETTh1 720 96 160 4 320 8 4 0.10 5e-5 16 0.20 30 10 16 ;;

        ETTh2:96) run_duct_ablation_cell ETTh2 96 96 96 2 192 16 8 0.10 1e-4 8 0.20 12 6 16 ;;
        ETTh2:192) run_duct_ablation_cell ETTh2 192 96 96 2 192 16 8 0.10 5e-5 8 0.20 30 10 16 ;;
        ETTh2:336) run_duct_ablation_cell ETTh2 336 96 128 2 256 16 8 0.10 1e-4 16 0.20 30 10 16 ;;
        ETTh2:720) run_duct_ablation_cell ETTh2 720 96 96 2 192 16 8 0.10 1e-4 8 0.20 12 6 16 ;;

        ETTm1:96) run_duct_ablation_cell ETTm1 96 96 192 4 576 16 8 0.10 3e-5 16 0.20 12 6 16 ;;
        ETTm1:192) run_duct_ablation_cell ETTm1 192 96 160 5 320 16 8 0.10 7e-5 16 0.20 12 6 16 ;;
        ETTm1:336) run_duct_ablation_cell ETTm1 336 96 192 4 576 16 8 0.10 3e-5 16 0.20 12 6 16 ;;
        ETTm1:720) run_duct_ablation_cell ETTm1 720 96 160 4 320 12 6 0.10 5e-5 16 0.20 12 6 16 ;;

        ETTm2:96) run_duct_ablation_cell ETTm2 96 96 224 3 448 16 8 0.10 7e-5 16 0.20 12 6 16 ;;
        ETTm2:192) run_duct_ablation_cell ETTm2 192 96 96 2 192 24 12 0.10 7e-5 16 0.20 12 6 16 ;;
        ETTm2:336) run_duct_ablation_cell ETTm2 336 96 96 2 192 24 12 0.10 7e-5 16 0.20 12 6 16 ;;
        ETTm2:720) run_duct_ablation_cell ETTm2 720 96 96 2 192 16 8 0.10 5e-5 16 0.50 12 6 16 ;;

        electricity:96) run_duct_ablation_cell electricity 96 96 192 4 384 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        electricity:192) run_duct_ablation_cell electricity 192 96 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        electricity:336) run_duct_ablation_cell electricity 336 96 224 3 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        electricity:720) run_duct_ablation_cell electricity 720 96 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;

        weather:96) run_duct_ablation_cell weather 96 96 192 3 384 16 8 0.10 5e-5 32 0.20 30 10 16 ;;
        weather:192) run_duct_ablation_cell weather 192 96 192 3 384 16 8 0.10 5e-5 32 0.20 30 10 16 ;;
        weather:336) run_duct_ablation_cell weather 336 96 160 4 320 8 4 0.10 5e-5 16 0.20 30 10 16 ;;
        weather:720) run_duct_ablation_cell weather 720 96 160 4 320 8 4 0.10 5e-5 16 0.20 30 10 16 ;;

        exchange_rate:96) run_duct_ablation_cell exchange_rate 96 96 128 2 256 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        exchange_rate:192) run_duct_ablation_cell exchange_rate 192 96 192 3 384 16 8 0.10 7e-5 8 0.20 30 10 16 ;;
        exchange_rate:336) run_duct_ablation_cell exchange_rate 336 96 160 4 480 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        exchange_rate:720) run_duct_ablation_cell exchange_rate 720 96 160 5 320 16 8 0.10 1e-4 16 0.20 30 10 16 ;;

        traffic:96) run_duct_ablation_cell traffic 96 96 160 3 320 16 8 0.10 7e-5 2 0.20 30 10 2 ;;
        traffic:192) run_duct_ablation_cell traffic 192 96 160 3 320 16 8 0.10 7e-5 2 0.20 30 10 2 ;;
        traffic:336) run_duct_ablation_cell traffic 336 96 160 3 320 16 8 0.10 7e-5 2 0.20 30 10 2 ;;
        traffic:720) run_duct_ablation_cell traffic 720 96 160 3 320 16 8 0.10 7e-5 2 0.20 30 10 2 ;;

        solar:96) run_duct_ablation_cell solar 96 96 160 4 320 16 8 0.05 7e-5 16 0.20 12 6 16 ;;
        solar:192) run_duct_ablation_cell solar 192 96 192 4 384 16 8 0.10 1.5e-4 16 0.20 12 6 16 ;;
        solar:336) run_duct_ablation_cell solar 336 96 224 4 448 16 8 0.10 1e-4 16 0.20 12 6 16 ;;
        solar:720) run_duct_ablation_cell solar 720 96 192 4 384 16 8 0.10 1.5e-4 16 0.20 12 6 16 ;;

        PEMS03:12) run_duct_ablation_cell PEMS03 12 48 192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS03:24) run_duct_ablation_cell PEMS03 24 48 192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS03:36) run_duct_ablation_cell PEMS03 36 48 192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS03:48) run_duct_ablation_cell PEMS03 48 48 192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;

        PEMS04:12) run_duct_ablation_cell PEMS04 12 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS04:24) run_duct_ablation_cell PEMS04 24 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS04:36) run_duct_ablation_cell PEMS04 36 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS04:48) run_duct_ablation_cell PEMS04 48 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;

        PEMS07:12) run_duct_ablation_cell PEMS07 12 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 12 6 16 ;;
        PEMS07:24) run_duct_ablation_cell PEMS07 24 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 12 6 16 ;;
        PEMS07:36) run_duct_ablation_cell PEMS07 36 48 224 4 448 16 8 0.10 3e-5 16 0.20 12 6 16 ;;
        PEMS07:48) run_duct_ablation_cell PEMS07 48 48 96 2 192 12 6 0.10 7e-5 16 0.20 12 6 16 ;;

        PEMS08:12) run_duct_ablation_cell PEMS08 12 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS08:24) run_duct_ablation_cell PEMS08 24 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS08:36) run_duct_ablation_cell PEMS08 36 48 224 4 448 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;
        PEMS08:48) run_duct_ablation_cell PEMS08 48 48 192 4 576 16 8 0.10 1.5e-4 16 0.20 30 10 16 ;;

        *)
            echo "No fixed DUCT config for ${dataset}:${horizon}" >&2
            return 2
            ;;
    esac
}

run_cell_list() {
    local cells=$1
    for cell in $cells; do
        local dataset=${cell%%:*}
        local horizon=${cell##*:}
        run_configured_cell "$dataset" "$horizon"
    done
}
