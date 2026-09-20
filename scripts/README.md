# Experiment Scripts

Scripts are grouped by experiment purpose:

- `comparison/`: baseline comparison sweeps, split by dataset family.
  - `comparison/models/`: run Naive, DLinear, iTransformer, PatchTST, or DUCT separately.
  - `comparison/models/run_recent_tslib.sh`: run TimeMixer, PAttn, WPMixer,
    FreTS, and TSMixer over all 58 standard, PEMS, and CloudTS cells with
    restart-safe skipping.
- `ablation/`: DUCT component ablations.
  - `run_core_ablation.sh`: DUCT objective, branch/fusion, and augmentation ablations on ETT.
  - `run_highvar_ablation.sh`: high-variate branch/fusion ablations for the variate-interaction claim.
  - `run_ett_ablation.sh`: backward-compatible wrapper for `run_core_ablation.sh`.
- `parameter_analysis/`: hyperparameter sweeps and analysis utilities.
  - `run_temperature_sweep.sh`: InfoNCE temperature `tau`.
  - `run_loss_weight_sweep.sh`: contrastive loss weight `beta`.
  - `run_branch_loss_weight_sweep.sh`: auxiliary branch prediction loss weight.
  - `run_fusion_weight_sweep.sh`: fixed patch/variate fusion weights.
  - `run_mask_ratio_sweep.sh`: MAE patch mask ratio.
  - `run_pretrain_budget_sweep.sh`: number of pretraining epochs.
  - `run_pretrain_pool_sweep.sh`: source datasets used for pretraining.
  - `summarize_sweeps.py`: flatten completed sweeps into `summary.csv`.

All scripts are launched from the repository root or by passing the script path
to `bash`. Common environment overrides:

```bash
EPOCHS=3 PRETRAIN_EPOCHS=3 HORIZONS="96" bash scripts/comparison/run_ett.sh
MODEL_DATASETS="ETTm1 traffic" HORIZONS="96" bash scripts/comparison/models/run_patchtst.sh
MODEL_DATASETS="ETTm1 traffic" HORIZONS="96" bash scripts/comparison/models/run_duct.sh
PYTHON=python3 BATCH=4 bash scripts/comparison/run_electricity.sh
SWEEP=mask_ratio VALUES="0.2 0.4" bash scripts/parameter_analysis/run_ett_param_sweep.sh
PARAM_EVAL_DATASETS="ETTm1 ETTh2" PARAM_HORIZONS="96 720" bash scripts/parameter_analysis/run_temperature_sweep.sh
BRANCH_LOSS_WEIGHT_VALUES="0 0.2 0.5" bash scripts/parameter_analysis/run_branch_loss_weight_sweep.sh
FUSION_PATCH_WEIGHT_VALUES="0 0.5 1" bash scripts/parameter_analysis/run_fusion_weight_sweep.sh
python scripts/parameter_analysis/summarize_sweeps.py --root logs/parameter_analysis
```

Default outputs:

- comparison: `logs/comparison/<dataset-group>/`
- dataset comparison: `logs/comparison/by_dataset/<dataset-group>/`
- single comparison model: `logs/comparison/by_model/<model>/<group>/`
- ablation: `logs/ablation/<ablation-group>/seed_<seed>/<section>/<ablation-name>/`
- parameter analysis: `logs/parameter_analysis/<suite>/seed_<seed>/<value>/`
- checkpoints: matching subdirectories under `checkpoints/`

Each run directory contains:

- `run.log`: stdout/stderr from pretraining and finetuning commands.
- `run_config.env`: the dataset/model/horizon/seed/checkpoint settings for the run.
- `metrics.json`: aggregate metrics written by `finetune.py`.
- `cells/`: per-cell metrics; prediction `.npz` files are only written when `--save_predictions` is passed.

The DUCT ablation suites separate four questions:

- Objective: `full_global`, `global_mae_only`, `global_contrastive_only`, and `scratch_no_pretrain`.
- Branch contribution: `patch_only_forecast`, `var_only_forecast`, and the learned dual-branch model.
- Fusion: learned fusion versus `fixed_equal_fusion`.
- Branch supervision: default auxiliary branch losses versus `no_branch_aux`.
- Augmentation: `all_augmentations`, `no_augmentation`, and `no_time_warp`.

The parameter-analysis suites answer five selection questions:

- `temperature`: which InfoNCE temperature is stable.
- `loss_weight`: how much contrastive loss to mix into MAE.
- `branch_loss_weight`: how strongly to supervise each branch prediction.
- `fusion_weight`: which fixed patch/variate mixture is best.
- `mask_ratio`: how much patch masking to use.
- `pretrain_budget`: whether pretraining is undertrained or saturated.
- `pretrain_pool`: which ETT source pool produces the best transfer.
