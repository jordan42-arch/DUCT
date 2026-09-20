"""DUCT pretraining: InfoNCE + MAE. Writes a backbone-only checkpoint."""
import argparse

from exp import Exp_Pretrain
from models import PRETRAINED_MODELS
from utils import DEFAULT_DATA_DIR


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DUCT", choices=tuple(sorted(PRETRAINED_MODELS)))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1")
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--save_path", default="checkpoints/pretrained_backbone.pt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--mae_weight", type=float, default=1.0)
    ap.add_argument("--contrastive_weight", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--augmentation_mode", default="all",
                    help="all|none|noise|scaling|time_warp or comma-separated subsets")
    # ablations
    ap.add_argument("--no_contrastive", action="store_true")
    ap.add_argument("--no_mae", action="store_true")
    ap.add_argument("--no_augmentation", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    Exp_Pretrain(args).run()


if __name__ == "__main__":
    main()
