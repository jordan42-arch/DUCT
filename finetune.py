"""Train one (dataset, pred_len) cell. Also drives the baselines."""
import argparse

from data_provider import SUPPORTED_DATASETS
from exp import Exp_Finetune
from models import SUPPORTED_MODELS
from utils import DEFAULT_DATA_DIR


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--dataset", default="ETTh1", choices=SUPPORTED_DATASETS)
    ap.add_argument("--model", default="DUCT", choices=SUPPORTED_MODELS)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--branch_loss_weight", type=float, default=0.2,
                    help="Auxiliary branch loss weight for dual-branch DUCT finetuning.")
    ap.add_argument("--fusion_init_patch_weight", type=float, default=None,
                    help="Initial dual-branch DUCT patch-branch fusion weight in [0,1].")
    ap.add_argument("--freeze_fusion", action="store_true",
                    help="Freeze dual-branch DUCT fusion logits after optional initialization.")
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--results_dir", required=True,
                    help="Results dir; writes metrics.json and cells/<key>/metrics.json.")
    ap.add_argument("--save_predictions", action="store_true",
                    help="Also save pred_*.npz files. Disabled by default because "
                         "large datasets can produce multi-GB files.")
    return ap


def main():
    args = build_parser().parse_args()
    Exp_Finetune(args).run()


if __name__ == "__main__":
    main()
