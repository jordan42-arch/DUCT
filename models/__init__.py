"""Model registry. One file per model; adding one means a file and an entry."""
from .DUCT import DUCT, PatchBranch, VariateBranch
from .DLinear import DLinear
from .FreTS import FreTS
from .iTransformer import iTransformer, iTransformerBaseline
from .Naive import Naive
from .PAttn import PAttn
from .PatchTST import PatchTST
from .TimeMixer import TimeMixer
from .TSMixer import TSMixer

# Models trained with the two-stage (pretrain -> finetune) pipeline.
PRETRAINED_MODELS = {"DUCT"}
# Subset of PRETRAINED_MODELS exposing y_patch / y_var / fusion_logits.
DUAL_BRANCH_MODELS = {"DUCT"}


def _build_wpmixer(n_vars, lookback, pred_len):
    # Keep PyWavelets optional for users running any other model.
    from .WPMixer import WPMixer
    return WPMixer(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        patch_len=16, stride=8, d_model=128,
    )

# Baseline hyperparameters are fixed by protocol, not swept, so they live here.
_BASELINE_BUILDERS = {
    "Naive": lambda n_vars, lookback, pred_len: Naive(n_vars, lookback, pred_len),
    "DLinear": lambda n_vars, lookback, pred_len: DLinear(n_vars, lookback, pred_len),
    "iTransformer": lambda n_vars, lookback, pred_len: iTransformer(n_vars, lookback, pred_len),
    "PatchTST": lambda n_vars, lookback, pred_len: PatchTST(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=3,
    ),
    "TimeMixer": lambda n_vars, lookback, pred_len: TimeMixer(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        d_model=32, n_layers=2, d_ff=64, dropout=0.1,
        down_sampling_layers=3, down_sampling_window=2,
    ),
    "FreTS": lambda n_vars, lookback, pred_len: FreTS(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        embed_size=128, hidden_size=256,
    ),
    "TSMixer": lambda n_vars, lookback, pred_len: TSMixer(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        d_model=128, n_layers=2, dropout=0.1,
    ),
    "PAttn": lambda n_vars, lookback, pred_len: PAttn(
        n_vars=n_vars, lookback=lookback, pred_len=pred_len,
        patch_len=16, stride=8, d_model=128, n_heads=8, d_ff=256,
    ),
    "WPMixer": _build_wpmixer,
}

SUPPORTED_MODELS = tuple(sorted(PRETRAINED_MODELS)) + tuple(_BASELINE_BUILDERS)


def build_baseline(name: str, n_vars: int, lookback: int, pred_len: int):
    """Instantiate a baseline by name."""
    if name not in _BASELINE_BUILDERS:
        raise ValueError(f"unknown baseline: {name}")
    return _BASELINE_BUILDERS[name](n_vars, lookback, pred_len)


__all__ = [
    "DUCT",
    "PatchBranch",
    "VariateBranch",
    "DLinear",
    "FreTS",
    "iTransformer",
    "iTransformerBaseline",
    "Naive",
    "PAttn",
    "PatchTST",
    "TimeMixer",
    "TSMixer",
    "PRETRAINED_MODELS",
    "DUAL_BRANCH_MODELS",
    "SUPPORTED_MODELS",
    "build_baseline",
]
