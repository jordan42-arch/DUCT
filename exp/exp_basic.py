"""Base experiment: what both stages share. Subclasses implement `run`."""
import torch

from models import PRETRAINED_MODELS, DUCT, DUAL_BRANCH_MODELS
from utils import pick_device, set_seed


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.device = pick_device()

    # ------------------------------------------------------------------ model

    def uses_pretraining(self):
        return self.args.model in PRETRAINED_MODELS

    def is_dual_branch(self):
        return self.args.model in DUAL_BRANCH_MODELS

    def build_duct(self, n_vars, pred_len, **overrides):
        """Construct the DUCT model with the shared architecture arguments."""
        args = self.args
        kwargs = dict(
            n_vars=n_vars,
            lookback=args.lookback,
            pred_len=pred_len,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
        kwargs.update(overrides)
        return DUCT(**kwargs)

    def build_model(self, n_vars):
        raise NotImplementedError

    # -------------------------------------------------------------- optimizer

    def build_optimizer(self, model, epochs):
        opt = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return opt, sched

    @staticmethod
    def optimizer_step(opt, loss, model):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # -------------------------------------------------------------------- run

    def run(self):
        raise NotImplementedError
