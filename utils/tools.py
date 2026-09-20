"""Training helpers."""
import numpy as np
import torch

# ETT-small, not datasets/, so pre-multi-dataset commands still resolve.
DEFAULT_DATA_DIR = "datasets/ETT-small"


def set_seed(s: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Best-score tracker; `step` returns True when training should stop."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("inf")
        self.best_aux = None
        self.best_state = None
        self.wait = 0

    def step(self, score: float, model, aux=None) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.best_aux = aux
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    @property
    def triggered_score(self):
        return None if self.best_score == float("inf") else self.best_score
