"""Overlap metrics shared by the Visual Lab evaluation experiments (exp-012, exp-013)."""
import numpy as np


def dice(a, b):
    """Dice of two boolean masks; NaN when both are empty."""
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def recall_precision(pred, gt):
    tp = (pred & gt).sum()
    return (tp / gt.sum() if gt.sum() else np.nan,
            tp / pred.sum() if pred.sum() else np.nan)
