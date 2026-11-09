"""Tiny numpy-only helpers shared by training (gbm.py) and scoring
(score_models.py, imported by the API). Kept in its own module, separate
from gbm.py, so importing it does not pull in lightgbm/xgboost/sklearn -
those are only needed for training, not for turning a probability into a
logit at serving time.
"""

import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))
