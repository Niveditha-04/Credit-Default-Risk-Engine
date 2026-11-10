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


def apply_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    """Scores a raw predicted probability through a fitted calibrator,
    Platt or isotonic, without the caller needing to know which one it
    got. Platt (sklearn LogisticRegression) exposes predict_proba and is
    applied to the logit of p; isotonic (sklearn IsotonicRegression) has
    no predict_proba and is applied directly to p. Every scoring path
    (score_models, decision_policy, the MLflow pyfunc wrapper, the API)
    calls this one function, so a run that deploys an isotonic calibrator
    scores correctly everywhere with no per-caller branching."""
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    return calibrator.predict(p)
