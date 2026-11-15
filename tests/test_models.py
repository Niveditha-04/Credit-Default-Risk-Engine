"""apply_calibrator must dispatch correctly on both calibrator types it
is ever handed in this project: Platt (sklearn LogisticRegression, fit on
the logit of p) and isotonic (sklearn IsotonicRegression, fit directly on
p). A real frozen calibrator is never mocked here - both are fit on tiny
synthetic data and apply_calibrator is checked against calling the
calibrator's own native method directly.

The champion-alias parity test at the bottom needs the real MLflow
registry and the real frozen model, so it is marked requires_data and
skipped in CI.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.math_utils import apply_calibrator, logit

RAW_PRED = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.85, 0.9, 0.95])
Y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])


def test_apply_calibrator_platt_matches_predict_proba_on_logit():
    platt = LogisticRegression().fit(logit(RAW_PRED).reshape(-1, 1), Y)
    expected = platt.predict_proba(logit(RAW_PRED).reshape(-1, 1))[:, 1]
    result = apply_calibrator(platt, RAW_PRED)
    assert np.array_equal(result, expected)


def test_apply_calibrator_isotonic_matches_predict():
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(RAW_PRED, Y)
    expected = isotonic.predict(RAW_PRED)
    result = apply_calibrator(isotonic, RAW_PRED)
    assert np.array_equal(result, expected)


def test_apply_calibrator_outputs_differ_between_platt_and_isotonic():
    # Sanity check that the two branches are not accidentally computing
    # the same thing - they are different calibration methods and should
    # not produce identical output on this non-trivial sample.
    platt = LogisticRegression().fit(logit(RAW_PRED).reshape(-1, 1), Y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(RAW_PRED, Y)
    platt_out = apply_calibrator(platt, RAW_PRED)
    iso_out = apply_calibrator(isotonic, RAW_PRED)
    assert not np.array_equal(platt_out, iso_out)


def test_apply_calibrator_platt_is_monotonic_in_the_raw_score():
    # A calibrator must never reorder applicants by risk: a higher raw
    # score must never come out with a lower calibrated PD, whatever the
    # calibration method. Platt is a strictly increasing sigmoid of
    # logit(p), so it is monotonic by construction - this proves
    # apply_calibrator's dispatch doesn't accidentally break that.
    platt = LogisticRegression().fit(logit(RAW_PRED).reshape(-1, 1), Y)
    ordered = np.sort(RAW_PRED)
    calibrated = apply_calibrator(platt, ordered)
    assert np.all(np.diff(calibrated) >= 0)


def test_apply_calibrator_isotonic_is_monotonic_in_the_raw_score():
    # Isotonic regression is fit under a monotonicity constraint, but
    # that constraint is on IsotonicRegression itself, not on
    # apply_calibrator's dispatch - this proves the dispatch preserves it.
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(RAW_PRED, Y)
    ordered = np.sort(RAW_PRED)
    calibrated = apply_calibrator(isotonic, ordered)
    assert np.all(np.diff(calibrated) >= 0)


# ---------------------------------------------------------------------
# Loads models:/credit_default_lightgbm_champion@champion from the
# MLflow registry and asserts its predictions equal the frozen model's
# predictions exactly (not just close) on the full validation set -
# proof that registering the model through MLflow's pyfunc flavor
# didn't change anything about how it scores a loan.
# ---------------------------------------------------------------------

MLFLOW_DB = Path("mlflow.db")
VALIDATION_PARQUET = Path("data/processed/validation.parquet")
FROZEN_CONFIG = Path("reports/metrics/frozen_model_config.json")

requires_real_mlflow = pytest.mark.skipif(
    not (MLFLOW_DB.exists() and VALIDATION_PARQUET.exists() and FROZEN_CONFIG.exists()),
    reason="real MLflow registry and model artifacts not present (expected outside a developer checkout)",
)


@pytest.mark.requires_data
@requires_real_mlflow
def test_registered_champion_alias_matches_frozen_model_exactly():
    import json

    import mlflow
    import pandas as pd

    from src.evaluation.score_models import score_lightgbm
    from src.models.mlflow_utils import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    loaded = mlflow.pyfunc.load_model("models:/credit_default_lightgbm_champion@champion")

    with open(FROZEN_CONFIG) as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]

    validation = pd.read_parquet(VALIDATION_PARQUET)
    registry_preds = np.array(loaded.predict(validation[cfg["features"]]))
    frozen_preds = score_lightgbm(validation, cfg)

    assert np.array_equal(registry_preds, frozen_preds), (
        f"registry model predictions do not exactly match the frozen model "
        f"(max abs diff {np.max(np.abs(registry_preds - frozen_preds))})"
    )
