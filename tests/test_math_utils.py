"""apply_calibrator must dispatch correctly on both calibrator types it
is ever handed in this project: Platt (sklearn LogisticRegression, fit on
the logit of p) and isotonic (sklearn IsotonicRegression, fit directly on
p). A real frozen calibrator is never mocked here - both are fit on tiny
synthetic data and apply_calibrator is checked against calling the
calibrator's own native method directly.
"""

import numpy as np
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
