"""Fit the same out-of-sample Platt calibration used for the LightGBM
champion onto the two comparison models (XGBoost, scorecard).

This is NOT a retuning step: every model here is already fully fitted.
This script only loads them, scores validation, and fits a 1D logistic
regression (Platt, on logit(p)) on top - the same post-processing step,
applied identically to all three models for a fair comparison on test.

Methodology (identical to the LightGBM champion's calibration):
    fit on validation Jan-Sep 2014, evaluate on the Oct-Dec 2014 holdout
    (never seen while fitting), then refit the calibrator on the full
    2014 for deployment.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss

from src.models.gbm import fit_platt
from src.models.math_utils import apply_calibrator

VALIDATION_PARQUET = "data/processed/validation.parquet"
METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]


def time_split(validation: pd.DataFrame):
    issue_month = validation["issue_d"].values
    fit_mask = issue_month < np.datetime64("2014-10-01")
    return fit_mask, ~fit_mask


def calibrate_model(raw_pred: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, holdout_mask: np.ndarray) -> dict:
    fit_pred, fit_y = raw_pred[fit_mask], y[fit_mask]
    holdout_pred, holdout_y = raw_pred[holdout_mask], y[holdout_mask]

    platt = fit_platt(fit_pred, fit_y)
    holdout_calibrated = apply_calibrator(platt, holdout_pred)

    result = {
        "method": "platt",
        "fit_period": "2014-01 to 2014-09",
        "holdout_period": "2014-10 to 2014-12",
        "n_fit": int(fit_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "brier_uncalibrated_holdout": round(float(brier_score_loss(holdout_y, holdout_pred)), 5),
        "brier_platt_holdout": round(float(brier_score_loss(holdout_y, holdout_calibrated)), 5),
        "mean_predicted_pd_before_holdout_pct": round(float(holdout_pred.mean()) * 100, 4),
        "actual_default_rate_holdout_pct": round(float(holdout_y.mean()) * 100, 4),
        "mean_predicted_pd_after_holdout_pct": round(float(holdout_calibrated.mean()) * 100, 4),
    }

    # Refit on full 2014 for deployment.
    final_calibrator = fit_platt(raw_pred, y)
    final_pred = apply_calibrator(final_calibrator, raw_pred)
    result["final_calibrator_refit_on"] = "full 2014 (Jan-Dec)"
    result["mean_predicted_pd_before_full_2014_pct"] = round(float(raw_pred.mean()) * 100, 4)
    result["actual_default_rate_full_2014_pct"] = round(float(y.mean()) * 100, 4)
    result["mean_predicted_pd_after_full_2014_pct"] = round(float(final_pred.mean()) * 100, 4)

    return result, final_calibrator


def main() -> None:
    validation = pd.read_parquet(VALIDATION_PARQUET)
    y_validation = validation["target"].values
    fit_mask, holdout_mask = time_split(validation)

    # --- XGBoost ---
    xgb_dir = MODELS_DIR / "xgboost"
    xgb_booster = joblib.load(xgb_dir / "model.joblib")
    with open(xgb_dir / "features.json") as f:
        xgb_feats = json.load(f)
    xgb_val = validation.copy()
    for col in CATEGORICAL_COLUMNS:
        xgb_val[col] = xgb_val[col].astype("category")
    dval = xgb.DMatrix(xgb_val[xgb_feats], enable_categorical=True)
    xgb_pred = xgb_booster.predict(dval)

    xgb_result, xgb_calibrator = calibrate_model(xgb_pred, y_validation, fit_mask, holdout_mask)
    joblib.dump(xgb_calibrator, xgb_dir / "calibrator.joblib")
    with open(METRICS_DIR / "xgboost_calibration.json", "w") as f:
        json.dump(xgb_result, f, indent=2)
    print("=== XGBoost calibration ===")
    print(json.dumps(xgb_result, indent=2))

    # --- Scorecard ---
    sc_dir = MODELS_DIR / "scorecard"
    binning_process = joblib.load(sc_dir / "binning_process.joblib")
    logreg = joblib.load(sc_dir / "model.joblib")
    with open(sc_dir / "features.json") as f:
        sc_feats = json.load(f)
    sc_val_candidates = binning_process.variable_names
    woe_val = pd.DataFrame(
        binning_process.transform(validation[sc_val_candidates], metric="woe"),
        columns=sc_val_candidates,
        index=validation.index,
    )[sc_feats]
    sc_pred = logreg.predict_proba(woe_val)[:, 1]

    sc_result, sc_calibrator = calibrate_model(sc_pred, y_validation, fit_mask, holdout_mask)
    joblib.dump(sc_calibrator, sc_dir / "calibrator.joblib")
    with open(METRICS_DIR / "scorecard_calibration.json", "w") as f:
        json.dump(sc_result, f, indent=2)
    print("\n=== Scorecard calibration ===")
    print(json.dumps(sc_result, indent=2))


if __name__ == "__main__":
    main()
