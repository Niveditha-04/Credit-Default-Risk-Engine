"""Shared scoring functions for all three frozen models. Used by both the
one-time test scoring (score_test.py) and can be dry-run against
validation to catch bugs before test is ever read.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.models.math_utils import apply_calibrator

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
V1_HISTORY_DIR = "data/processed/models/lightgbm_v1_history"


def load_frozen_config(path: str = "reports/metrics/frozen_model_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def v1_lightgbm_cfg(frozen: dict) -> dict:
    """v1's LightGBM scoring config, explicit, from lightgbm_v1_history.
    Every script that reports a v1 result (SHAP, test evaluation,
    decision policy, economics, drift, any post-hoc v1 section) calls
    this, never frozen["lightgbm_chosen"] directly: that key now
    describes whichever model is currently served (v2 as of this
    project's current state), not v1. model_path/calibrator.path are
    overridden to the actual backed-up v1 files under
    lightgbm_v1_history/, because lightgbm_v1_history's own path
    fields still point at the live lightgbm/ path it shared with v1
    before v2 was promoted over it."""
    cfg = dict(frozen["lightgbm_v1_history"])
    cfg["model_path"] = f"{V1_HISTORY_DIR}/model_chosen.joblib"
    cfg["calibrator"] = dict(cfg["calibrator"])
    cfg["calibrator"]["path"] = f"{V1_HISTORY_DIR}/calibrator.joblib"
    return cfg


def score_lightgbm(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    booster = joblib.load(cfg["model_path"])
    calibrator = joblib.load(cfg["calibrator"]["path"])
    feats = cfg["features"]
    df2 = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df2[col] = df2[col].astype("category")
    raw_pred = booster.predict(df2[feats], num_iteration=booster.best_iteration)
    return apply_calibrator(calibrator, raw_pred)


def score_xgboost(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    import xgboost as xgb

    booster = joblib.load(cfg["model_path"])
    calibrator = joblib.load(cfg["calibrator"]["path"])
    feats = cfg["features"]
    df2 = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df2[col] = df2[col].astype("category")
    dmat = xgb.DMatrix(df2[feats], enable_categorical=True)
    raw_pred = booster.predict(dmat)
    return apply_calibrator(calibrator, raw_pred)


def score_scorecard(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    binning_process = joblib.load(cfg["binning_process_path"])
    logreg = joblib.load(cfg["model_path"])
    calibrator = joblib.load(cfg["calibrator"]["path"])
    final_feats = cfg["features"]
    candidates = binning_process.variable_names

    woe = pd.DataFrame(
        binning_process.transform(df[candidates], metric="woe"), columns=candidates, index=df.index
    )[final_feats]
    raw_pred = logreg.predict_proba(woe)[:, 1]
    return apply_calibrator(calibrator, raw_pred)


def score_all(df: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Scores all three of v1's frozen models (lightgbm_v1_history, not
    lightgbm_chosen - this is the one-time test evaluation's scoring
    pass, a v1 output; xgboost and scorecard are untouched by the v2
    promotion, so they are unaffected either way)."""
    out = pd.DataFrame(index=df.index)
    out["pd_lightgbm"] = score_lightgbm(df, v1_lightgbm_cfg(frozen))
    out["pd_xgboost"] = score_xgboost(df, frozen["xgboost"])
    out["pd_scorecard"] = score_scorecard(df, frozen["scorecard"])
    return out
