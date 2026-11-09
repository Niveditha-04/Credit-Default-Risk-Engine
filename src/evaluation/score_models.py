"""Shared scoring functions for all three frozen models. Used by both the
one-time test scoring (score_test.py) and can be dry-run against
validation to catch bugs before test is ever read.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.models.gbm import logit

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]


def load_frozen_config(path: str = "reports/metrics/frozen_model_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def score_lightgbm(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    booster = joblib.load(cfg["model_path"])
    calibrator = joblib.load(cfg["calibrator"]["path"])
    feats = cfg["features"]
    df2 = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df2[col] = df2[col].astype("category")
    raw_pred = booster.predict(df2[feats], num_iteration=booster.best_iteration)
    return calibrator.predict_proba(logit(raw_pred).reshape(-1, 1))[:, 1]


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
    return calibrator.predict_proba(logit(raw_pred).reshape(-1, 1))[:, 1]


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
    return calibrator.predict_proba(logit(raw_pred).reshape(-1, 1))[:, 1]


def score_all(df: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pd_lightgbm"] = score_lightgbm(df, frozen["lightgbm_chosen"])
    out["pd_xgboost"] = score_xgboost(df, frozen["xgboost"])
    out["pd_scorecard"] = score_scorecard(df, frozen["scorecard"])
    return out
