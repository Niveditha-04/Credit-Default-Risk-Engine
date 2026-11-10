"""Optional, validation-only check (2026-11-10 incident follow-up):
retrain XGBoost with the exact frozen hyperparameters on id-ordered
train, to see how much the same row-order sensitivity found in LightGBM
(subsample=0.6 is XGBoost's row-subsampling analog of bagging_fraction)
also affects XGBoost's validation AUC. Writes only to
data/processed/models/candidates/ - never touches the frozen xgboost
artifacts, and never scores test. This is a diagnostic rerun, not a
replacement for the frozen (and now unrecoverable-bit-exact) XGBoost
model.
"""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.models.gbm import CATEGORICAL_COLUMNS, new_run_id

METRICS_DIR = Path("reports/metrics")
CANDIDATES_DIR = Path("data/processed/models/candidates")


def main() -> None:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    xgb_cfg = frozen["xgboost"]
    feats = xgb_cfg["features"]
    params = dict(xgb_cfg["hyperparameters"])
    n_trees = xgb_cfg["n_trees"]
    frozen_validation_auc = xgb_cfg["validation_auc"]

    train = pd.read_parquet("data/processed/train.parquet").sort_values("id").reset_index(drop=True)
    validation = pd.read_parquet("data/processed/validation.parquet")
    for col in CATEGORICAL_COLUMNS:
        train[col] = train[col].astype("category")
        validation[col] = validation[col].astype("category").cat.set_categories(train[col].cat.categories)

    dtrain = xgb.DMatrix(train[feats], label=train["target"].to_numpy(), enable_categorical=True)
    booster = xgb.train(params, dtrain, num_boost_round=n_trees)

    dval = xgb.DMatrix(validation[feats], enable_categorical=True)
    preds = booster.predict(dval)
    validation_auc = round(float(roc_auc_score(validation["target"].to_numpy(), preds)), 4)

    run_id = new_run_id()
    candidate_dir = CANDIDATES_DIR / run_id / "xgboost_validation_only_check"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(candidate_dir / "model.json"))

    result = {
        "label": "validation-only row-order sensitivity check, not a replacement model - test was never scored",
        "row_order": "id-sorted (ascending)",
        "n_trees": n_trees,
        "validation_auc_this_run": validation_auc,
        "validation_auc_frozen": frozen_validation_auc,
        "diff": round(validation_auc - frozen_validation_auc, 4),
        "candidate_dir": str(candidate_dir),
    }
    with open(METRICS_DIR / "xgboost_order_sensitivity_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
