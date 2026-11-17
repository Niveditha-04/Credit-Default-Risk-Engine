"""Row-order sensitivity check for LightGBM, training in memory only and
never touching test or any frozen artifact.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.score_models import v1_lightgbm_cfg
from src.models.gbm import CATEGORICAL_COLUMNS
from src.report_paths import MODEL_CONFIG_PATH


def diagnose_lightgbm_row_order_bagging() -> None:
    """Does row order change the trained model when LightGBM's
    bagging_fraction/feature_fraction are < 1.0? Trains the frozen
    chosen hyperparameter config exactly ONCE per condition (no tuning
    search) with a fixed num_boost_round (no early stopping noise), on:
      A. train in id order,       original bagging/feature fractions
      B. train in shuffled order, original bagging/feature fractions
      C. train in id order,       bagging_fraction=1.0, feature_fraction=1.0
      D. train in shuffled order, bagging_fraction=1.0, feature_fraction=1.0

    If A != B but C == D, that confirms row order interacts with
    LightGBM's bagging/feature subsampling: it samples by physical row
    position for a given seed, not by a stable key, so the same seed
    selects different rows under a different row order.

    Does not touch any frozen artifact: trains in memory only, writes
    nothing to data/processed/models/."""
    # v1 explicitly (lightgbm_v1_history, not lightgbm_chosen): v2's
    # hyperparameters carry monotone_constraints that v1's don't, so
    # using lightgbm_chosen here would silently train a different model
    # than the one reports/metrics/models/training_order_sensitivity.json
    # (cited in README) actually describes.
    with open(MODEL_CONFIG_PATH) as f:
        frozen = json.load(f)
    cfg = v1_lightgbm_cfg(frozen)
    feats = cfg["features"]
    n_trees = cfg["n_trees"]
    base_params = dict(cfg["hyperparameters"])

    train = pd.read_parquet("data/processed/train.parquet")
    validation = pd.read_parquet("data/processed/validation.parquet")
    for col in CATEGORICAL_COLUMNS:
        train[col] = train[col].astype("category")
        validation[col] = validation[col].astype("category").cat.set_categories(train[col].cat.categories)

    y_val = validation["target"].to_numpy()
    X_val = validation[feats]

    train_id_order = train.sort_values("id").reset_index(drop=True)
    train_shuffled = train.sample(frac=1.0, random_state=123).reset_index(drop=True)

    def fit_and_score(train_df: pd.DataFrame, params: dict) -> float:
        dataset = lgb.Dataset(train_df[feats], label=train_df["target"].to_numpy(), categorical_feature=CATEGORICAL_COLUMNS)
        booster = lgb.train(params, dataset, num_boost_round=n_trees)
        preds = booster.predict(X_val)
        return float(roc_auc_score(y_val, preds))

    results = {}
    for label, df, bagging_ff in [
        ("A_id_order_original_fractions", train_id_order, (base_params["bagging_fraction"], base_params["feature_fraction"])),
        ("B_shuffled_original_fractions", train_shuffled, (base_params["bagging_fraction"], base_params["feature_fraction"])),
        ("C_id_order_no_subsampling", train_id_order, (1.0, 1.0)),
        ("D_shuffled_no_subsampling", train_shuffled, (1.0, 1.0)),
    ]:
        params = dict(base_params)
        params["bagging_fraction"], params["feature_fraction"] = bagging_ff
        auc = fit_and_score(df, params)
        results[label] = round(auc, 6)
        print(f"{label}: validation AUC = {auc:.6f}")

    print()
    print(json.dumps(results, indent=2))
    a_vs_b_differ = not np.isclose(results["A_id_order_original_fractions"], results["B_shuffled_original_fractions"])
    c_vs_d_differ = not np.isclose(results["C_id_order_no_subsampling"], results["D_shuffled_no_subsampling"])
    print()
    print(f"A vs B differ (order matters WITH subsampling): {a_vs_b_differ}")
    print(f"C vs D differ (order matters WITHOUT subsampling): {c_vs_d_differ}")
    if a_vs_b_differ and not c_vs_d_differ:
        print("CONFIRMED: row order + bagging/feature subsampling is the cause.")
    elif not a_vs_b_differ and not c_vs_d_differ:
        print("Row order did not affect either condition here - cause is something else.")
    else:
        print("Unclear from this single run - C vs D also differ, so subsampling alone doesn't explain it.")


if __name__ == "__main__":
    diagnose_lightgbm_row_order_bagging()
