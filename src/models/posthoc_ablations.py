"""Post-hoc ablations against the frozen LightGBM champion's own
hyperparameters, with no tuning search at all: the search is not rerun,
the champion's (num_leaves, learning_rate, min_child_samples,
feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2)
are reused exactly as frozen. Validation (2014) is the primary number
for each ablation, same as every other model-selection decision in this
project; test (2015) AUC is reported alongside but labeled post-hoc,
since none of these ablations were part of the champion-selection
process test was frozen against.

Four ablations:
  a. LightGBM on the scorecard's 17 features only.
  b. A WOE logistic regression on all 63 scorecard candidate features
     (no correlation/VIF/sign-correction reduction - the champion
     scorecard's reduction steps are skipped on purpose here).
  c. LightGBM on the champion's 69 features plus sub_grade.
  d. LightGBM on the champion's 69 features with monotone constraints
     on 8 features where the business direction is unambiguous.

This never touches data/processed/models/ or frozen_model_config.json.
Every model here is fit fresh in memory and only ever written to
reports/metrics/.
"""

import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list
from src.models.scorecard import candidate_features

METRICS_DIR = Path("reports/metrics")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
SEED = 42
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

# The frozen champion's own hyperparameters (reports/metrics/frozen_model_config.json).
# No tuning search is run anywhere in this file.
FROZEN_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "seed": SEED,
    "verbosity": -1,
    "num_leaves": 15,
    "learning_rate": 0.08,
    "min_child_samples": 1500,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}

MONOTONE_DIRECTIONS = {
    "fico_mean": -1,
    "dti": 1,
    "loan_to_income": 1,
    "revol_util": 1,
    "bc_util": 1,
    "inq_last_6mths": 1,
    "acc_open_past_24mths": 1,
    "annual_inc": -1,
}


def load_splits(extra_join_sub_grade: bool = False):
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    if extra_join_sub_grade:
        con = duckdb.connect()
        sub_grade = con.execute(f"SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')").fetchdf()
        train = train.merge(sub_grade, on="id", how="left")
        validation = validation.merge(sub_grade, on="id", how="left")
        test = test.merge(sub_grade, on="id", how="left")

    cat_cols = list(CATEGORICAL_COLUMNS)
    if extra_join_sub_grade:
        cat_cols = cat_cols + ["sub_grade"]
    for col in cat_cols:
        train[col] = train[col].astype("category")
        validation[col] = validation[col].astype("category").cat.set_categories(train[col].cat.categories)
        test[col] = test[col].astype("category").cat.set_categories(train[col].cat.categories)

    return train, validation, test


def fit_frozen_lightgbm(train, validation, feats, monotone_constraints=None):
    dataset_cat_cols = [c for c in CATEGORICAL_COLUMNS + ["sub_grade"] if c in feats]
    lgb_train = lgb.Dataset(
        train[feats], label=train["target"].values, categorical_feature=dataset_cat_cols,
        free_raw_data=False, params={"feature_pre_filter": False},
    )
    lgb_val = lgb.Dataset(validation[feats], label=validation["target"].values, reference=lgb_train, free_raw_data=False)

    params = dict(FROZEN_LGB_PARAMS)
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def auc_on(booster_or_pred_fn, df, feats):
    pred = booster_or_pred_fn(df[feats])
    return float(roc_auc_score(df["target"].values, pred))


def ablation_a_scorecard_features(train, validation, test):
    feats = list(candidate_features_17())
    booster = fit_frozen_lightgbm(train, validation, feats)
    pred_fn = lambda X: booster.predict(X, num_iteration=booster.best_iteration)
    return {
        "ablation": "a_lightgbm_on_scorecard_17_features",
        "n_features": len(feats),
        "features": feats,
        "n_trees": booster.best_iteration,
        "validation_auc": round(auc_on(pred_fn, validation, feats), 4),
        "test_auc_post_hoc": round(auc_on(pred_fn, test, feats), 4),
    }


def candidate_features_17() -> list[str]:
    with open(METRICS_DIR / "feature_selection_log.json") as f:
        return json.load(f)["final_features"]


def ablation_b_woe_logreg_63_candidates(train, validation, test):
    candidates = candidate_features()
    cat_in_candidates = [c for c in CATEGORICAL_COLUMNS if c in candidates]

    bp = BinningProcess(variable_names=candidates, categorical_variables=cat_in_candidates)
    bp.fit(train[candidates], train["target"].values)

    woe_train = pd.DataFrame(bp.transform(train[candidates], metric="woe"), columns=candidates, index=train.index)
    woe_validation = pd.DataFrame(
        bp.transform(validation[candidates], metric="woe"), columns=candidates, index=validation.index
    )
    woe_test = pd.DataFrame(bp.transform(test[candidates], metric="woe"), columns=candidates, index=test.index)

    clf = LogisticRegression(penalty=None, max_iter=1000)
    clf.fit(woe_train, train["target"].values)

    coefs = clf.coef_[0]
    n_positive_coefs = int((coefs > 0).sum())

    val_pred = clf.predict_proba(woe_validation)[:, 1]
    test_pred = clf.predict_proba(woe_test)[:, 1]
    return {
        "ablation": "b_woe_logreg_on_63_candidates_no_reduction",
        "n_features": len(candidates),
        "n_positive_coefficients_sign_mismatch": n_positive_coefs,
        "validation_auc": round(float(roc_auc_score(validation["target"].values, val_pred)), 4),
        "test_auc_post_hoc": round(float(roc_auc_score(test["target"].values, test_pred)), 4),
    }


def ablation_c_sub_grade_added(train, validation, test, champion_feats):
    feats = champion_feats + ["sub_grade"]
    booster = fit_frozen_lightgbm(train, validation, feats)
    pred_fn = lambda X: booster.predict(X, num_iteration=booster.best_iteration)
    return {
        "ablation": "c_lightgbm_champion_features_plus_sub_grade",
        "n_features": len(feats),
        "n_trees": booster.best_iteration,
        "validation_auc": round(auc_on(pred_fn, validation, feats), 4),
        "test_auc_post_hoc": round(auc_on(pred_fn, test, feats), 4),
    }, booster, feats


def ablation_d_monotone_constraints(train, validation, test, champion_feats):
    feats = list(champion_feats)
    monotone_constraints = [MONOTONE_DIRECTIONS.get(f, 0) for f in feats]
    booster = fit_frozen_lightgbm(train, validation, feats, monotone_constraints=monotone_constraints)
    pred_fn = lambda X: booster.predict(X, num_iteration=booster.best_iteration)
    result = {
        "ablation": "d_lightgbm_champion_features_monotone_constraints",
        "n_features": len(feats),
        "n_trees": booster.best_iteration,
        "monotone_constraints": {f: d for f, d in zip(feats, monotone_constraints) if d != 0},
        "validation_auc": round(auc_on(pred_fn, validation, feats), 4),
        "test_auc_post_hoc": round(auc_on(pred_fn, test, feats), 4),
    }
    return result, booster, feats


def income_monotonicity_check(booster, feats, test, label: str) -> dict:
    """Mean SHAP (pred_contrib) for annual_inc by income bucket, on test,
    post-hoc. Compared against the champion's existing finding
    (reports/metrics/shap_followup_checks_summary.json): annual_inc alone
    is most protective around $90K-110K, then weakens above that -
    checks whether that non-monotonicity survives a hard monotone
    constraint on annual_inc."""
    X = test[feats].copy()
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    contrib = booster.predict(X, pred_contrib=True, num_iteration=booster.best_iteration)
    annual_inc_idx = feats.index("annual_inc")
    shap_annual_inc = contrib[:, annual_inc_idx]

    bins = [0, 30000, 50000, 70000, 90000, 110000, 150000, 200000, 300000, 500000, 1e7]
    income_df = pd.DataFrame({"annual_inc": test["annual_inc"].values, "shap": shap_annual_inc})
    income_df["bucket"] = pd.cut(income_df["annual_inc"], bins=bins)
    means = income_df.groupby("bucket", observed=True)["shap"].mean()
    values = means.values
    diffs = np.diff(values)
    is_monotonic = bool(np.all(diffs <= 1e-9) or np.all(diffs >= -1e-9))
    # Reversal magnitude: once SHAP hits its most protective (most negative)
    # bucket, how much does it climb back up afterward? A strict monotonic
    # check alone treats any wiggle as "not monotonic" without saying how
    # big that wiggle is, which is the actual business question here.
    peak_idx = int(np.argmin(values))
    after_peak = values[peak_idx:]
    reversal_magnitude = float(after_peak.max() - after_peak.min())
    return {
        "model": label,
        "mean_shap_annual_inc_by_income_bucket": {str(k): round(float(v), 4) for k, v in means.items()},
        "monotonic_across_buckets": is_monotonic,
        "most_protective_bucket": str(means.index[peak_idx]),
        "reversal_magnitude_after_peak": round(reversal_magnitude, 4),
    }


def main() -> None:
    train, validation, test = load_splits(extra_join_sub_grade=False)
    train_sg, validation_sg, test_sg = load_splits(extra_join_sub_grade=True)
    champion_feats = feature_list(list(train.columns))

    log_access(
        "posthoc_ablations.py",
        len(test),
        "Post-hoc: reads test.parquet to report test AUC for 4 ablations fit with the "
        "frozen champion's own hyperparameters (no tuning search). Validation (2014) is "
        "the primary number for every ablation; test AUC here is post-hoc only, never "
        "used to pick or tune any of these ablations. No change to the frozen champion.",
    )

    print("=== Ablation a: LightGBM on the scorecard's 17 features ===")
    result_a = ablation_a_scorecard_features(train, validation, test)
    print(json.dumps(result_a, indent=2))

    print("\n=== Ablation b: WOE logistic regression on all 63 scorecard candidates ===")
    result_b = ablation_b_woe_logreg_63_candidates(train, validation, test)
    print(json.dumps(result_b, indent=2))

    print("\n=== Ablation c: LightGBM, champion features + sub_grade ===")
    result_c, _booster_c, _feats_c = ablation_c_sub_grade_added(train_sg, validation_sg, test_sg, champion_feats)
    print(json.dumps(result_c, indent=2))

    print("\n=== Ablation d: LightGBM, champion features, monotone constraints ===")
    result_d, booster_d, feats_d = ablation_d_monotone_constraints(train, validation, test, champion_feats)
    print(json.dumps(result_d, indent=2))

    print("\n=== Income monotonicity check (post-hoc, test) ===")
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    champion_booster = joblib.load(frozen["lightgbm_chosen"]["model_path"])
    champion_feats = frozen["lightgbm_chosen"]["features"]
    champion_check = income_monotonicity_check(champion_booster, champion_feats, test, "champion")
    ablation_d_check = income_monotonicity_check(booster_d, feats_d, test, "ablation_d_monotone")
    print(json.dumps(champion_check, indent=2))
    print(json.dumps(ablation_d_check, indent=2))

    with open(METRICS_DIR / "frozen_model_config.json") as f:
        champion_val_auc = json.load(f)["lightgbm_chosen"]["train_validation_metrics"]["validation_auc"]
    with open(METRICS_DIR / "test_core_metrics.json") as f:
        champion_test_auc = json.load(f)["lightgbm"]["auc"]

    summary = {
        "label": "post-hoc",
        "note": (
            "Every ablation reuses the frozen champion's own hyperparameters "
            "(num_leaves=15, learning_rate=0.08, min_child_samples=1500, "
            "feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, "
            "lambda_l1=0.1, lambda_l2=1.0); none of these ablations rerun a tuning "
            "search. Validation (2014) is the primary number for every ablation, same "
            "as the champion-selection process. Test (2015) AUC is reported for context "
            "only and is explicitly post-hoc: none of these ablations were fit, chosen, "
            "or compared against each other before test was touched for the main evaluation."
        ),
        "champion_validation_auc": champion_val_auc,
        "champion_test_auc_reference": champion_test_auc,
        "ablations": [result_a, result_b, result_c, result_d],
        "income_monotonicity_check": {
            "champion": champion_check,
            "ablation_d_monotone_constraints": ablation_d_check,
            "reversal_shrinks_under_constraint": (
                ablation_d_check["reversal_magnitude_after_peak"] < champion_check["reversal_magnitude_after_peak"]
            ),
            "reversal_magnitude_ratio_d_over_champion": round(
                ablation_d_check["reversal_magnitude_after_peak"] / champion_check["reversal_magnitude_after_peak"], 4
            ),
        },
    }
    for res in summary["ablations"]:
        res["validation_auc_minus_champion"] = round(res["validation_auc"] - champion_val_auc, 4)
        res["test_auc_minus_champion_post_hoc"] = round(res["test_auc_post_hoc"] - champion_test_auc, 4)

    with open(METRICS_DIR / "posthoc_ablations_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
