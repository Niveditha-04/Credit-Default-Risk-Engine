"""Post-hoc checks around v2, grouped by topic in one module instead of
three separate scripts: the ablations that led to v2's monotone
-constrained design, the pre-registered evaluation that adopted it, a
robustness check on that decision's label, and a later, separate check
of the now-served v2 model directly against test (2015).

None of this touches data/processed/models/ or frozen_model_config.json
except by reading it; every model refit here is fit fresh, in memory,
and only ever written to reports/metrics/.

Sections, run in order by main():
  1. Ablations: four variants of the champion's own hyperparameters (no
     tuning search), validation (2014) primary, test (2015) post-hoc.
  2. v2 evaluation: v1 (frozen, unchanged) vs. a freshly-fit v2 against
     the pre-registered adoption rule in docs/v2_decision_rule.md, on
     2016 vintages.
  3. v2 label robustness: the same evaluation rerun with a second,
     "corrected" 24-month label, chosen after section 2 already ran, to
     check whether the adoption decision depends on it.
  4. v2 on test (2015): the actual served v2 model (its saved booster
     and calibrator, not a fresh refit) scored on test, post-hoc -
     v2 was already selected and already calibrated on 2015 before this
     check, so 2015 is not a clean holdout for it.
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

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.evaluation.test_evaluation import (
    ks_statistic,
    paired_bootstrap,
    single_model_bootstrap_ci,
    sub_grade_ordinal,
)
from src.models.gbm import feature_list
from src.models.scorecard import candidate_features

METRICS_DIR = Path("reports/metrics")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
REPLAY_PARQUET = "data/processed/replay.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
SEED = 42

# The frozen champion's own hyperparameters (reports/metrics/frozen_model_config.json).
# No tuning search is run anywhere in this module. Shared by the
# ablations (section 1) and v2 (sections 2-3), which reuse them exactly.
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
FROZEN_N_TREES = 273

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


def label_24_month(loan_status: pd.Series, issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.DataFrame:
    """Shared by section 2 (v2 evaluation) and section 3's "original"
    label (v2 label robustness): 1 = charged off within 24 months of
    issue, 0 = not charged off, observed past 24 months, excluded =
    charged off more than 24 months after issue or with no recorded
    last_pymnt_d at all."""
    months = (
        (pd.to_datetime(last_pymnt_d).dt.year - pd.to_datetime(issue_d).dt.year) * 12
        + (pd.to_datetime(last_pymnt_d).dt.month - pd.to_datetime(issue_d).dt.month)
    )
    was_bad = loan_status.isin(["Charged Off", "Default"])
    fast_default = was_bad & (months <= 24)
    cannot_label = was_bad & ((months > 24) | months.isna())
    label = fast_default.astype(float)
    label[cannot_label] = float("nan")
    return pd.DataFrame({"label_24m": label, "excluded": cannot_label})


def v1_scoring_cfg(frozen: dict) -> dict:
    """v1's scoring config, read from lightgbm_v1_history, with
    model_path/calibrator.path corrected: frozen_model_config.json's
    lightgbm_v1_history still points both at the live lightgbm/ path it
    shared with v1 before v2 was promoted over it, so those fields are
    overridden here to the actual backed-up v1 files under
    lightgbm_v1_history/ - the only place v1's own booster and
    calibrator still live. Shared by sections 2 and 3, both of which
    score "v1" against v2 and would otherwise silently score v2 twice."""
    cfg = dict(frozen["lightgbm_v1_history"])
    cfg["model_path"] = "data/processed/models/lightgbm_v1_history/model_chosen.joblib"
    cfg["calibrator"] = dict(cfg["calibrator"])
    cfg["calibrator"]["path"] = "data/processed/models/lightgbm_v1_history/calibrator.joblib"
    return cfg


def expected_calibration_error(y: pd.Series, pred, n_bins: int = 10) -> float:
    """Shared by sections 2 and 3. A different ECE implementation from
    src.evaluation.posthoc_validation's run_calibration (that one reads
    an already-built calibration table; this one bins the raw
    predictions itself), kept separate since they serve different
    inputs."""
    df = pd.DataFrame({"pred": pred, "y": y})
    df["bin"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    table = df.groupby("bin").agg(n=("y", "size"), mean_pred=("pred", "mean"), mean_actual=("y", "mean"))
    weighted_gap = (table["n"] * (table["mean_pred"] - table["mean_actual"]).abs()).sum()
    return float(weighted_gap / table["n"].sum())


# ---------------------------------------------------------------------
# 1. Ablations (was posthoc_ablations.py)
# ---------------------------------------------------------------------

ABLATIONS_MAX_BOOST_ROUNDS = 3000
ABLATIONS_EARLY_STOPPING_ROUNDS = 50


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
        num_boost_round=ABLATIONS_MAX_BOOST_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(ABLATIONS_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def auc_on(booster_or_pred_fn, df, feats):
    pred = booster_or_pred_fn(df[feats])
    return float(roc_auc_score(df["target"].values, pred))


def candidate_features_17() -> list[str]:
    with open(METRICS_DIR / "feature_selection_log.json") as f:
        return json.load(f)["final_features"]


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


def run_ablations() -> None:
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
    # "Champion" throughout this section means v1 (the model every
    # ablation here reuses hyperparameters from, and the model ablation
    # d's monotone constraints led to v2 from) - same v1_history path
    # correction as src.evaluation.posthoc_validation's replay-performance
    # and walk-forward sections, needed because frozen_model_config.json's
    # lightgbm_chosen now describes v2 after v2's promotion, and
    # lightgbm_v1_history's own model_path/calibrator.path fields were
    # never updated off the live lightgbm/ path they shared with v1
    # before that promotion.
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    v1_history = frozen["lightgbm_v1_history"]
    champion_booster = joblib.load("data/processed/models/lightgbm_v1_history/model_chosen.joblib")
    champion_feats = v1_history["features"]
    champion_check = income_monotonicity_check(champion_booster, champion_feats, test, "champion")
    ablation_d_check = income_monotonicity_check(booster_d, feats_d, test, "ablation_d_monotone")
    print(json.dumps(champion_check, indent=2))
    print(json.dumps(ablation_d_check, indent=2))

    champion_val_auc = v1_history["train_validation_metrics"]["validation_auc"]
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


# ---------------------------------------------------------------------
# 2. v2 evaluation (was posthoc_v2_evaluation.py)
# ---------------------------------------------------------------------

EVAL_WINDOW_START = "2016-01-01"
EVAL_WINDOW_END = "2016-10-31"

# Adoption rule, reports/metrics/v2_decision_rule.md / docs/v2_decision_rule.md.
AUC_SLACK = 0.002
ECE_SLACK = 0.005


def build_eval_population() -> pd.DataFrame:
    con = duckdb.connect()
    window = con.execute(
        f"""
        SELECT id, issue_d, loan_status, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND issue_d >= '{EVAL_WINDOW_START}' AND issue_d <= '{EVAL_WINDOW_END}'
        """
    ).fetchdf()
    labelled = label_24_month(window["loan_status"], window["issue_d"], window["last_pymnt_d"])
    window = pd.concat([window, labelled], axis=1)
    window = window[~window["excluded"]].copy()

    feats = pd.read_parquet(REPLAY_PARQUET)
    merged = window.merge(feats, on="id", how="inner", suffixes=("", "_feat"))
    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")
    return merged


def fit_v2(champion_feats: list[str]) -> lgb.Booster:
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    fold_train = pd.concat([train, validation], ignore_index=True)
    for col in CATEGORICAL_COLUMNS:
        fold_train[col] = fold_train[col].astype("category")

    monotone_constraints = [MONOTONE_DIRECTIONS.get(f, 0) for f in champion_feats]
    dataset = lgb.Dataset(
        fold_train[champion_feats], label=fold_train["target"].values,
        categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False,
    )
    params = dict(FROZEN_LGB_PARAMS)
    params["monotone_constraints"] = monotone_constraints
    return lgb.train(params, dataset, num_boost_round=FROZEN_N_TREES)


def run_v2_evaluation() -> None:
    population = build_eval_population()
    log_access(
        "posthoc_v2_evaluation.py",
        len(population),
        "Post-hoc, exploratory: scores v1 (frozen champion, unchanged) and a freshly-fit, "
        "in-memory-only v2 on 2016 vintages (issue <= 2016-10-31, 24-month label) to "
        "evaluate the pre-registered adoption rule in docs/v2_decision_rule.md, committed "
        "before this script was run. No change to the frozen v1 champion.",
    )

    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    frozen = load_frozen_config()
    population["pd_v1"] = score_lightgbm(population, v1_scoring_cfg(frozen))

    v2_booster = fit_v2(champion_feats)
    population["pd_v2"] = v2_booster.predict(population[champion_feats], num_iteration=v2_booster.best_iteration)

    y = population["label_24m"]
    auc_v1 = round(float(roc_auc_score(y, population["pd_v1"])), 4)
    auc_v2 = round(float(roc_auc_score(y, population["pd_v2"])), 4)
    ece_v1 = round(expected_calibration_error(y, population["pd_v1"]), 4)
    ece_v2 = round(expected_calibration_error(y, population["pd_v2"]), 4)

    auc_condition = auc_v2 >= auc_v1 - AUC_SLACK
    ece_condition = ece_v2 <= ece_v1 + ECE_SLACK
    adopt = bool(auc_condition and ece_condition)

    result = {
        "label": "post-hoc, exploratory",
        "rule_file": "docs/v2_decision_rule.md",
        "evaluation_population": {
            "window": [EVAL_WINDOW_START, EVAL_WINDOW_END],
            "n_scored": len(population),
            "label_24m_rate_pct": round(100 * float(y.mean()), 2),
        },
        "v1": {"auc": auc_v1, "ece": ece_v1},
        "v2": {"auc": auc_v2, "ece": ece_v2},
        "rule": {
            "auc_condition": f"AUC(v2) >= AUC(v1) - {AUC_SLACK}",
            "auc_condition_value": f"{auc_v2} >= {round(auc_v1 - AUC_SLACK, 4)}",
            "auc_condition_met": auc_condition,
            "ece_condition": f"ECE(v2) <= ECE(v1) + {ECE_SLACK}",
            "ece_condition_value": f"{ece_v2} <= {round(ece_v1 + ECE_SLACK, 4)}",
            "ece_condition_met": ece_condition,
        },
        "adopt_v2": adopt,
        "note": (
            "Adoption here means recording this decision and the numbers it was made on. "
            "It does not change the frozen v1 champion's artifacts, MLflow alias, or "
            "served model - promoting an actual v2 champion is a separate, later action."
        ),
    }
    with open(METRICS_DIR / "posthoc_v2_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------
# 3. v2 label robustness (was posthoc_v2_label_robustness.py)
# ---------------------------------------------------------------------


def label_24_month_corrected(loan_status: pd.Series, issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.Series:
    """Corrected label (new, chosen only after section 2 already ran
    and reported its result; reports/metrics/posthoc_v2_evaluation.json
    is left exactly as it was written):
        1 = charged off within 24 months of issue, OR charged off with
            no recorded last_pymnt_d at all (no payment ever made reads
            as an immediate, fast failure, not an unknown-timing one)
        0 = charged off more than 24 months after issue (it survived
            the 24-month window before eventually failing), OR not
            charged off
    No exclusions: every loan in the population gets a label."""
    months = (
        (pd.to_datetime(last_pymnt_d).dt.year - pd.to_datetime(issue_d).dt.year) * 12
        + (pd.to_datetime(last_pymnt_d).dt.month - pd.to_datetime(issue_d).dt.month)
    )
    was_bad = loan_status.isin(["Charged Off", "Default"])
    fast_default = was_bad & (months <= 24)
    no_payment_ever = was_bad & months.isna()
    return (fast_default | no_payment_ever).astype(float)


def build_population_both_labels() -> pd.DataFrame:
    con = duckdb.connect()
    window = con.execute(
        f"""
        SELECT id, issue_d, loan_status, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND issue_d >= '{EVAL_WINDOW_START}' AND issue_d <= '{EVAL_WINDOW_END}'
        """
    ).fetchdf()

    original = label_24_month(window["loan_status"], window["issue_d"], window["last_pymnt_d"])
    window["label_24m_original"] = original["label_24m"]
    window["excluded_original"] = original["excluded"]
    window["label_24m_corrected"] = label_24_month_corrected(window["loan_status"], window["issue_d"], window["last_pymnt_d"])

    feats = pd.read_parquet(REPLAY_PARQUET)
    merged = window.merge(feats, on="id", how="inner", suffixes=("", "_feat"))
    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")
    return merged


def run_v2_label_robustness() -> None:
    population = build_population_both_labels()
    log_access(
        "posthoc_v2_label_robustness.py",
        len(population),
        "Post-hoc, exploratory robustness check (does not change the already-recorded "
        "pre-registered v2 decision): rescores v1 and a freshly-refit v2 on the same "
        "2016-vintage population with a second, corrected 24-month label, chosen after "
        "the pre-registered evaluation already ran. No change to the frozen v1 champion.",
    )

    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    frozen = load_frozen_config()
    population["pd_v1"] = score_lightgbm(population, v1_scoring_cfg(frozen))

    v2_booster = fit_v2(champion_feats)
    population["pd_v2"] = v2_booster.predict(population[champion_feats], num_iteration=v2_booster.best_iteration)

    results = {}
    for label_name, label_col, excluded_col in [
        ("original", "label_24m_original", "excluded_original"),
        ("corrected", "label_24m_corrected", None),
    ]:
        if excluded_col is not None:
            scored = population[~population[excluded_col]]
        else:
            scored = population
        y = scored[label_col]
        auc_v1 = round(float(roc_auc_score(y, scored["pd_v1"])), 4)
        auc_v2 = round(float(roc_auc_score(y, scored["pd_v2"])), 4)
        ece_v1 = round(expected_calibration_error(y, scored["pd_v1"]), 4)
        ece_v2 = round(expected_calibration_error(y, scored["pd_v2"]), 4)
        auc_condition = auc_v2 >= auc_v1 - AUC_SLACK
        ece_condition = ece_v2 <= ece_v1 + ECE_SLACK
        results[label_name] = {
            "n_scored": len(scored),
            "label_rate_pct": round(100 * float(y.mean()), 2),
            "v1": {"auc": auc_v1, "ece": ece_v1},
            "v2": {"auc": auc_v2, "ece": ece_v2},
            "auc_condition_met": auc_condition,
            "ece_condition_met": ece_condition,
            "adopt_v2_under_this_label": bool(auc_condition and ece_condition),
        }

    summary = {
        "label": "post-hoc, exploratory robustness check - does not change the pre-registered decision",
        "pre_registered_result_untouched": "reports/metrics/posthoc_v2_evaluation.json",
        "note": (
            "The corrected label was chosen after the pre-registered run already reported "
            "its result. It reclassifies two cases the original label excluded: a "
            "charge-off more than 24 months after issue now counts as 0 (survived the "
            "24-month window), and a charge-off with no recorded last_pymnt_d now counts "
            "as 1 (no payment ever made reads as an immediate failure)."
        ),
        "results_by_label": results,
        "decision_changes": results["original"]["adopt_v2_under_this_label"] != results["corrected"]["adopt_v2_under_this_label"],
    }
    with open(METRICS_DIR / "posthoc_v2_label_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 4. v2 on test (2015): the served model, scored directly (new)
# ---------------------------------------------------------------------


def run_v2_on_test_2015() -> None:
    """Scores the actual served v2 (its saved booster and calibrator,
    read from frozen_model_config.json's lightgbm_chosen, not a fresh
    refit) on test (2015). This is a genuinely new, separate post-hoc
    read of test: v2 was selected on 2016 vintages (section 2) and its
    calibrator was fit on 2015 afterward (not an evaluation, a
    calibration step, see docs/validation_report.md), but v2's AUC,
    Gini, and KS on the full 2015 test population were never computed
    before this check. 2015 is therefore not a clean holdout for v2 in
    the way it was for v1: v2's calibrator already saw these labels.
    Calibration (a monotone, score-preserving transform) does not
    change AUC, Gini, or KS, so this caveat affects how the numbers
    below should be read, not their arithmetic."""
    test = pd.read_parquet(TEST_PARQUET)
    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    assert champion_feats == cfg["features"]

    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")
    test["pd_v2"] = score_lightgbm(test, cfg)

    log_access(
        "posthoc_v2.py:run_v2_on_test_2015",
        len(test),
        "Post-hoc: scores the served v2 model (saved booster and calibrator, no refit) "
        "on test (2015) to report its own AUC/Gini/KS with bootstrap CIs. v2's Platt "
        "calibrator was already fit on 2015 (a calibration step, not model selection), "
        "so 2015 is not a clean holdout for v2 the way it was for v1. No model, feature, "
        "or policy change.",
    )

    test_scored = pd.read_parquet(TEST_SCORED)
    merged = test[["id"]].merge(
        test_scored[["id", "target", "sub_grade", "pd_lightgbm"]], on="id", how="left"
    )
    assert merged["target"].isna().sum() == 0
    merged["pd_v2"] = test["pd_v2"].values
    merged["sub_grade_ordinal"] = sub_grade_ordinal(merged["sub_grade"])
    y = merged["target"].values

    auc_v2 = roc_auc_score(y, merged["pd_v2"].values)
    gini_v2 = 2 * auc_v2 - 1
    ks_v2 = ks_statistic(y, merged["pd_v2"].values)
    auc_ci = single_model_bootstrap_ci(y, merged["pd_v2"].values, roc_auc_score)

    diff_vs_v1 = paired_bootstrap(y, merged["pd_v2"].values, merged["pd_lightgbm"].values, roc_auc_score)
    diff_vs_sub_grade = paired_bootstrap(y, merged["pd_v2"].values, merged["sub_grade_ordinal"].values, roc_auc_score)

    result = {
        "label": "post-hoc, v2's own calibrator was fit on this same 2015 data (calibration, "
        "not model selection), so 2015 is not a clean holdout for v2 the way it was for v1",
        "n": len(merged),
        "v2_on_test_2015": {
            "auc": round(float(auc_v2), 4),
            "auc_ci95": auc_ci,
            "gini": round(float(gini_v2), 4),
            "ks": round(float(ks_v2), 4),
        },
        "paired_auc_diff_v2_minus_v1": diff_vs_v1,
        "paired_auc_diff_v2_minus_sub_grade": diff_vs_sub_grade,
        "note": (
            "Calibration (v2's Platt scaling, fit on 2015) is a monotone transform of the "
            "raw booster score, so it does not change AUC, Gini, or KS - these metrics are "
            "identical whether scored with or without the calibrator applied. v1's score "
            "(pd_lightgbm) and target come from the already-saved, one-time "
            "test_scored.parquet; v2 is scored fresh here with its own saved artifacts "
            "(reports/metrics/frozen_model_config.json's lightgbm_chosen)."
        ),
    }
    with open(METRICS_DIR / "posthoc_v2_test_2015.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main() -> None:
    print("=== 1. Ablations ===")
    run_ablations()
    print("\n=== 2. v2 evaluation ===")
    run_v2_evaluation()
    print("\n=== 3. v2 label robustness ===")
    run_v2_label_robustness()
    print("\n=== 4. v2 on test (2015) ===")
    run_v2_on_test_2015()


if __name__ == "__main__":
    main()
