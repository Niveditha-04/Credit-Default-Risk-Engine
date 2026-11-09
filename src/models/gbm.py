"""Gradient boosting: LightGBM (main) and XGBoost (comparison).

Uses the full allow-listed feature set (72 features: 61 raw + 11
engineered), not the 17-feature scorecard subset - trees handle
redundancy and missing values natively, so there is no need for the
scorecard's WOE-space correlation/VIF pruning here.

Fit on train (2012-2013), early-stopped and tuned on validation (2014).
Calibrated on validation too. Test (2015) is never touched.

IMPORTANT: validation is used for both early stopping/tuning AND
calibration in this script. That means the validation AUC reported here
is slightly optimistic as an estimate of out-of-sample performance - it
was itself a target of model selection. The honest, once-only read on
truly unseen data happens in Phase 5 against test (2015).
"""

import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")

SEED = 42
BANNED_COLUMNS = {
    "id", "issue_d", "split", "target",
    "addr_state", "zip_code", "grade", "sub_grade", "int_rate", "installment", "funded_amnt",
}
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

N_TUNING_TRIALS_LGB = 20
N_TUNING_TRIALS_XGB = 10
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

TRAIN_VAL_AUC_GAP_FLAG = 0.05
VALIDATION_AUC_INVESTIGATE_THRESHOLD = 0.75
SCORECARD_VALIDATION_AUC = 0.665  # from Phase 3, reports/metrics/scorecard_validation_metrics.json


def feature_list(train_columns: list[str]) -> list[str]:
    feats = [c for c in train_columns if c not in {"id", "issue_d", "split", "target"}]
    leaked = BANNED_COLUMNS & set(feats)
    assert not leaked, f"Banned columns present in GBM feature list: {leaked}"
    assert "id" not in feats and "issue_d" not in feats and "split" not in feats and "target" not in feats
    return sorted(feats)


def load_data():
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    feats = feature_list(list(train.columns))
    for col in CATEGORICAL_COLUMNS:
        train[col] = train[col].astype("category")
        validation[col] = (
            validation[col]
            .astype("category")
            .cat.set_categories(train[col].cat.categories)
        )
    return train, validation, feats


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def sample_lgb_params(rng: random.Random) -> dict:
    return {
        "num_leaves": rng.choice([15, 31, 63, 127]),
        "learning_rate": rng.choice([0.01, 0.02, 0.05, 0.08]),
        "min_child_samples": rng.choice([20, 50, 100, 200]),
        "feature_fraction": rng.choice([0.6, 0.8, 1.0]),
        "bagging_fraction": rng.choice([0.6, 0.8, 1.0]),
        "bagging_freq": rng.choice([0, 1, 5]),
        "lambda_l1": rng.choice([0.0, 0.1, 1.0]),
        "lambda_l2": rng.choice([0.0, 0.1, 1.0]),
    }


def sample_xgb_params(rng: random.Random) -> dict:
    return {
        "max_depth": rng.choice([3, 4, 6, 8]),
        "eta": rng.choice([0.01, 0.02, 0.05, 0.08]),
        "min_child_weight": rng.choice([1, 5, 20, 50]),
        "subsample": rng.choice([0.6, 0.8, 1.0]),
        "colsample_bytree": rng.choice([0.6, 0.8, 1.0]),
        "reg_alpha": rng.choice([0.0, 0.1, 1.0]),
        "reg_lambda": rng.choice([0.0, 0.1, 1.0]),
    }


def tune_lightgbm(train, validation, feats, y_train, y_validation):
    rng = random.Random(SEED)
    # feature_pre_filter=False: each trial tries a different min_child_samples
    # (min_data_in_leaf), but this Dataset is built once and reused across
    # all trials. Without this, LightGBM pre-filters features based on
    # whichever min_data_in_leaf the Dataset first saw and errors on later
    # trials that use a smaller value.
    lgb_train = lgb.Dataset(
        train[feats], label=y_train, categorical_feature=CATEGORICAL_COLUMNS,
        free_raw_data=False, params={"feature_pre_filter": False},
    )
    lgb_val = lgb.Dataset(validation[feats], label=y_validation, reference=lgb_train, free_raw_data=False)

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "seed": SEED,
        "verbosity": -1,
    }

    trials = []
    best = {"auc": -1.0, "params": None, "booster": None, "best_iteration": None}
    for trial_id in range(N_TUNING_TRIALS_LGB):
        params = {**base_params, **sample_lgb_params(rng)}
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred = booster.predict(validation[feats], num_iteration=booster.best_iteration)
        auc = roc_auc_score(y_validation, val_pred)
        trial_log = {"trial": trial_id, "model": "lightgbm", "auc": round(auc, 5),
                     "best_iteration": booster.best_iteration, **params}
        trials.append(trial_log)
        print(f"  [lgb trial {trial_id}] auc={auc:.5f} best_iter={booster.best_iteration}")
        if auc > best["auc"]:
            best = {"auc": auc, "params": params, "booster": booster, "best_iteration": booster.best_iteration}

    return best, trials


def tune_xgboost(train, validation, feats, y_train, y_validation):
    rng = random.Random(SEED + 1)
    dtrain = xgb.DMatrix(train[feats], label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(validation[feats], label=y_validation, enable_categorical=True)

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 4,
    }

    trials = []
    best = {"auc": -1.0, "params": None, "booster": None, "best_iteration": None}
    for trial_id in range(N_TUNING_TRIALS_XGB):
        params = {**base_params, **sample_xgb_params(rng)}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dval, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        val_pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        auc = roc_auc_score(y_validation, val_pred)
        trial_log = {"trial": trial_id, "model": "xgboost", "auc": round(auc, 5),
                     "best_iteration": booster.best_iteration, **params}
        trials.append(trial_log)
        print(f"  [xgb trial {trial_id}] auc={auc:.5f} best_iter={booster.best_iteration}")
        if auc > best["auc"]:
            best = {"auc": auc, "params": params, "booster": booster, "best_iteration": booster.best_iteration}

    return best, trials


def calibrate_and_compare(y_validation, val_pred_raw):
    """Fit isotonic and Platt (sigmoid) calibration on validation, compare
    by Brier score, return whichever is better plus before/after PD stats."""
    isotonic = IsotonicRegression(out_of_bounds="clip")
    iso_pred = isotonic.fit_transform(val_pred_raw, y_validation)
    brier_iso = brier_score_loss(y_validation, iso_pred)

    platt = LogisticRegression()
    platt.fit(val_pred_raw.reshape(-1, 1), y_validation)
    platt_pred = platt.predict_proba(val_pred_raw.reshape(-1, 1))[:, 1]
    brier_platt = brier_score_loss(y_validation, platt_pred)

    brier_uncalibrated = brier_score_loss(y_validation, val_pred_raw)

    result = {
        "brier_uncalibrated": round(float(brier_uncalibrated), 5),
        "brier_isotonic": round(float(brier_iso), 5),
        "brier_platt": round(float(brier_platt), 5),
        "chosen_method": "isotonic" if brier_iso < brier_platt else "platt",
        "mean_predicted_pd_before_pct": round(float(val_pred_raw.mean()) * 100, 2),
        "actual_default_rate_pct": round(float(y_validation.mean()) * 100, 2),
        "mean_predicted_pd_after_isotonic_pct": round(float(iso_pred.mean()) * 100, 2),
        "mean_predicted_pd_after_platt_pct": round(float(platt_pred.mean()) * 100, 2),
    }
    return result, isotonic, platt


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train, validation, feats = load_data()
    print(f"Feature list: {len(feats)} features")
    print(f"Banned columns check passed: none of {sorted(BANNED_COLUMNS)} present")
    with open(METRICS_DIR / "gbm_feature_list.json", "w") as f:
        json.dump(feats, f, indent=2)

    y_train = train["target"].values
    y_validation = validation["target"].values

    print(f"\n=== Tuning LightGBM ({N_TUNING_TRIALS_LGB} trials, seed={SEED}) ===")
    best_lgb, lgb_trials = tune_lightgbm(train, validation, feats, y_train, y_validation)

    print(f"\n=== Tuning XGBoost ({N_TUNING_TRIALS_XGB} trials, seed={SEED}) ===")
    best_xgb, xgb_trials = tune_xgboost(train, validation, feats, y_train, y_validation)

    all_trials = lgb_trials + xgb_trials
    with open(METRICS_DIR / "gbm_tuning_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2, default=str)
    pd.DataFrame(all_trials).to_csv(METRICS_DIR / "gbm_tuning_trials.csv", index=False)

    print(f"\nBest LightGBM: AUC={best_lgb['auc']:.5f}, best_iteration={best_lgb['best_iteration']}")
    print(f"Best XGBoost:  AUC={best_xgb['auc']:.5f}, best_iteration={best_xgb['best_iteration']}")
    print(f"Scorecard validation AUC (Phase 3): {SCORECARD_VALIDATION_AUC}")

    booster = best_lgb["booster"]
    train_pred_raw = booster.predict(train[feats], num_iteration=booster.best_iteration)
    val_pred_raw = booster.predict(validation[feats], num_iteration=booster.best_iteration)

    train_auc = roc_auc_score(y_train, train_pred_raw)
    val_auc = roc_auc_score(y_validation, val_pred_raw)
    gap = train_auc - val_auc

    eval_metrics = {
        "model": "lightgbm",
        "train_auc": round(train_auc, 4),
        "train_gini": round(2 * train_auc - 1, 4),
        "train_ks": round(ks_statistic(y_train, train_pred_raw), 4),
        "validation_auc": round(val_auc, 4),
        "validation_gini": round(2 * val_auc - 1, 4),
        "validation_ks": round(ks_statistic(y_validation, val_pred_raw), 4),
        "train_minus_validation_auc_gap": round(gap, 4),
        "gap_flagged": bool(gap > TRAIN_VAL_AUC_GAP_FLAG),
        "validation_auc_above_investigate_threshold": bool(val_auc > VALIDATION_AUC_INVESTIGATE_THRESHOLD),
        "scorecard_validation_auc": SCORECARD_VALIDATION_AUC,
        "lift_over_scorecard_auc": round(val_auc - SCORECARD_VALIDATION_AUC, 4),
        "note": (
            "Validation was used for early stopping AND hyperparameter "
            "selection AND calibration in this phase, so this validation "
            "AUC is optimistic as an out-of-sample estimate. Test (2015) "
            "is untouched and will give the honest read in Phase 5."
        ),
    }
    with open(METRICS_DIR / "gbm_validation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(json.dumps(eval_metrics, indent=2))

    if gap > TRAIN_VAL_AUC_GAP_FLAG:
        print(f"\n!!! FLAG: train-validation AUC gap {gap:.4f} exceeds {TRAIN_VAL_AUC_GAP_FLAG} !!!")
    if val_auc > VALIDATION_AUC_INVESTIGATE_THRESHOLD:
        print(f"\n!!! FLAG: validation AUC {val_auc:.4f} exceeds {VALIDATION_AUC_INVESTIGATE_THRESHOLD}, investigate for leakage !!!")

    print("\n=== Feature importance (gain), top 15 ===")
    importance = pd.DataFrame(
        {"feature": booster.feature_name(), "gain": booster.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importance["gain_pct_of_total"] = round(100 * importance["gain"] / importance["gain"].sum(), 2)
    importance.to_csv(METRICS_DIR / "gbm_feature_importance.csv", index=False)
    print(importance.head(15).to_string(index=False))

    rollout_related = [
        "mths_since_recent_bc_dlq", "mths_since_last_major_derog", "mths_since_recent_inq",
        "mths_since_recent_revol_delinq", "never_flag__mths_since_recent_bc_dlq",
        "never_flag__mths_since_last_major_derog", "never_flag__mths_since_recent_inq",
        "never_flag__mths_since_recent_revol_delinq",
    ]
    top15_names = set(importance.head(15)["feature"])
    rollout_in_top15 = top15_names & set(rollout_related)
    print(f"\nRollout-affected columns in top 15 by gain: {rollout_in_top15 or 'none'}")

    print("\n=== Calibration: isotonic vs Platt (validation) ===")
    calib_result, isotonic_model, platt_model = calibrate_and_compare(y_validation, val_pred_raw)
    with open(METRICS_DIR / "gbm_calibration.json", "w") as f:
        json.dump(calib_result, f, indent=2)
    print(json.dumps(calib_result, indent=2))

    import joblib

    joblib.dump(booster, MODELS_DIR / "lightgbm_model.joblib")
    joblib.dump(best_xgb["booster"], MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(isotonic_model, MODELS_DIR / "lightgbm_isotonic_calibrator.joblib")
    joblib.dump(platt_model, MODELS_DIR / "lightgbm_platt_calibrator.joblib")
    with open(MODELS_DIR / "gbm_feature_list.json", "w") as f:
        json.dump(feats, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
