"""Gradient boosting: LightGBM (main) and XGBoost (comparison).

Uses the full allow-listed feature set (69 features: 59 raw + 10
engineered - fico_range_low/high and log_annual_inc were dropped as exact
duplicates of fico_mean/annual_inc before this freeze), not the
17-feature scorecard subset - trees handle redundancy and missing values
natively, so there is no need for the scorecard's WOE-space
correlation/VIF pruning here.

Fit on train (2012-2013), early-stopped and tuned on validation (2014).
Two LightGBM searches run: an unconstrained one (20 trials) and a small
regularized one (10 trials, num_leaves <= 15, min_child_samples >= 500,
lambda_l2 > 0). The final LightGBM model is the SIMPLEST config (fewest
trees) within 0.002 validation AUC of the best trial found across both
searches - a complexity-vs-performance tradeoff, not just "best AUC".
XGBoost gets the same 20-trial budget as LightGBM's unconstrained search.

Calibrated on an OUT-OF-SAMPLE split of validation itself: fit on
Jan-Sep 2014, evaluate on Oct-Dec 2014 (never seen by the calibrator),
then the winning method is refit on the full 2014 for deployment. Test
(2015) is never touched anywhere in this script.

IMPORTANT: validation is used for early stopping, hyperparameter search,
model selection (the 0.002 rule), AND (a held-out slice of) calibration.
That means the validation AUC reported here is optimistic as an estimate
of out-of-sample performance - it was itself a target of model selection
at every step. The honest, once-only read on truly unseen data happens in
Phase 5 against test (2015).
"""

import json
import os
import random
import stat
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.models.math_utils import apply_calibrator, logit

TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")
CANDIDATES_DIR = MODELS_DIR / "candidates"

# Isotonic is deployed only when it beats Platt by more than this much
# holdout Brier score; a smaller edge stays on Platt, since Platt's
# two-parameter curve generalizes more reliably than isotonic's
# step function on a near-tie.
ISOTONIC_SELECTION_MARGIN = 0.0005

# 2026-11-10 incident: a run meant only to verify that adding ORDER BY to
# the data pipeline didn't change results (it did - see
# stability.py, row order interacts with LightGBM's
# seeded bagging/feature subsampling) silently overwrote the frozen,
# already-verified model_chosen.joblib and xgboost/model.joblib with a
# DIFFERENT model, with no backup. Every run now writes to its own
# data/processed/models/candidates/<run_id>/ directory instead of the
# frozen data/processed/models/<model>/ paths. Only an explicit
# --promote flag copies a candidate's artifacts over the frozen ones
# (and only after confirming the files there aren't locked - see
# promote_candidate_artifacts()). Nothing is lost by default: every run's
# output survives under its own candidate directory even if never
# promoted.


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def promote_requested() -> bool:
    return "--promote" in sys.argv


def lock_frozen_file(path: Path) -> None:
    """Read-only for everyone - a plain joblib.dump()/open(...,'w') to
    this exact path will now raise PermissionError instead of silently
    overwriting a verified artifact."""
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def promote_candidate_artifacts(candidate_dir: Path, frozen_dir: Path) -> None:
    """Copies every file from a candidate run directory over the frozen
    directory, unlocking (temporarily) and re-locking read-only
    afterward. Only called when --promote was explicitly passed."""
    import shutil

    frozen_dir.mkdir(parents=True, exist_ok=True)
    for src in candidate_dir.iterdir():
        dst = frozen_dir / src.name
        if dst.exists():
            os.chmod(dst, stat.S_IWUSR | stat.S_IRUSR)
        shutil.copy2(src, dst)
        lock_frozen_file(dst)
    print(f"PROMOTED: {candidate_dir} -> {frozen_dir} (frozen files are now read-only)")

SEED = 42
METADATA_COLUMNS = {"id", "issue_d", "split", "target"}
FROZEN_CONFIG_PATH = METRICS_DIR / "frozen_model_config.json"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

N_TUNING_TRIALS_LGB = 20
N_TUNING_TRIALS_LGB_REGULARIZED = 10
N_TUNING_TRIALS_XGB = 20
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

TRAIN_VAL_AUC_GAP_FLAG = 0.05
VALIDATION_AUC_INVESTIGATE_THRESHOLD = 0.75
SCORECARD_VALIDATION_AUC = 0.665  # from Phase 3, reports/metrics/scorecard_validation_metrics.json
SIMPLICITY_AUC_TOLERANCE = 0.002  # "simplest config within 0.002 AUC of the best"


def feature_list(train_columns: list[str]) -> list[str]:
    """The model's feature list must equal the already-frozen allow-list
    exactly - same 69 names, same (alphabetical) order - not just "none of
    a handful of specifically-named banned columns are present." A
    deny-list only catches columns someone thought to list in advance; a
    stray duplicate column from an unrelated query change would not be on
    anyone's deny-list, since it is not a leakage column at all. An exact
    allow-list catches ANY unexpected column, named or not, because
    anything not on the frozen list is rejected on sight.

    This assumes a frozen champion config already exists (true from Phase
    5 onward) - it is a regression guard for an established pipeline, not
    a bootstrapping mechanism for the very first run."""
    derived = sorted(c for c in train_columns if c not in METADATA_COLUMNS)

    with open(FROZEN_CONFIG_PATH) as f:
        frozen_features = json.load(f)["lightgbm_chosen"]["features"]

    if derived != frozen_features:
        extra = [c for c in derived if c not in frozen_features]
        missing = [c for c in frozen_features if c not in derived]
        raise AssertionError(
            "Feature list does not exactly match the frozen allow-list "
            f"(reports/metrics/frozen_model_config.json). Extra columns present now "
            f"that are not in the frozen list: {extra}. Columns in the frozen list "
            f"missing now: {missing}."
        )
    return derived


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


def sample_lgb_regularized_params(rng: random.Random) -> dict:
    """Constrained search: shallow trees, large leaves, non-zero L2."""
    return {
        "num_leaves": rng.choice([7, 10, 15]),
        "learning_rate": rng.choice([0.02, 0.05, 0.08]),
        "min_child_samples": rng.choice([500, 750, 1000, 1500]),
        "feature_fraction": rng.choice([0.6, 0.8, 1.0]),
        "bagging_fraction": rng.choice([0.6, 0.8, 1.0]),
        "bagging_freq": rng.choice([0, 1, 5]),
        "lambda_l1": rng.choice([0.0, 0.1, 1.0]),
        "lambda_l2": rng.choice([0.1, 0.5, 1.0, 2.0]),
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


def tune_lightgbm(train, validation, feats, y_train, y_validation, search_name, param_sampler, n_trials, seed,
                   log_to_mlflow=False):
    rng = random.Random(seed)
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
    boosters = {}
    for trial_id in range(n_trials):
        params = {**base_params, **param_sampler(rng)}
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred = booster.predict(validation[feats], num_iteration=booster.best_iteration)
        auc = roc_auc_score(y_validation, val_pred)
        global_trial_id = f"{search_name}_{trial_id}"
        trial_log = {"trial": global_trial_id, "model": "lightgbm", "search": search_name, "auc": round(auc, 5),
                     "best_iteration": booster.best_iteration, **params}
        trials.append(trial_log)
        boosters[global_trial_id] = booster
        print(f"  [lgb {search_name} trial {trial_id}] auc={auc:.5f} best_iter={booster.best_iteration}")

        if log_to_mlflow:
            import mlflow
            with mlflow.start_run(run_name=f"lgb_{global_trial_id}", nested=True):
                mlflow.log_params({k: v for k, v in params.items() if k not in ("deterministic", "force_row_wise")})
                mlflow.log_param("search", search_name)
                mlflow.log_metric("validation_auc", auc)
                mlflow.log_metric("best_iteration", booster.best_iteration)

    return trials, boosters


def tune_xgboost(train, validation, feats, y_train, y_validation, log_to_mlflow=False):
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

        if log_to_mlflow:
            import mlflow
            with mlflow.start_run(run_name=f"xgb_{trial_id}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("validation_auc", auc)
                mlflow.log_metric("best_iteration", booster.best_iteration)

    return best, trials


def fit_isotonic(fit_pred: np.ndarray, fit_y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(fit_pred, fit_y)
    return model


def fit_platt(fit_pred: np.ndarray, fit_y: np.ndarray) -> LogisticRegression:
    # Platt scaling is logistic regression on the LOGIT of the raw predicted
    # probability, not the raw probability itself - using raw probability as
    # the single feature is the common implementation shortcut, but the
    # textbook definition calibrates on the model's score in logit space.
    model = LogisticRegression()
    model.fit(logit(fit_pred).reshape(-1, 1), fit_y)
    return model


def calibrate_out_of_sample(validation: pd.DataFrame, val_pred_raw: np.ndarray, y_validation: np.ndarray):
    """Time-split calibration, evaluated out-of-sample within 2014 itself:
    fit calibrators on Jan-Sep 2014, score the held-out Oct-Dec 2014 slice
    (never seen by the calibrator), compare by Brier score there. Then
    refit the winning method on the FULL 2014 for deployment, since the
    time-split was only to get an honest comparison, not to shrink the
    final calibrator's training data.

    Isotonic is deployed only if its holdout Brier score beats Platt's by
    more than ISOTONIC_SELECTION_MARGIN (0.0005); otherwise Platt is
    deployed. Every downstream scoring path reads whichever calibrator
    this picks through the shared apply_calibrator(), so either choice
    scores correctly with no caller-side branching."""
    issue_month = validation["issue_d"].values
    fit_mask = issue_month < np.datetime64("2014-10-01")
    holdout_mask = ~fit_mask

    fit_pred, fit_y = val_pred_raw[fit_mask], y_validation[fit_mask]
    holdout_pred, holdout_y = val_pred_raw[holdout_mask], y_validation[holdout_mask]

    isotonic = fit_isotonic(fit_pred, fit_y)
    platt = fit_platt(fit_pred, fit_y)

    iso_holdout_pred = apply_calibrator(isotonic, holdout_pred)
    platt_holdout_pred = apply_calibrator(platt, holdout_pred)

    brier_uncalibrated = brier_score_loss(holdout_y, holdout_pred)
    brier_iso = brier_score_loss(holdout_y, iso_holdout_pred)
    brier_platt = brier_score_loss(holdout_y, platt_holdout_pred)
    chosen_method = "isotonic" if brier_platt - brier_iso > ISOTONIC_SELECTION_MARGIN else "platt"

    result = {
        "fit_period": "2014-01 to 2014-09",
        "holdout_period": "2014-10 to 2014-12",
        "n_fit": int(fit_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "brier_uncalibrated_holdout": round(float(brier_uncalibrated), 5),
        "brier_isotonic_holdout": round(float(brier_iso), 5),
        "brier_platt_holdout": round(float(brier_platt), 5),
        "chosen_method": chosen_method,
        "mean_predicted_pd_before_holdout_pct": round(float(holdout_pred.mean()) * 100, 4),
        "actual_default_rate_holdout_pct": round(float(holdout_y.mean()) * 100, 4),
        "mean_predicted_pd_after_isotonic_holdout_pct": round(float(iso_holdout_pred.mean()) * 100, 4),
        "mean_predicted_pd_after_platt_holdout_pct": round(float(platt_holdout_pred.mean()) * 100, 4),
    }

    # Refit the winner on the FULL 2014 for the deployed calibrator.
    if chosen_method == "isotonic":
        final_calibrator = fit_isotonic(val_pred_raw, y_validation)
    else:
        final_calibrator = fit_platt(val_pred_raw, y_validation)
    final_pred_full = apply_calibrator(final_calibrator, val_pred_raw)

    result["mean_predicted_pd_before_full_2014_pct"] = round(float(val_pred_raw.mean()) * 100, 4)
    result["actual_default_rate_full_2014_pct"] = round(float(y_validation.mean()) * 100, 4)
    result["mean_predicted_pd_after_full_2014_pct"] = round(float(final_pred_full.mean()) * 100, 4)
    result["final_calibrator_refit_on"] = "full 2014 (Jan-Dec)"

    return result, chosen_method, final_calibrator


def main(log_to_mlflow: bool = False, run_id: str | None = None) -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()
    candidate_dir = CANDIDATES_DIR / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training output for this run goes to {candidate_dir} - "
          f"the frozen data/processed/models/ artifacts are not touched unless --promote is passed.")

    train, validation, feats = load_data()
    print(f"Feature list: {len(feats)} features")
    print("Exact allow-list check passed: feature list matches frozen_model_config.json exactly")
    with open(METRICS_DIR / "gbm_feature_list.json", "w") as f:
        json.dump(feats, f, indent=2)

    y_train = train["target"].values
    y_validation = validation["target"].values

    print(f"\n=== Tuning LightGBM, unconstrained ({N_TUNING_TRIALS_LGB} trials, seed={SEED}) ===")
    unconstrained_trials, unconstrained_boosters = tune_lightgbm(
        train, validation, feats, y_train, y_validation, "unconstrained", sample_lgb_params, N_TUNING_TRIALS_LGB, SEED,
        log_to_mlflow=log_to_mlflow,
    )

    print(f"\n=== Tuning LightGBM, regularized ({N_TUNING_TRIALS_LGB_REGULARIZED} trials, seed={SEED + 100}) ===")
    regularized_trials, regularized_boosters = tune_lightgbm(
        train, validation, feats, y_train, y_validation, "regularized", sample_lgb_regularized_params,
        N_TUNING_TRIALS_LGB_REGULARIZED, SEED + 100, log_to_mlflow=log_to_mlflow,
    )

    lgb_trials = unconstrained_trials + regularized_trials
    lgb_boosters = {**unconstrained_boosters, **regularized_boosters}

    print(f"\n=== Tuning XGBoost ({N_TUNING_TRIALS_XGB} trials, seed={SEED}) ===")
    best_xgb, xgb_trials = tune_xgboost(train, validation, feats, y_train, y_validation, log_to_mlflow=log_to_mlflow)

    all_trials = lgb_trials + xgb_trials
    with open(METRICS_DIR / "gbm_tuning_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2, default=str)
    pd.DataFrame(all_trials).to_csv(METRICS_DIR / "gbm_tuning_trials.csv", index=False)

    # Model selection: best AUC across both LightGBM searches, then the
    # SIMPLEST (fewest trees) trial within SIMPLICITY_AUC_TOLERANCE of it.
    best_trial = max(lgb_trials, key=lambda t: t["auc"])
    best_auc = best_trial["auc"]
    within_tolerance = [t for t in lgb_trials if t["auc"] >= best_auc - SIMPLICITY_AUC_TOLERANCE]
    chosen_trial = min(within_tolerance, key=lambda t: t["best_iteration"])

    print(f"\nBest LightGBM trial: {best_trial['trial']}, AUC={best_trial['auc']:.5f}, "
          f"n_trees={best_trial['best_iteration']}")
    print(f"Chosen LightGBM trial (simplest within {SIMPLICITY_AUC_TOLERANCE} AUC): {chosen_trial['trial']}, "
          f"AUC={chosen_trial['auc']:.5f}, n_trees={chosen_trial['best_iteration']}")
    print(f"Best XGBoost:  AUC={best_xgb['auc']:.5f}, best_iteration={best_xgb['best_iteration']}")
    print(f"Scorecard validation AUC (Phase 3): {SCORECARD_VALIDATION_AUC}")

    def model_metrics(booster, trial, label):
        train_pred = booster.predict(train[feats], num_iteration=booster.best_iteration)
        val_pred = booster.predict(validation[feats], num_iteration=booster.best_iteration)
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_validation, val_pred)
        gap = train_auc - val_auc
        metrics = {
            "label": label,
            "trial": trial["trial"],
            "n_trees": trial["best_iteration"],
            "train_auc": round(train_auc, 4),
            "train_gini": round(2 * train_auc - 1, 4),
            "train_ks": round(ks_statistic(y_train, train_pred), 4),
            "validation_auc": round(val_auc, 4),
            "validation_gini": round(2 * val_auc - 1, 4),
            "validation_ks": round(ks_statistic(y_validation, val_pred), 4),
            "train_minus_validation_auc_gap": round(gap, 4),
            "gap_flagged": bool(gap > TRAIN_VAL_AUC_GAP_FLAG),
            "validation_auc_above_investigate_threshold": bool(val_auc > VALIDATION_AUC_INVESTIGATE_THRESHOLD),
        }
        return metrics, train_pred, val_pred

    best_metrics, _, _ = model_metrics(lgb_boosters[best_trial["trial"]], best_trial, "best")
    chosen_metrics, _chosen_train_pred, chosen_val_pred = model_metrics(
        lgb_boosters[chosen_trial["trial"]], chosen_trial, "chosen"
    )

    eval_metrics = {
        "model": "lightgbm",
        "selection_rule": f"simplest (fewest trees) trial within {SIMPLICITY_AUC_TOLERANCE} validation AUC of the best",
        "best": best_metrics,
        "chosen": chosen_metrics,
        "scorecard_validation_auc": SCORECARD_VALIDATION_AUC,
        "chosen_lift_over_scorecard_auc": round(chosen_metrics["validation_auc"] - SCORECARD_VALIDATION_AUC, 4),
        "note": (
            "Validation was used for early stopping, hyperparameter search, "
            "the best-vs-chosen selection rule, AND calibration in this "
            "phase, so this validation AUC is optimistic as an "
            "out-of-sample estimate. Test (2015) is untouched and will "
            "give the honest read in Phase 5."
        ),
    }
    with open(METRICS_DIR / "gbm_validation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(json.dumps(eval_metrics, indent=2))

    for label, m in [("best", best_metrics), ("chosen", chosen_metrics)]:
        if m["gap_flagged"]:
            print(f"\n!!! FLAG: {label} model train-validation AUC gap {m['train_minus_validation_auc_gap']:.4f} "
                  f"exceeds {TRAIN_VAL_AUC_GAP_FLAG} !!!")
        if m["validation_auc_above_investigate_threshold"]:
            print(f"\n!!! FLAG: {label} model validation AUC {m['validation_auc']:.4f} exceeds "
                  f"{VALIDATION_AUC_INVESTIGATE_THRESHOLD}, investigate for leakage !!!")

    # The CHOSEN model is what gets frozen and carried into Phase 5.
    chosen_booster = lgb_boosters[chosen_trial["trial"]]

    print("\n=== Feature importance (gain), top 15, chosen model ===")
    importance = pd.DataFrame(
        {"feature": chosen_booster.feature_name(), "gain": chosen_booster.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importance["gain_pct_of_total"] = round(100 * importance["gain"] / importance["gain"].sum(), 2)
    importance.to_csv(METRICS_DIR / "gbm_feature_importance.csv", index=False)
    print(importance.head(15).to_string(index=False))

    rollout_related = [
        "mths_since_recent_bc_dlq", "mths_since_last_major_derog", "mths_since_recent_inq",
        "mths_since_recent_revol_delinq", "never_flag__mths_since_recent_bc_dlq",
        "never_flag__mths_since_last_major_derog", "never_flag__mths_since_recent_inq",
        "never_flag__mths_since_recent_revol_delinq", "acc_open_past_24mths", "bc_open_to_buy",
        "mths_since_recent_bc", "bc_util", "total_bc_limit",
    ]
    top15_names = set(importance.head(15)["feature"])
    rollout_in_top15 = top15_names & set(rollout_related)
    print(f"\nRollout-affected columns in top 15 by gain: {rollout_in_top15 or 'none'}")

    print("\n=== Calibration: out-of-sample time-split (fit Jan-Sep 2014, evaluate Oct-Dec 2014) ===")
    calib_result, chosen_calib_method, final_calibrator = calibrate_out_of_sample(
        validation, chosen_val_pred, y_validation
    )
    with open(METRICS_DIR / "gbm_calibration.json", "w") as f:
        json.dump(calib_result, f, indent=2)
    print(json.dumps(calib_result, indent=2))

    import joblib

    lgb_dir = candidate_dir / "lightgbm"
    xgb_dir = candidate_dir / "xgboost"
    lgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(chosen_booster, lgb_dir / "model_chosen.joblib")
    joblib.dump(lgb_boosters[best_trial["trial"]], lgb_dir / "model_best_unconstrained.joblib")
    joblib.dump(final_calibrator, lgb_dir / "calibrator.joblib")
    with open(lgb_dir / "features.json", "w") as f:
        json.dump(feats, f, indent=2)

    joblib.dump(best_xgb["booster"], xgb_dir / "model.joblib")
    with open(xgb_dir / "features.json", "w") as f:
        json.dump(feats, f, indent=2)

    print(f"\nDone. Candidate artifacts written to {candidate_dir} (frozen artifacts untouched).")

    return {
        "feats": feats,
        "run_id": run_id,
        "candidate_dir": str(candidate_dir),
        "chosen_trial": chosen_trial,
        "chosen_metrics": chosen_metrics,
        "best_trial": best_trial,
        "best_metrics": best_metrics,
        "best_xgb": best_xgb,
        "calib_result": calib_result,
        "chosen_calib_method": chosen_calib_method,
    }


def run_with_mlflow() -> None:
    """Runs the unchanged main() pipeline (with per-trial child-run
    logging switched on), then logs the parent run's params/metrics/
    artifacts and registers the champion model. Logging never changes the
    computation - every training call, prediction, and file write in
    main() is identical whether this wrapper or main() directly is
    called."""
    import mlflow

    from src.models.mlflow_utils import (
        log_common_context,
        log_file_if_exists,
        register_champion,
        setup_mlflow,
    )

    setup_mlflow()
    with mlflow.start_run(run_name="gbm"):
        log_common_context()
        mlflow.log_params({
            "n_tuning_trials_lgb_unconstrained": N_TUNING_TRIALS_LGB,
            "n_tuning_trials_lgb_regularized": N_TUNING_TRIALS_LGB_REGULARIZED,
            "n_tuning_trials_xgb": N_TUNING_TRIALS_XGB,
            "simplicity_auc_tolerance": SIMPLICITY_AUC_TOLERANCE,
            "seed": SEED,
        })

        run_id = new_run_id()
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("promote_requested", promote_requested())
        result = main(log_to_mlflow=True, run_id=run_id)
        feats = result["feats"]
        candidate_dir = Path(result["candidate_dir"])

        with open(METRICS_DIR / "gbm_validation_metrics.json") as f:
            val_metrics = json.load(f)
        chosen = val_metrics["chosen"]
        mlflow.log_params({
            "chosen_trial": chosen["trial"],
            "chosen_n_trees": chosen["n_trees"],
        })
        mlflow.log_metrics({
            "chosen_train_auc": chosen["train_auc"],
            "chosen_validation_auc": chosen["validation_auc"],
            "chosen_validation_gini": chosen["validation_gini"],
            "chosen_validation_ks": chosen["validation_ks"],
            "chosen_train_validation_gap": chosen["train_minus_validation_auc_gap"],
            "xgboost_validation_auc": result["best_xgb"]["auc"],
        })

        # Test (2015) is NOT rescored here - attached from the already
        # committed Phase 5 files for reference only.
        test_metrics_path = METRICS_DIR / "test_core_metrics.json"
        if test_metrics_path.exists():
            with open(test_metrics_path) as f:
                test_metrics = json.load(f)
            for model_key, prefix in [("lightgbm", "test_lightgbm"), ("xgboost", "test_xgboost")]:
                m = test_metrics.get(model_key, {})
                mlflow.log_metrics({
                    f"{prefix}_{k}": v for k, v in m.items() if isinstance(v, (int, float))
                })

        for fname in [
            "gbm_feature_list.json", "gbm_tuning_trials.csv", "gbm_validation_metrics.json",
            "gbm_feature_importance.csv", "gbm_calibration.json", "frozen_model_config.json",
            "shap_summary.json", "shap_global_importance.csv",
        ]:
            log_file_if_exists(str(METRICS_DIR / fname))
        log_file_if_exists("reports/figures/shap_dependence_top6.png")

        if not promote_requested():
            print(f"\n--promote not passed: frozen data/processed/models/ artifacts are untouched. "
                  f"This run's output is at {candidate_dir}. Rerun with --promote to make it champion.")
            print(f"MLflow run logged: {mlflow.active_run().info.run_id}")
            return

        print(f"\n--promote passed: copying {candidate_dir} over the frozen artifacts...")
        promote_candidate_artifacts(candidate_dir / "lightgbm", MODELS_DIR / "lightgbm")
        promote_candidate_artifacts(candidate_dir / "xgboost", MODELS_DIR / "xgboost")

        # --- Register the champion: LightGBM booster + Platt calibrator,
        # chained as one deployable pyfunc model, with a real signature
        # and input example (plain string dtype for categoricals - see
        # LightGBMChampionModel docstring for why).
        model_info = register_champion(feats, MODELS_DIR / "lightgbm", VALIDATION_PARQUET)
        print(f"\nRegistered credit_default_lightgbm_champion v{model_info.registered_model_version}, "
              f"alias 'champion' set.")
        print(f"MLflow run logged: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_with_mlflow()
