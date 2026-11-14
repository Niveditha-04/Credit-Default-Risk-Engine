"""Reproduces v2, the served LightGBM champion (refit on 2012-2014, the
frozen v1 hyperparameters and tree count, 8 monotone constraints, Platt
calibrator fit on 2015).

Every run refits v2 and writes it to data/processed/models/candidates/
v2_promotion/lightgbm/ - the frozen, served data/processed/models/lightgbm/
path, frozen_model_config.json, and MLflow are untouched unless --promote
is passed on the command line, same convention as src.models.gbm's own
--promote flag. This never reads reports/metrics/test_scored.parquet,
the locked one-time test evaluation file, and never writes to it.

v2 was defined and evaluated against a pre-registered rule in
docs/v2_decision_rule.md, committed before any v2 code ran. That
evaluation (reports/metrics/posthoc_v2_evaluation.json) said adopt v2 on
2016 vintages, confirmed by a label robustness check
(reports/metrics/posthoc_v2_label_robustness.json). v1's artifacts are
preserved as history, not deleted, the one time this script is run
with --promote.

Steps:
1. Refit v2 (same deterministic configuration already evaluated - no
   tuning, reproduces exactly).
2. Fit v2's own Platt calibrator on 2015 (test) - calibration only, not
   evaluation: v2 was already selected on 2016 vintages, before this
   read of test. Out-of-sample within 2015 itself (fit Jan-Sep, holdout
   -check Oct-Dec, refit on the full year for deployment), the same
   discipline already used for v1's own calibration on 2014. Logged as
   a new, explicit, authorized read of test.
3. Writes v2's booster, calibrator, and (unchanged - same 69 features,
   same order) features.json to a candidate directory.

With --promote, in addition:
4. Backs up the frozen data/processed/models/lightgbm/ path (v1's
   artifacts, the only time this is the live model) to
   data/processed/models/lightgbm_v1_history/, then promotes the v2
   candidate over the frozen path using src.models.gbm's own
   promote_candidate_artifacts (unlocks, copies, re-locks read-only).
5. Updates reports/metrics/frozen_model_config.json: lightgbm_chosen now
   describes v2; the previous lightgbm_chosen content is preserved under
   a new key, lightgbm_v1_history, with its model_path/calibrator.path
   pointed at the backed-up files (not the live path, which this step
   is about to overwrite). scorecard and xgboost sections are untouched.
6. In MLflow: sets alias "v1_champion" on the current champion version
   (so it stays addressable by name; MLflow reserves alias names
   matching ^v\\d+$, so a literal "v1" alias is rejected), then registers
   v2 as a new version and lets register_champion_standalone() set
   "champion" on it.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list, lock_frozen_file, promote_candidate_artifacts
from src.models.math_utils import apply_calibrator, logit
from src.models.mlflow_utils import (
    TRACKING_URI,
    register_champion_standalone,
    setup_mlflow,
)
from src.models.posthoc_v2 import (
    FROZEN_LGB_PARAMS,
    FROZEN_N_TREES,
    MONOTONE_DIRECTIONS,
)

METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")
FROZEN_LGB_DIR = MODELS_DIR / "lightgbm"
V1_HISTORY_DIR = MODELS_DIR / "lightgbm_v1_history"
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
REGISTERED_MODEL_NAME = "credit_default_lightgbm_champion"


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def calibrate_v2_on_test(booster: lgb.Booster, champion_feats: list[str]) -> tuple[LogisticRegression, dict]:
    test = pd.read_parquet(TEST_PARQUET)
    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")

    log_access(
        "promote_v2.py",
        len(test),
        "Explicit, authorized read of test (2015) for CALIBRATION ONLY, not evaluation: "
        "v2 was already selected against v1 on 2016 vintages, before this read, under the "
        "pre-registered rule in docs/v2_decision_rule.md. This fits v2's own Platt "
        "calibrator, out-of-sample within 2015 itself, the same discipline already used "
        "for v1's calibration on 2014. No model selection happens here.",
    )

    raw_pred = booster.predict(test[champion_feats], num_iteration=booster.best_iteration)
    y = test["target"].values
    issue_month = test["issue_d"].values
    fit_mask = issue_month < pd.Timestamp("2015-10-01").to_datetime64()
    holdout_mask = ~fit_mask

    fit_pred, fit_y = raw_pred[fit_mask], y[fit_mask]
    holdout_pred, holdout_y = raw_pred[holdout_mask], y[holdout_mask]

    platt_fit = LogisticRegression()
    platt_fit.fit(logit(fit_pred).reshape(-1, 1), fit_y)
    holdout_calibrated = apply_calibrator(platt_fit, holdout_pred)

    evidence = {
        "method": "platt",
        "fit_period": "2015-01 to 2015-09",
        "holdout_period": "2015-10 to 2015-12",
        "n_fit": int(fit_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "brier_uncalibrated_holdout": round(float(brier_score_loss(holdout_y, holdout_pred)), 5),
        "brier_platt_holdout": round(float(brier_score_loss(holdout_y, holdout_calibrated)), 5),
        "mean_predicted_pd_before_holdout_pct": round(float(holdout_pred.mean()) * 100, 4),
        "actual_default_rate_holdout_pct": round(float(holdout_y.mean()) * 100, 4),
        "mean_predicted_pd_after_platt_holdout_pct": round(float(holdout_calibrated.mean()) * 100, 4),
    }

    # Refit on the full year for deployment.
    final_calibrator = LogisticRegression()
    final_calibrator.fit(logit(raw_pred).reshape(-1, 1), y)
    final_pred_full = apply_calibrator(final_calibrator, raw_pred)
    evidence["final_calibrator_refit_on"] = "full 2015 (Jan-Dec)"
    evidence["mean_predicted_pd_before_full_2015_pct"] = round(float(raw_pred.mean()) * 100, 4)
    evidence["actual_default_rate_full_2015_pct"] = round(float(y.mean()) * 100, 4)
    evidence["mean_predicted_pd_after_full_2015_pct"] = round(float(final_pred_full.mean()) * 100, 4)

    return final_calibrator, evidence


def promote_requested() -> bool:
    return "--promote" in sys.argv


def main() -> None:
    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    print("=== Step 1: refit v2 ===")
    booster = fit_v2(champion_feats)
    print(f"v2 trees: {booster.num_trees()}")

    print("\n=== Step 2: fit v2's Platt calibrator on 2015 (calibration only) ===")
    calibrator, calib_evidence = calibrate_v2_on_test(booster, champion_feats)
    print(json.dumps(calib_evidence, indent=2))

    print("\n=== Step 3: write v2 candidate ===")
    candidate_dir = MODELS_DIR / "candidates" / "v2_promotion" / "lightgbm"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, candidate_dir / "model_chosen.joblib")
    joblib.dump(calibrator, candidate_dir / "calibrator.joblib")
    shutil.copy2(FROZEN_LGB_DIR / "features.json", candidate_dir / "features.json")
    candidate_hash = sha256_of(candidate_dir / "model_chosen.joblib")
    print(f"v2 candidate written to {candidate_dir} (SHA-256 {candidate_hash}).")

    if not promote_requested():
        print(
            "\n--promote not passed: stopping here. The frozen data/processed/models/lightgbm/ "
            "path, frozen_model_config.json, and MLflow are untouched. Rerun with --promote "
            "to deploy this candidate as the served model."
        )
        return

    print("\n=== --promote passed: back up v1's frozen artifacts ===")
    if V1_HISTORY_DIR.exists():
        shutil.rmtree(V1_HISTORY_DIR)
    V1_HISTORY_DIR.mkdir(parents=True)
    for fname in ["model_chosen.joblib", "calibrator.joblib", "features.json"]:
        shutil.copy2(FROZEN_LGB_DIR / fname, V1_HISTORY_DIR / fname)
        lock_frozen_file(V1_HISTORY_DIR / fname)
    print(f"v1 artifacts backed up to {V1_HISTORY_DIR}")

    print("\n=== Step 4: promote the v2 candidate over the frozen path ===")
    promote_candidate_artifacts(candidate_dir, FROZEN_LGB_DIR)

    v2_hash = sha256_of(FROZEN_LGB_DIR / "model_chosen.joblib")
    assert v2_hash == candidate_hash, "promoted artifact hash does not match the candidate just written"
    print(f"v2 model_chosen.joblib SHA-256: {v2_hash}")

    print("\n=== Step 5: update frozen_model_config.json ===")
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)

    v1_chosen = frozen["lightgbm_chosen"]
    frozen["lightgbm_v1_history"] = {
        **v1_chosen,
        "model_path": str(V1_HISTORY_DIR / "model_chosen.joblib"),
        "calibrator": {**v1_chosen["calibrator"], "path": str(V1_HISTORY_DIR / "calibrator.joblib")},
        "note": "v1, the champion before v2. Artifacts preserved, read-only, at "
        f"{V1_HISTORY_DIR}, registered in MLflow with alias 'v1_champion' ('v1' itself is a "
        "reserved MLflow alias pattern).",
    }
    frozen["lightgbm_chosen"] = {
        "model_type": "lightgbm",
        "role": "champion (v2)",
        "selection_rule": (
            "Pre-registered in docs/v2_decision_rule.md before any v2 code ran: adopt v2 "
            "if AUC(v2) >= AUC(v1) - 0.002 and ECE(v2) <= ECE(v1) + 0.005 on 2016 vintages "
            "(issue <= 2016-10-31, 24-month label). Result: AUC(v2)=0.6978 vs. "
            "AUC(v1)=0.6899, ECE(v2)=0.0079 vs. ECE(v1)=0.0080 - both conditions met. "
            "Confirmed by a label robustness check (reports/metrics/posthoc_v2_label_robustness.json)."
        ),
        "n_features": len(champion_feats),
        "features": champion_feats,
        "hyperparameters": {**{k: v for k, v in FROZEN_LGB_PARAMS.items()}, "monotone_constraints": "see MONOTONE_DIRECTIONS in src/models/posthoc_v2.py"},
        "n_trees": FROZEN_N_TREES,
        "trained_on": "2012-2014 (train + validation concatenated)",
        "model_path": str(FROZEN_LGB_DIR / "model_chosen.joblib"),
        "model_sha256": v2_hash,
        "calibrator": {
            "method": "platt",
            "path": str(FROZEN_LGB_DIR / "calibrator.joblib"),
            "fit_on": calib_evidence["final_calibrator_refit_on"],
            "out_of_sample_selection_evidence": calib_evidence,
        },
        "v1_history_key": "lightgbm_v1_history",
    }
    with open(METRICS_DIR / "frozen_model_config.json", "w") as f:
        json.dump(frozen, f, indent=2)
    print("frozen_model_config.json updated: lightgbm_chosen = v2, lightgbm_v1_history = v1")

    print("\n=== Step 6: MLflow - alias v1 on the current champion, then register v2 as champion ===")
    setup_mlflow()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    current_champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    # MLflow reserves alias names matching ^v\d+$ (collides with its own
    # models:/name/<version-number> addressing), so a literal "v1" alias
    # is rejected - "v1_champion" carries the same meaning without
    # hitting that reserved pattern.
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "v1_champion", current_champion.version)
    print(f"Alias 'v1_champion' set on version {current_champion.version} (the model that was 'champion' until now)")

    register_champion_standalone()

    summary = {
        "label": "promotion (not post-hoc/exploratory - this changes the served model)",
        "v2_model_sha256": v2_hash,
        "v1_history_artifacts": str(V1_HISTORY_DIR),
        "v1_mlflow_alias": "v1_champion",
        "calibration_evidence": calib_evidence,
    }
    with open(METRICS_DIR / "v2_promotion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
