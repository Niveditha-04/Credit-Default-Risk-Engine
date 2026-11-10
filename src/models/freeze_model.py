"""Freeze every model artifact that Phase 5 is allowed to touch.

Reads only already-saved metrics files from Phase 3 and Phase 4 (never
retrains anything) and writes reports/metrics/frozen_model_config.json.
Phase 5 must load its models, features, and calibrator from this file
alone - no retraining, no re-tuning, no feature changes once test (2015)
is in scope.

Each model's artifacts live in their own subfolder under
data/processed/models/: scorecard/, lightgbm/, xgboost/.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")
OUT_PATH = METRICS_DIR / "frozen_model_config.json"

CHAMPION_AUC_TOLERANCE_NOTE = (
    "lightgbm_chosen is the champion: within 0.0021 validation AUC of "
    "XGBoost (the best alternative) while using 273 trees instead of "
    "2122 - a much simpler model for a negligible performance difference. "
    "XGBoost and the scorecard are carried forward as comparison models "
    "in Phase 5, not candidates for champion."
)


def git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_status_clean() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return len(out.strip()) == 0


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    scorecard_metrics = load_json(METRICS_DIR / "scorecard_validation_metrics.json")
    selection_log = load_json(METRICS_DIR / "feature_selection_log.json")
    score_summary = load_json(METRICS_DIR / "scorecard_score_summary.json")
    scorecard_calib = load_json(METRICS_DIR / "scorecard_calibration.json")

    gbm_metrics = load_json(METRICS_DIR / "gbm_validation_metrics.json")
    gbm_feats = load_json(METRICS_DIR / "gbm_feature_list.json")
    gbm_calib = load_json(METRICS_DIR / "gbm_calibration.json")
    tuning_trials = load_json(METRICS_DIR / "gbm_tuning_trials.json")
    xgb_calib = load_json(METRICS_DIR / "xgboost_calibration.json")

    chosen_trial = next(t for t in tuning_trials if t["trial"] == gbm_metrics["chosen"]["trial"])
    best_trial = next(t for t in tuning_trials if t["trial"] == gbm_metrics["best"]["trial"])
    best_xgb_trial = max((t for t in tuning_trials if t["model"] == "xgboost"), key=lambda t: t["auc"])

    def hparams(trial: dict) -> dict:
        return {k: v for k, v in trial.items() if k not in ("trial", "model", "search", "auc", "best_iteration")}

    chosen_n_trees = int(chosen_trial["best_iteration"])
    chosen_num_leaves = int(chosen_trial["num_leaves"])
    best_n_trees = int(best_trial["best_iteration"])
    best_num_leaves = int(best_trial["num_leaves"])

    champion_vs_xgb_auc_gap = round(best_xgb_trial["auc"] - gbm_metrics["chosen"]["validation_auc"], 4)

    config = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_hash_at_freeze": git_hash(),
        "git_status_clean_at_freeze": git_status_clean(),
        "note_on_git_hash": "This is the HEAD commit BEFORE the freeze commit that adds this file itself.",
        "champion": "lightgbm_chosen",
        "champion_reason": CHAMPION_AUC_TOLERANCE_NOTE,
        "champion_vs_xgboost_validation_auc_gap": champion_vs_xgb_auc_gap,
        "seeds": {
            "global": 42,
            "lightgbm_unconstrained_search": 42,
            "lightgbm_regularized_search": 142,
            "xgboost_search": 43,
        },
        "test_touch_policy": (
            "Phase 5 touches test (2015) exactly once, scoring ONLY the frozen "
            "artifacts listed in this file. No retraining, no feature changes, "
            "no re-tuning, no re-selection of calibration method in Phase 5."
        ),
        "scorecard": {
            "model_type": "logistic_regression_on_woe",
            "role": "comparison",
            "n_features": len(selection_log["final_features"]),
            "features": selection_log["final_features"],
            "binning_process_path": "data/processed/models/scorecard/binning_process.joblib",
            "model_path": "data/processed/models/scorecard/model.joblib",
            "points_scaling": {
                "base_points": score_summary["base_points"],
                "base_odds": score_summary["base_odds"],
                "pdo": score_summary["pdo"],
                "factor": score_summary["factor"],
                "offset": score_summary["offset"],
            },
            "train_validation_metrics": scorecard_metrics,
            "calibrator": {
                "method": "platt",
                "path": "data/processed/models/scorecard/calibrator.joblib",
                "fit_on": scorecard_calib["final_calibrator_refit_on"],
                "out_of_sample_selection_evidence": scorecard_calib,
            },
        },
        "lightgbm_chosen": {
            "model_type": "lightgbm",
            "role": "champion",
            "selection_rule": gbm_metrics["selection_rule"],
            "n_features": len(gbm_feats),
            "features": gbm_feats,
            "hyperparameters": hparams(chosen_trial),
            "n_trees": chosen_n_trees,
            "num_leaves": chosen_num_leaves,
            "total_leaves": chosen_n_trees * chosen_num_leaves,
            "model_path": "data/processed/models/lightgbm/model_chosen.joblib",
            "calibrator": {
                "method": gbm_calib["chosen_method"],
                "path": "data/processed/models/lightgbm/calibrator.joblib",
                "fit_on": gbm_calib["final_calibrator_refit_on"],
                "out_of_sample_selection_evidence": {
                    "fit_period": gbm_calib["fit_period"],
                    "holdout_period": gbm_calib["holdout_period"],
                    "brier_uncalibrated_holdout": gbm_calib["brier_uncalibrated_holdout"],
                    "brier_isotonic_holdout": gbm_calib["brier_isotonic_holdout"],
                    "brier_platt_holdout": gbm_calib["brier_platt_holdout"],
                    "mean_predicted_pd_before_holdout_pct": gbm_calib["mean_predicted_pd_before_holdout_pct"],
                    "actual_default_rate_holdout_pct": gbm_calib["actual_default_rate_holdout_pct"],
                },
            },
            "train_validation_metrics": gbm_metrics["chosen"],
        },
        "lightgbm_best_unconstrained_reference_only": {
            "note": "Not used in Phase 5. Kept only to show what the simplicity-selection rule traded away.",
            "hyperparameters": hparams(best_trial),
            "n_trees": best_n_trees,
            "num_leaves": best_num_leaves,
            "total_leaves": best_n_trees * best_num_leaves,
            "model_path": "data/processed/models/lightgbm/model_best_unconstrained.joblib",
            "train_validation_metrics": gbm_metrics["best"],
        },
        "xgboost": {
            "model_type": "xgboost",
            "role": "comparison",
            "n_features": len(gbm_feats),
            "features": gbm_feats,
            "hyperparameters": hparams(best_xgb_trial),
            "n_trees": int(best_xgb_trial["best_iteration"]),
            "validation_auc": best_xgb_trial["auc"],
            "model_path": "data/processed/models/xgboost/model.joblib",
            "calibrator": {
                "method": "platt",
                "path": "data/processed/models/xgboost/calibrator.joblib",
                "fit_on": xgb_calib["final_calibrator_refit_on"],
                "out_of_sample_selection_evidence": xgb_calib,
            },
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
