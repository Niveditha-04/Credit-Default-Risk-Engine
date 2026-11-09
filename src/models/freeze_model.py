"""Freeze every model artifact that Phase 5 is allowed to touch.

Reads only already-saved metrics files from Phase 3 and Phase 4 (never
retrains anything) and writes reports/metrics/frozen_model_config.json.
Phase 5 must load its models, features, and calibrator from this file
alone - no retraining, no re-tuning, no feature changes once test (2015)
is in scope.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("data/processed/models")
OUT_PATH = METRICS_DIR / "frozen_model_config.json"


def git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    scorecard_metrics = load_json(METRICS_DIR / "scorecard_validation_metrics.json")
    selection_log = load_json(METRICS_DIR / "feature_selection_log.json")
    score_summary = load_json(METRICS_DIR / "scorecard_score_summary.json")

    gbm_metrics = load_json(METRICS_DIR / "gbm_validation_metrics.json")
    gbm_feats = load_json(METRICS_DIR / "gbm_feature_list.json")
    gbm_calib = load_json(METRICS_DIR / "gbm_calibration.json")
    tuning_trials = load_json(METRICS_DIR / "gbm_tuning_trials.json")

    chosen_trial = next(t for t in tuning_trials if t["trial"] == gbm_metrics["chosen"]["trial"])
    best_xgb_trial = max((t for t in tuning_trials if t["model"] == "xgboost"), key=lambda t: t["auc"])

    chosen_hparams = {
        k: v for k, v in chosen_trial.items()
        if k not in ("trial", "model", "search", "auc", "best_iteration")
    }
    xgb_hparams = {
        k: v for k, v in best_xgb_trial.items()
        if k not in ("trial", "model", "search", "auc", "best_iteration")
    }

    config = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash_at_freeze": git_hash(),
        "note_on_git_hash": "This is the HEAD commit BEFORE the freeze commit that adds this file itself.",
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
            "n_features": len(selection_log["final_features"]),
            "features": selection_log["final_features"],
            "binning_process_path": "data/processed/models/binning_process.joblib",
            "model_path": "data/processed/models/scorecard_logreg.joblib",
            "points_scaling": {
                "base_points": score_summary["base_points"],
                "base_odds": score_summary["base_odds"],
                "pdo": score_summary["pdo"],
                "factor": score_summary["factor"],
                "offset": score_summary["offset"],
            },
            "train_validation_metrics": scorecard_metrics,
        },
        "lightgbm_chosen": {
            "model_type": "lightgbm",
            "selection_rule": gbm_metrics["selection_rule"],
            "n_features": len(gbm_feats),
            "features": gbm_feats,
            "hyperparameters": chosen_hparams,
            "n_trees": gbm_metrics["chosen"]["n_trees"],
            "model_path": "data/processed/models/lightgbm_model.joblib",
            "calibrator": {
                "method": gbm_calib["chosen_method"],
                "path": "data/processed/models/lightgbm_calibrator.joblib",
                "fit_on": gbm_calib["final_calibrator_refit_on"],
                "out_of_sample_selection_evidence": {
                    "fit_period": gbm_calib["fit_period"],
                    "holdout_period": gbm_calib["holdout_period"],
                    "brier_uncalibrated_holdout": gbm_calib["brier_uncalibrated_holdout"],
                    "brier_isotonic_holdout": gbm_calib["brier_isotonic_holdout"],
                    "brier_platt_holdout": gbm_calib["brier_platt_holdout"],
                },
            },
            "train_validation_metrics": gbm_metrics["chosen"],
        },
        "lightgbm_best_unconstrained_reference_only": {
            "note": "Not used in Phase 5. Kept only to show what the selection rule traded away.",
            "hyperparameters": {
                k: v for k, v in (next(t for t in tuning_trials if t["trial"] == gbm_metrics["best"]["trial"])).items()
                if k not in ("trial", "model", "search", "auc", "best_iteration")
            },
            "n_trees": gbm_metrics["best"]["n_trees"],
            "model_path": "data/processed/models/lightgbm_model_best_unconstrained.joblib",
            "train_validation_metrics": gbm_metrics["best"],
        },
        "xgboost": {
            "model_type": "xgboost",
            "n_features": len(gbm_feats),
            "features": gbm_feats,
            "hyperparameters": xgb_hparams,
            "n_trees": int(best_xgb_trial["best_iteration"]),
            "validation_auc": best_xgb_trial["auc"],
            "model_path": "data/processed/models/xgboost_model.joblib",
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
