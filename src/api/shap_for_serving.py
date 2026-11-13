"""Mean |SHAP| ranking for whichever model is currently served (the
frozen `lightgbm_chosen`), on test (2015), full set, no sampling - the
exact top-10 of this ranking is what src.api.scoring.ScoringEngine uses
to decide whether a missing field warrants a "lowers fidelity" warning.

This is a narrower, serving-focused sibling of src/explain/shap_analysis.py
(which does the full SHAP writeup - dependence plots, income-group check,
WOE-direction comparison - for v1, already documented throughout
README.md and docs/validation_report.md as v1's result). Re-running
shap_analysis.py itself for a newly-served model would silently break
that existing, already-cited documentation; this script only produces
what src.api.scoring.ScoringEngine actually reads at request time, and
is meant to be rerun after every promotion.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
TEST_PARQUET = "data/processed/test.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
OUT_CSV = METRICS_DIR / "shap_global_importance_served.csv"
OUT_JSON = METRICS_DIR / "shap_global_importance_served_summary.json"


def sha256_of(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]
    feats = cfg["features"]
    model_hash = sha256_of(cfg["model_path"])

    test = pd.read_parquet(TEST_PARQUET)
    log_access(
        "shap_for_serving.py",
        len(test),
        "Serving asset, not evaluation: re-read test.parquet to rank features by mean "
        "|SHAP| for the currently-served model, used only by the API's missing-field "
        "warning. No model, feature, or policy change.",
    )
    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")
    X = test[feats]

    booster = joblib.load(cfg["model_path"])
    contrib = booster.predict(X, pred_contrib=True, num_iteration=booster.best_iteration)
    shap_values = contrib[:, :-1]
    base_value = contrib[:, -1]

    raw_pred = booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)
    max_diff = float(np.max(np.abs(shap_values.sum(axis=1) + base_value - raw_pred)))
    assert max_diff < 1e-6, f"additivity check failed: max diff {max_diff:.2e}"

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
    importance = pd.DataFrame({"mean_abs_shap": mean_abs_shap})
    importance["mean_abs_shap_rank"] = range(1, len(importance) + 1)
    importance.to_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV} ({len(importance)} features), additivity max diff {max_diff:.2e}")
    print(importance.head(10).to_string())

    summary = {
        "source_model_sha256": model_hash,
        "source_model_path": cfg["model_path"],
        "computed_on": "test (2015), full set, no sampling",
        "n_rows": len(test),
        "top_10": mean_abs_shap.head(10).index.tolist(),
        "used_by": "src.api.scoring.ScoringEngine for the missing_top10_shap_feature_lowers_fidelity warning",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
