"""Post-incident evidence protection (2026-11-10). The only model
artifacts that survived the row-order/bagging incident with a verifiable
backup were the ones already logged to MLflow (the LightGBM champion).
This locks down the one remaining piece of evidence that was NOT backed
up anywhere until now: the saved test (2015) predictions
(reports/metrics/test_scored.parquet), which is what reports/metrics/
test_core_metrics.json and frozen_model_config.json's test_* numbers
were computed from. If that file were ever silently overwritten the way
the model artifacts were, there would be no way to re-derive those
already-committed numbers without re-touching test, which the project's
rules forbid.

There is no separate saved "validation predictions" file anywhere in
this repo - validation is always rescored on the fly, never cached to
disk - so there is nothing else to protect here.

Steps: hash test_scored.parquet, make it read-only, upload it as an
MLflow artifact on the run that actually backs the current frozen
model (f49885e7d05c41afa4a3a811defb5b6f - the original "gbm" run,
restored and verified byte-identical in the 2026-11-10 recovery), and
record the hash in reports/metrics/evidence_hashes.json. Also recomputes
the XGBoost test AUC directly from the saved predictions as a live check
against the frozen number (0.6886).
"""

import json
import os
import stat
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.gbm import lock_frozen_file
from src.models.mlflow_utils import TRACKING_URI

METRICS_DIR = Path("reports/metrics")
TEST_SCORED_PARQUET = METRICS_DIR / "test_scored.parquet"
GBM_PARENT_RUN_ID = "f49885e7d05c41afa4a3a811defb5b6f"
FROZEN_XGBOOST_TEST_AUC = 0.6886


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert TEST_SCORED_PARQUET.exists(), f"{TEST_SCORED_PARQUET} not found"

    test_scored_hash = sha256_of(TEST_SCORED_PARQUET)
    lock_frozen_file(TEST_SCORED_PARQUET)
    mode = stat.S_IMODE(os.stat(TEST_SCORED_PARQUET).st_mode)
    print(f"{TEST_SCORED_PARQUET}: sha256={test_scored_hash}, now read-only (mode {oct(mode)})")

    df = pd.read_parquet(TEST_SCORED_PARQUET)
    recomputed_xgb_auc = round(float(roc_auc_score(df["target"], df["pd_xgboost"])), 4)
    print(f"Recomputed XGBoost test AUC from saved predictions: {recomputed_xgb_auc} "
          f"(frozen: {FROZEN_XGBOOST_TEST_AUC})")
    assert recomputed_xgb_auc == FROZEN_XGBOOST_TEST_AUC, (
        f"recomputed {recomputed_xgb_auc} != frozen {FROZEN_XGBOOST_TEST_AUC} - "
        "test_scored.parquet may not be the original scoring pass"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    client.log_artifact(GBM_PARENT_RUN_ID, str(TEST_SCORED_PARQUET))
    print(f"Logged {TEST_SCORED_PARQUET} as an artifact on run {GBM_PARENT_RUN_ID}")

    result = {
        "test_scored_parquet": {
            "path": str(TEST_SCORED_PARQUET),
            "sha256": test_scored_hash,
            "made_read_only": True,
            "logged_as_mlflow_artifact_on_run": GBM_PARENT_RUN_ID,
        },
        "saved_validation_predictions": {
            "exists": False,
            "note": "no separate validation predictions file exists in this repo - "
            "validation is always rescored on the fly from the frozen artifacts, never cached to disk",
        },
        "xgboost_test_auc_recomputed_from_saved_predictions": recomputed_xgb_auc,
        "xgboost_test_auc_frozen": FROZEN_XGBOOST_TEST_AUC,
        "match": recomputed_xgb_auc == FROZEN_XGBOOST_TEST_AUC,
    }
    with open(METRICS_DIR / "evidence_hashes.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
