"""Registers the champion in MLflow using the already-trained, already-
frozen booster and calibrator on disk - no tuning search, no retraining.

The signature MLflow infers for a registered model is built from a
sample of rows (build_signature_sample() in mlflow_utils.py), chosen to
include a missing value in every column that has one anywhere in
validation. emp_length is about 6% missing, so the sample must contain
at least one missing emp_length row for MLflow to register it as a
nullable column; scoring any row with a missing emp_length through a
registered model that lacks this guarantee would fail schema
enforcement.
"""

import json
from pathlib import Path

import mlflow

from src.models.mlflow_utils import log_common_context, register_champion, setup_mlflow

METRICS_DIR = Path("reports/metrics")
VALIDATION_PARQUET = "data/processed/validation.parquet"
MODELS_DIR = Path("data/processed/models")


def main() -> None:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    feats = frozen["lightgbm_chosen"]["features"]

    setup_mlflow()
    with mlflow.start_run(run_name="gbm_register_champion"):
        log_common_context()
        mlflow.set_tag("purpose", "register_champion_with_nullable_signature")
        model_info = register_champion(feats, MODELS_DIR / "lightgbm", VALIDATION_PARQUET)
        print(f"Registered credit_default_lightgbm_champion v{model_info.registered_model_version}, "
              f"alias 'champion' set.")
        print(f"MLflow run logged: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
