"""One-off: re-register the champion with a corrected MLflow signature.

The original registration (gbm.py's run_with_mlflow) inferred its schema
from validation.head(5), which by chance had no missing emp_length in
that 5-row sample, so MLflow declared emp_length a non-nullable integer.
Scoring the full validation set through the registered alias then failed
with "cannot convert NA to integer" (emp_length is ~6% missing). The
booster and calibrator artifacts themselves are untouched - this only
re-logs them with build_signature_sample()'s fuller sample, which is now
also what gbm.py itself uses, so a future full rerun won't reintroduce
this.

Does not re-run the tuning search: model_chosen.joblib and
calibrator.joblib on disk are already the frozen, verified champion.
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
    with mlflow.start_run(run_name="gbm_reregister_champion_fixed_signature"):
        log_common_context()
        mlflow.set_tag("purpose", "fix_mlflow_schema_emp_length_integer_bug")
        model_info = register_champion(feats, MODELS_DIR / "lightgbm", VALIDATION_PARQUET)
        print(f"Registered credit_default_lightgbm_champion v{model_info.registered_model_version}, "
              f"alias 'champion' set.")
        print(f"MLflow run logged: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
