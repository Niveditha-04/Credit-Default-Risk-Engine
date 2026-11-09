"""Load models:/credit_default_lightgbm_champion@champion from the MLflow
registry and assert its predictions equal the frozen model's predictions
exactly (not just close) on the full validation set. This is the proof
that registering the model through MLflow's pyfunc flavor didn't change
anything about how it scores a loan.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.score_models import score_lightgbm
from src.models.mlflow_utils import TRACKING_URI

METRICS_DIR = Path("reports/metrics")
VALIDATION_PARQUET = "data/processed/validation.parquet"
MODEL_URI = "models:/credit_default_lightgbm_champion@champion"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    loaded = mlflow.pyfunc.load_model(MODEL_URI)

    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]

    validation = pd.read_parquet(VALIDATION_PARQUET)
    registry_preds = np.array(loaded.predict(validation[cfg["features"]]))
    frozen_preds = score_lightgbm(validation, cfg)

    max_abs_diff = float(np.max(np.abs(registry_preds - frozen_preds)))
    exactly_equal = bool(np.array_equal(registry_preds, frozen_preds))

    result = {
        "model_uri": MODEL_URI,
        "n_rows_compared": len(validation),
        "max_abs_diff": max_abs_diff,
        "exactly_equal": exactly_equal,
    }
    assert exactly_equal, f"Registry model predictions do not exactly match the frozen model: {result}"

    with open(METRICS_DIR / "mlflow_champion_alias_verification.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
