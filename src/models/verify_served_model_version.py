"""Confirms three things agree: the MLflow registry's @champion alias
(currently v3), the booster artifact baked into the local joblib file the
API loads, and the model_version string the API actually reports. The API
intentionally never talks to a live MLflow server at request time (it
only reads frozen_model_config.json + joblib files), so this is the one
place that closes the loop between "what MLflow says is champion" and
"what the running service is actually scoring with."
"""

import hashlib
import json
import tempfile
from pathlib import Path

import mlflow

from src.api.scoring import ScoringEngine
from src.models.mlflow_utils import TRACKING_URI

METRICS_DIR = Path("reports/metrics")
REGISTERED_MODEL_NAME = "credit_default_lightgbm_champion"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    alias_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")

    with open("reports/metrics/frozen_model_config.json") as f:
        frozen = json.load(f)
    local_path = frozen["lightgbm_chosen"]["model_path"]
    local_sha = sha256_of(local_path)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = client.download_artifacts(
            alias_version.run_id, "lightgbm_champion/artifacts/model_chosen.joblib", tmp
        )
        registry_sha = sha256_of(downloaded)

    engine = ScoringEngine()

    result = {
        "registry_alias_points_to_version": alias_version.version,
        "registry_booster_sha256": registry_sha,
        "local_baked_booster_sha256": local_sha,
        "registry_matches_local_artifact": registry_sha == local_sha,
        "served_model_version_string": engine.model_version,
        "served_sha256_prefix_matches_local": engine.model_version.split("+sha256:")[1] == local_sha[:12],
    }
    assert result["registry_matches_local_artifact"], result
    assert result["served_sha256_prefix_matches_local"], result

    with open(METRICS_DIR / "served_model_version_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
