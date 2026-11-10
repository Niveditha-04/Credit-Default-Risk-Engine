"""Post-build checks for the served API: latency against a running
Docker container, the built image's size, and that the registry/local
artifact/served model_version string all agree.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import mlflow
import numpy as np

METRICS_DIR = Path("reports/metrics")
BASE_URL = "http://localhost:8000"
N_REQUESTS = 1000
IMAGE = "credit-default-risk-engine:latest"
REGISTERED_MODEL_NAME = "credit_default_lightgbm_champion"

LATENCY_PAYLOAD = {
    "loan_amnt": 15000.0, "annual_inc": 65000.0, "dti": 18.5,
    "fico_range_low": 680.0, "fico_range_high": 684.0,
    "earliest_cr_line": "2005-03-01",
    "home_ownership": "RENT", "purpose": "debt_consolidation",
    "verification_status": "Verified", "emp_length": 5.0,
    "revol_util": 45.0, "revol_bal": 8000.0, "open_acc": 10.0,
    "total_acc": 20.0, "pub_rec": 0.0, "pub_rec_bankruptcies": 0.0,
    "inq_last_6mths": 1.0, "mort_acc": 1.0, "acc_open_past_24mths": 3.0,
}


def _load_average() -> list:
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return []


def _cpu_model() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def _run_sequential(client: httpx.Client, include_reason_codes: bool, n: int) -> np.ndarray:
    latencies_ms = np.empty(n)
    params = {"include_reason_codes": str(include_reason_codes).lower()}
    for i in range(n):
        t0 = time.perf_counter()
        resp = client.post(f"{BASE_URL}/score", json=LATENCY_PAYLOAD, params=params)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, resp.text
    return latencies_ms


def _summarize(latencies_ms: np.ndarray) -> dict:
    return {
        "n": len(latencies_ms),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "min_ms": round(float(latencies_ms.min()), 3),
        "max_ms": round(float(latencies_ms.max()), 3),
    }


def latency_test() -> None:
    """1,000 sequential requests against the RUNNING Docker container
    (not TestClient - a real HTTP round trip over localhost), with and
    without reason codes. These are local laptop numbers on one machine,
    not a production SLA measurement - no load balancer, no concurrency,
    no network beyond localhost."""
    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{BASE_URL}/health")
        assert health.status_code == 200, "container not reachable at " + BASE_URL
        print("Health check:", health.json())

        print(f"\nRunning {N_REQUESTS} sequential requests WITH reason codes...")
        with_reasons = _run_sequential(client, True, N_REQUESTS)

        print(f"Running {N_REQUESTS} sequential requests WITHOUT reason codes...")
        without_reasons = _run_sequential(client, False, N_REQUESTS)

    result = {
        "label": "local laptop numbers, not production - single container, no load balancer, sequential requests over localhost",
        "machine": {
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_1_5_15min": _load_average(),
            "note": "load average is whatever else was running on this laptop at measurement time "
            "(OS, browser, etc.) - nothing else from this project (no concurrent build, test, or "
            "training script) was running. Not a dedicated/isolated benchmark host.",
        },
        "n_requests_per_condition": N_REQUESTS,
        "with_reason_codes": _summarize(with_reasons),
        "without_reason_codes": _summarize(without_reasons),
    }
    with open(METRICS_DIR / "api_latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def record_docker_image_size() -> None:
    """Records the built Docker image's size and the free disk space at
    the time of recording. Run right after `docker build`."""
    image_id = subprocess.check_output(["docker", "images", "-q", IMAGE], text=True).strip()
    assert image_id, f"image {IMAGE} not found - run docker build first"

    size_bytes = int(subprocess.check_output(["docker", "inspect", IMAGE, "--format", "{{.Size}}"], text=True).strip())
    df = subprocess.check_output(["docker", "images", IMAGE, "--format", "{{.Size}}"], text=True).strip()

    free_bytes = shutil.disk_usage("/").free

    result = {
        "image": IMAGE,
        "image_id": image_id,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1_000_000, 1),
        "docker_images_reported_size": df,
        "free_disk_bytes_at_record_time": free_bytes,
        "free_disk_gb_at_record_time": round(free_bytes / 1_000_000_000, 1),
        "base_image": "python:3.11-slim",
        "note": "requirements-serve.txt no longer includes shap/matplotlib (see src/api/scoring.py - "
        "reason codes use booster.predict(pred_contrib=True) instead), which is most of the size drop "
        "from the first Phase 8 build.",
    }
    with open(METRICS_DIR / "docker_image.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_served_model_version() -> None:
    """Confirms three things agree: the MLflow registry's @champion alias,
    the booster artifact baked into the local joblib file the API loads,
    and the model_version string the API actually reports. The API
    intentionally never talks to a live MLflow server at request time (it
    only reads frozen_model_config.json + joblib files), so this is the
    one place that closes the loop between "what MLflow says is champion"
    and "what the running service is actually scoring with."
    """
    from src.api.scoring import ScoringEngine
    from src.models.mlflow_utils import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    alias_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")

    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    local_path = frozen["lightgbm_chosen"]["model_path"]
    local_sha = _sha256_of(local_path)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = client.download_artifacts(
            alias_version.run_id, "lightgbm_champion/artifacts/model_chosen.joblib", tmp
        )
        registry_sha = _sha256_of(downloaded)

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
    verify_served_model_version()
    print()
    record_docker_image_size()
    print()
    latency_test()
