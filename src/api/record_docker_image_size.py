"""Records the built Docker image's size and the free disk space at the
time of recording. Run right after `docker build`."""

import json
import shutil
import subprocess
from pathlib import Path

METRICS_DIR = Path("reports/metrics")
IMAGE = "credit-default-risk-engine:latest"


def main() -> None:
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


if __name__ == "__main__":
    main()
