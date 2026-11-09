"""Risk bands for the API's /score response: quintiles of the frozen
champion's calibrated PD on validation (2014) - defined on validation,
never on test, same split-discipline as the rest of this project.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.score_models import load_frozen_config, score_lightgbm

METRICS_DIR = Path("reports/metrics")
VALIDATION_PARQUET = "data/processed/validation.parquet"
OUT_PATH = Path("src/api/risk_bands.json")

BAND_NAMES = ["Very Low", "Low", "Medium", "High", "Very High"]
PERCENTILES = [0, 20, 40, 60, 80, 100]


def main() -> None:
    frozen = load_frozen_config()
    validation = pd.read_parquet(VALIDATION_PARQUET)
    pd_scores = score_lightgbm(validation, frozen["lightgbm_chosen"])

    edges = np.percentile(pd_scores, PERCENTILES)

    bands = []
    for i, name in enumerate(BAND_NAMES):
        lo, hi = float(edges[i]), float(edges[i + 1])
        is_last = i == len(BAND_NAMES) - 1
        mask = (pd_scores >= lo) & (pd_scores <= hi if is_last else pd_scores < hi)
        n = int(mask.sum())
        default_rate = float(validation["target"].values[mask].mean())
        bands.append({
            "band": name,
            "pd_min": round(lo, 6),
            "pd_max": round(hi, 6),
            "pd_max_inclusive": is_last,
            "validation_n": n,
            "validation_default_rate_pct": round(default_rate * 100, 2),
        })

    output = {
        "methodology": "Quintiles (20% each) of the champion's calibrated PD on validation (2014).",
        "defined_on": "validation",
        "bands": bands,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    with open(METRICS_DIR / "risk_bands.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
