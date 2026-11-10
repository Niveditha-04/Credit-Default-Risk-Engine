"""Builds the two small reference files the API loads at startup
(src/api/feature_ranges.json, src/api/risk_bands.json), both derived
only from TRAIN or VALIDATION - never test, same split discipline as
everything else in this project.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.target import labelled_population_query
from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.features.allow_list import feature_columns
from src.features.feature_engineering import REQUIRED_RAW_COLUMNS

METRICS_DIR = Path("reports/metrics")
FEATURE_RANGES_PATH = Path("src/api/feature_ranges.json")
RISK_BANDS_PATH = Path("src/api/risk_bands.json")
VALIDATION_PARQUET = "data/processed/validation.parquet"

CATEGORICAL_INPUT_FIELDS = ["home_ownership", "purpose", "verification_status"]
BAND_NAMES = ["Very Low", "Low", "Medium", "High", "Very High"]
PERCENTILES = [0, 20, 40, 60, 80, 100]


def compute_feature_ranges() -> None:
    """Training-data ranges for every API input field, used at request
    time to flag (not reject) values outside what the model was trained
    on. Computed from TRAIN only."""
    raw_feats = feature_columns()
    derivation_only = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_feats and c not in ("id", "issue_d", "split", "target")]
    input_fields = sorted(set(raw_feats) | set(derivation_only))

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    cols = ", ".join(f'"{c}"' for c in input_fields)
    query = f"SELECT {cols} FROM ({labelled_population_query()}) WHERE split = 'train' AND target IS NOT NULL"
    train = con.execute(query).fetchdf()

    ranges = {}
    for col in input_fields:
        if col in CATEGORICAL_INPUT_FIELDS:
            ranges[col] = {"type": "categorical", "known_values": sorted(train[col].dropna().unique().tolist())}
        elif col == "earliest_cr_line":
            ranges[col] = {
                "type": "date",
                "min": str(pd.to_datetime(train[col]).min().date()),
                "max": str(pd.to_datetime(train[col]).max().date()),
            }
        else:
            series = train[col].dropna()
            ranges[col] = {
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "pct_missing_in_train": round(100 * train[col].isna().mean(), 2),
            }

    with open(FEATURE_RANGES_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"Wrote {FEATURE_RANGES_PATH} ({len(ranges)} fields)")
    print(json.dumps(ranges, indent=2)[:2000])


def compute_risk_bands() -> None:
    """Quintiles of the frozen champion's calibrated PD on validation
    (2014) - defined on validation, never on test."""
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
    with open(RISK_BANDS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    with open(METRICS_DIR / "risk_bands.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    compute_feature_ranges()
    print()
    compute_risk_bands()
