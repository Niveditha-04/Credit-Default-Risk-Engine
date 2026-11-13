"""Builds the two small reference files the API loads at startup
(src/api/feature_ranges.json, src/api/risk_bands.json), rebuilt for
whichever model is currently served (src.evaluation.score_models'
frozen `lightgbm_chosen`). Each file records that model's SHA-256 so a
test can catch a serving asset left over from an earlier model.

Feature ranges come from train + validation (2012-2014), the exact
population the served model was trained on. Risk bands come from the
served model's calibrated PD on test (2015) - the year its calibrator
was fit on - never the 2016 vintages used for the pre-registered
decision rule, which carries live-evaluation meaning that must stay
untouched by a serving rebuild.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.target import labelled_population_query
from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.features.allow_list import feature_columns
from src.features.feature_engineering import REQUIRED_RAW_COLUMNS

METRICS_DIR = Path("reports/metrics")
FEATURE_RANGES_PATH = Path("src/api/feature_ranges.json")
RISK_BANDS_PATH = Path("src/api/risk_bands.json")
TEST_PARQUET = "data/processed/test.parquet"

CATEGORICAL_INPUT_FIELDS = ["home_ownership", "purpose", "verification_status"]
BAND_NAMES = ["Very Low", "Low", "Medium", "High", "Very High"]
PERCENTILES = [0, 20, 40, 60, 80, 100]


def _model_hash(frozen: dict) -> str:
    return frozen["lightgbm_chosen"]["model_sha256"]


def compute_feature_ranges() -> None:
    """Train+validation (2012-2014) ranges for every API input field,
    used at request time to flag (not reject) values outside what the
    served model was trained on. 2012-2014 because that is the served
    model's training population (v2 is refit on train+validation)."""
    frozen = load_frozen_config()
    raw_feats = feature_columns()
    derivation_only = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_feats and c not in ("id", "issue_d", "split", "target")]
    input_fields = sorted(set(raw_feats) | set(derivation_only))

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    cols = ", ".join(f'"{c}"' for c in input_fields)
    query = f"SELECT {cols} FROM ({labelled_population_query()}) WHERE split IN ('train', 'validation') AND target IS NOT NULL"
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

    ranges["_metadata"] = {
        "computed_on": "train + validation (2012-2014), the served model's training population",
        "source_model_sha256": _model_hash(frozen),
    }

    with open(FEATURE_RANGES_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"Wrote {FEATURE_RANGES_PATH} ({len(ranges) - 1} fields)")
    print(json.dumps(ranges, indent=2)[:2000])


def compute_risk_bands() -> None:
    """Quintiles of the served model's calibrated PD on test (2015) -
    the year its Platt calibrator was fit on. Reading test.parquet here
    is a serving asset, not an evaluation: no metric about the model's
    performance is produced, only PD cut points for the band labels the
    API reports. Never computed on the 2016 vintages used for the
    pre-registered decision rule."""
    frozen = load_frozen_config()
    test = pd.read_parquet(TEST_PARQUET)
    log_access(
        "build_assets.py:compute_risk_bands",
        len(test),
        "Serving asset, not evaluation: re-read test.parquet to find PD quintile cut points "
        "(the year the served model's calibrator was fit on) for the API's risk-band labels. "
        "No metric about the model's performance is computed or reported.",
    )
    pd_scores = score_lightgbm(test, frozen["lightgbm_chosen"])

    edges = np.percentile(pd_scores, PERCENTILES)

    bands = []
    for i, name in enumerate(BAND_NAMES):
        lo, hi = float(edges[i]), float(edges[i + 1])
        is_last = i == len(BAND_NAMES) - 1
        mask = (pd_scores >= lo) & (pd_scores <= hi if is_last else pd_scores < hi)
        n = int(mask.sum())
        default_rate = float(test["target"].values[mask].mean())
        bands.append({
            "band": name,
            "pd_min": round(lo, 6),
            "pd_max": round(hi, 6),
            "pd_max_inclusive": is_last,
            "test_n": n,
            "test_default_rate_pct": round(default_rate * 100, 2),
        })

    output = {
        "methodology": "Quintiles (20% each) of the served model's calibrated PD on test (2015), "
        "the year its calibrator was fit on. Never computed on the 2016 vintages used for the "
        "pre-registered decision rule.",
        "defined_on": "test (2015)",
        "source_model_sha256": _model_hash(frozen),
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
