"""Training-data ranges for every API input field, used at request time to
flag (not reject) values outside what the model was trained on. Computed
from TRAIN only - the same split discipline as everything else.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.data.target import labelled_population_query
from src.features.allow_list import feature_columns
from src.features.feature_engineering import REQUIRED_RAW_COLUMNS

OUT_PATH = Path("src/api/feature_ranges.json")
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"

CATEGORICAL_INPUT_FIELDS = ["home_ownership", "purpose", "verification_status"]


def main() -> None:
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

    with open(OUT_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(ranges)} fields)")
    print(json.dumps(ranges, indent=2)[:2000])


if __name__ == "__main__":
    main()
