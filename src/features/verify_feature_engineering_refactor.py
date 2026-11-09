"""Proves src/features/feature_engineering.py's pure-Python functions
reproduce the original inline-SQL engineered features exactly, row for
row, on every split - BEFORE build_features.py is switched over to call
them. Run once as a one-time migration check; kept so the proof is
reproducible, not just asserted in a commit message.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.target import labelled_population_query, replay_population_query
from src.features.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    REQUIRED_RAW_COLUMNS,
    engineer_features,
)

METRICS_DIR = Path("reports/metrics")


def compare_split(con, split_name: str, query: str) -> dict:
    raw = con.execute(query).fetchdf()
    old = pd.read_parquet(f"data/processed/{split_name}.parquet")
    old = old.set_index("id").loc[raw["id"]].reset_index()
    new_eng = engineer_features(raw)

    mismatches = {}
    for col in ENGINEERED_FEATURE_NAMES:
        old_vals = old[col].values.astype(float)
        new_vals = new_eng[col].values.astype(float)
        both_nan = np.isnan(old_vals) & np.isnan(new_vals)
        close = np.isclose(old_vals, new_vals, equal_nan=False, atol=1e-9) | both_nan
        n_mismatch = int((~close).sum())
        if n_mismatch > 0:
            mismatches[col] = n_mismatch
    return {"n_rows": len(raw), "mismatches": mismatches}


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    results = {}
    for split_name, filt in [
        ("train", "split = 'train' AND target IS NOT NULL"),
        ("validation", "split = 'validation' AND target IS NOT NULL"),
        ("test", "split = 'test' AND target IS NOT NULL"),
    ]:
        cols = ", ".join(f'"{c}"' for c in dict.fromkeys(["id"] + REQUIRED_RAW_COLUMNS))
        query = f"SELECT {cols} FROM ({labelled_population_query()}) WHERE {filt}"
        results[split_name] = compare_split(con, split_name, query)
        print(split_name, results[split_name])

    cols = ", ".join(f'"{c}"' for c in dict.fromkeys(["id"] + REQUIRED_RAW_COLUMNS))
    query = f"SELECT {cols} FROM ({replay_population_query()})"
    results["replay"] = compare_split(con, "replay", query)
    print("replay", results["replay"])

    all_clean = all(not r["mismatches"] for r in results.values())
    assert all_clean, f"Feature engineering refactor does not match original SQL output: {results}"

    with open(METRICS_DIR / "feature_engineering_refactor_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll splits match exactly. Wrote {METRICS_DIR / 'feature_engineering_refactor_verification.json'}")


if __name__ == "__main__":
    main()
