"""Parity check: 1,000 seeded test-set rows, sent ONE AT A TIME
through the live FastAPI app (TestClient - no network, but the full
request/response/validation path), must return the exact same PD as the
frozen champion's batch scoring (src.evaluation.score_models), within
1e-9. Includes rows with missing optional fields and every purpose
category in the sample by construction, not by luck.

This does not touch the test set's labels (target) or produce any metric
about the model's performance: it is a plumbing check (does the API
reproduce the offline score), not a test-set evaluation.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import ScoreRequest
from src.evaluation.score_models import load_frozen_config, score_lightgbm

METRICS_DIR = Path("reports/metrics")
TEST_PARQUET = "data/processed/test.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
SEED = 42
N_SAMPLE = 1000
TOLERANCE = 1e-9


def build_sample() -> pd.DataFrame:
    test = pd.read_parquet(TEST_PARQUET)
    con = duckdb.connect()
    cleaned = con.execute(
        f"SELECT id, fico_range_low, fico_range_high, earliest_cr_line "
        f"FROM read_parquet('{CLEANED_PARQUET}') WHERE id IN "
        f"(SELECT id FROM read_parquet('{TEST_PARQUET}'))"
    ).fetchdf()
    merged = test.merge(cleaned, on="id", how="inner")
    assert len(merged) == len(test), f"join dropped rows: {len(test)} -> {len(merged)}"

    # One row per purpose category first, so every category is guaranteed
    # to appear even though the categories are not evenly sized. (A manual
    # loop, not groupby().apply() - pandas can silently drop the grouping
    # column from the result when the applied function returns a frame
    # shaped just like its input, which it does here.)
    one_per_purpose = pd.concat(
        [merged[merged["purpose"] == p].sample(n=1, random_state=SEED) for p in sorted(merged["purpose"].unique())]
    )

    # A block of rows that have at least one missing optional field.
    optional_cols = [c for c in merged.columns if c not in (
        "id", "issue_d", "split", "target", "loan_amnt", "annual_inc", "dti",
        "fico_range_low", "fico_range_high", "earliest_cr_line",
        "home_ownership", "purpose", "verification_status",
    )]
    has_missing = merged[optional_cols].isna().any(axis=1)
    missing_rows = merged[has_missing].sample(
        n=min(100, int(has_missing.sum())), random_state=SEED
    )

    forced = pd.concat([one_per_purpose, missing_rows]).drop_duplicates(subset="id")
    remaining_needed = N_SAMPLE - len(forced)
    pool = merged[~merged["id"].isin(forced["id"])]
    filler = pool.sample(n=remaining_needed, random_state=SEED)

    sample = pd.concat([forced, filler]).drop_duplicates(subset="id").reset_index(drop=True)
    assert len(sample) == N_SAMPLE, f"expected {N_SAMPLE} rows, got {len(sample)}"
    return sample


def row_to_payload(row: pd.Series, schema_fields: list[str]) -> dict:
    payload = {}
    for f in schema_fields:
        v = row[f]
        if v is None or pd.isna(v):
            payload[f] = None
        elif f in ("earliest_cr_line", "issue_d"):
            payload[f] = str(pd.Timestamp(v).date())
        elif f in ("home_ownership", "purpose", "verification_status"):
            payload[f] = str(v)
        else:
            payload[f] = float(v)
    return payload


def main() -> None:
    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]

    sample = build_sample()
    batch_pd = score_lightgbm(sample, cfg)

    client = TestClient(app)
    schema_fields = list(ScoreRequest.model_fields.keys())

    diffs = []
    mismatches = []
    missing_value_rows_checked = 0
    purposes_seen = set()
    optional_cols = [c for c in sample.columns if c not in (
        "id", "issue_d", "split", "target", "loan_amnt", "annual_inc", "dti",
        "fico_range_low", "fico_range_high", "earliest_cr_line",
        "home_ownership", "purpose", "verification_status",
    )]

    for i in range(len(sample)):
        row = sample.iloc[i]
        payload = row_to_payload(row, schema_fields)
        purposes_seen.add(payload["purpose"])
        if any(payload[c] is None for c in optional_cols if c in payload):
            missing_value_rows_checked += 1

        resp = client.post("/score", json=payload)
        assert resp.status_code == 200, f"row {i} (id={row['id']}) failed: {resp.status_code} {resp.text}"
        api_pd = resp.json()["calibrated_pd"]

        diff = abs(api_pd - batch_pd[i])
        diffs.append(diff)
        if diff > TOLERANCE:
            mismatches.append({"id": int(row["id"]), "batch_pd": float(batch_pd[i]), "api_pd": api_pd, "diff": diff})

    diffs = np.array(diffs)
    result = {
        "n_rows_tested": len(sample),
        "tolerance": TOLERANCE,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "n_exact_within_tolerance": int((diffs <= TOLERANCE).sum()),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "purpose_categories_covered": sorted(purposes_seen),
        "n_purpose_categories_covered": len(purposes_seen),
        "n_rows_with_missing_optional_field": missing_value_rows_checked,
        "sent_one_at_a_time": True,
        "verdict": "PASS" if len(mismatches) == 0 else "FAIL",
    }

    with open(METRICS_DIR / "api_parity_test.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    assert len(mismatches) == 0, f"{len(mismatches)} rows exceeded tolerance {TOLERANCE}"


if __name__ == "__main__":
    main()
