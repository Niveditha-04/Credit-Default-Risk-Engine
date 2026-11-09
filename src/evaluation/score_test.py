"""THE ONE-TIME TOUCH OF TEST (2015).

Rules (per Nivi, before running this):
  - Load models ONLY from reports/metrics/frozen_model_config.json.
  - Log this access (timestamp, script, row count).
  - After this script runs and results are seen, no model, feature, or
    calibration changes. If a bug is found after viewing test, report it
    to Nivi first - do not silently fix and rerun.

This script scores test with the three frozen models (scoring functions
already dry-run and verified correct against validation in
score_models.py), joins the small set of additional fields needed for
economics/benchmark/stability/robustness analysis (all evaluation-only,
never model features), and saves everything to a single file. Every
downstream Phase 5 script reads from that saved file, not from test.parquet
again, so the actual read of test (2015) happens exactly once, here.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.evaluation.score_models import load_frozen_config, score_all
from src.evaluation.test_access_log import log_access

TEST_PARQUET = "data/processed/test.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
OUT_PATH = Path("reports/metrics/test_scored.parquet")


def main() -> None:
    frozen = load_frozen_config()

    test = pd.read_parquet(TEST_PARQUET)
    log_access("score_test.py", len(test), "Loaded test.parquet for the one-time Phase 5 scoring pass")

    scores = score_all(test, frozen)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    # Same population filter used to build test.parquet (term=36,
    # individual, issue year 2015) - evaluation-only fields never used as
    # model inputs: economics (funded_amnt, total_pymnt, recoveries,
    # collection_recovery_fee, installment), benchmark (grade, sub_grade,
    # int_rate), stability (addr_state), robustness (last_pymnt_d).
    extra = con.execute(
        f"""
        SELECT id, funded_amnt, total_pymnt, recoveries, collection_recovery_fee, installment,
               grade, sub_grade, int_rate, addr_state, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) = 2015
        """
    ).fetchdf()

    out = test[["id", "issue_d", "target", "loan_amnt"]].copy()
    out = pd.concat([out, scores], axis=1)
    out = out.merge(extra, on="id", how="left")
    assert out["funded_amnt"].isna().sum() == 0, "Some test ids did not match the cleaned table"
    assert len(out) == len(test), "Row count changed during join"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    log_access(
        "score_test.py", len(out),
        f"Saved scored test to {OUT_PATH} (3 model PDs + economics/benchmark/stability/robustness fields)",
    )

    print(f"Scored and saved {len(out)} test rows to {OUT_PATH}")
    print(out[["pd_lightgbm", "pd_xgboost", "pd_scorecard", "target"]].describe())


if __name__ == "__main__":
    main()
