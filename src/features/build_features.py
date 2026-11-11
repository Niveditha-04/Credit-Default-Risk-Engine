"""Build the final modelling tables (train/validation/test/replay) from the
cleaned, typed LendingClub table: apply the allow-list, engineer features,
and assert no banned column ever reaches the output.

Engineered features (all computed from origination-time data only):
    fico_mean               = (fico_range_low + fico_range_high) / 2
    credit_history_months   = months between earliest_cr_line and issue_d
    loan_to_income          = loan_amnt / annual_inc
    revol_bal_to_income     = revol_bal / annual_inc
    never_flag__<source>    = 1 if <source> is missing, 0 if present, for the
                               6 mths_since_* columns.

log_annual_inc is not in the feature set: it is a strictly monotonic
transform of annual_inc, so it is an exact duplicate for any tree split
boundary and has byte-identical IV to annual_inc. fico_range_low and
fico_range_high are similarly left out of the feature set (see
allow_list.py) in favor of the single fico_mean representative.

Four of those six source columns (mths_since_recent_bc_dlq,
mths_since_recent_inq, mths_since_recent_revol_delinq,
mths_since_last_major_derog) are 100% missing in January 2012: the field
did not exist yet in LendingClub's schema, confirmed at month granularity
in reports/metrics/field_rollout.csv. For loans issued before the field
existed, a missing value does not mean "this never happened" - it means
"not recorded yet". The flag is NULL for those loans rather than 1.
Two different gates are used, because August 2012 (the first calendar
month with any non-null data) was itself a partial rollout month -
tot_cur_bal, which rolled out alongside the bureau-data refresh, is still
~84% missing in August and only ~1% missing by September:
  - mths_since_recent_bc_dlq, mths_since_last_major_derog: gated on
    tot_cur_bal IS NULL for that row (same bureau-data refresh, verified
    row-level: whenever tot_cur_bal is null, these two are always null
    too, in every month of 2012 - so this gate never nulls a flag that
    had real signal).
  - mths_since_recent_inq, mths_since_recent_revol_delinq: these rolled
    out earlier (first non-null March 2012), so gated on a date cutoff,
    issue_d < 2012-04-01.
The other two source columns (mths_since_last_delinq, mths_since_last_record)
were already non-null in January 2012, so their missingness is true MNAR
throughout and the flag stays a plain 1/0 for every row.
"""

import json
from pathlib import Path

import duckdb

from src.data.target import labelled_population_query, replay_population_query
from src.features.allow_list import ALLOW_LIST, check_completeness, feature_columns
from src.features.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    REQUIRED_RAW_COLUMNS,
    engineer_features,
)

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
PROCESSED_DIR = Path("data/processed")


def engineered_feature_names() -> list[str]:
    return list(ENGINEERED_FEATURE_NAMES)


ALWAYS_SELECTED_COLUMNS = {"id", "issue_d", "split", "target"}


def derivation_only_columns() -> list[str]:
    """Raw columns needed to compute engineered features but not
    themselves model features (e.g. fico_range_low/high, which only
    exist to build fico_mean) - selected in the SQL, dropped before the
    table is written.

    issue_d is deliberately excluded here even though it's in
    REQUIRED_RAW_COLUMNS: it's already selected via the hardcoded
    "id, issue_d, split, target" prefix in build_select_clause(), so
    including it again would select it twice. DuckDB silently renames the
    second copy to issue_d_1, which would then leak through as a bogus
    extra "feature" - the exact allow-list check in gbm.py's
    feature_list() is the guard that would catch this if it ever
    happened again."""
    raw_feats = set(feature_columns())
    return [c for c in REQUIRED_RAW_COLUMNS if c not in raw_feats and c not in ALWAYS_SELECTED_COLUMNS]


def build_select_clause() -> str:
    check_completeness()
    raw_feats = feature_columns()
    all_cols = sorted(set(raw_feats) | set(derivation_only_columns()))
    select = ", ".join(f'"{c}"' for c in all_cols)
    return f"id, issue_d, split, target, {select}"


def hard_excluded_columns() -> set[str]:
    """Columns that must never appear anywhere in the output: decision is
    drop or target_source. (keep_not_feature columns like id/issue_d are
    intentionally kept as metadata, not as model features.)"""
    return {c for c, d in ALLOW_LIST.items() if d.decision in ("drop", "target_source")}


def assert_no_banned_columns(columns_present: set[str]) -> None:
    leaked = hard_excluded_columns() & columns_present
    assert not leaked, f"Banned columns leaked into the feature table: {sorted(leaked)}"


NUMERIC_ENGINEERED_FEATURES = [
    "fico_mean",
    "credit_history_months",
    "loan_to_income",
    "revol_bal_to_income",
]


def assert_engineered_features_finite(con: duckdb.DuckDBPyConnection, parquet_path: str, split: str) -> None:
    checks = ", ".join(
        f'sum(CASE WHEN isinf("{c}") THEN 1 ELSE 0 END) AS "{c}"' for c in NUMERIC_ENGINEERED_FEATURES
    )
    row = con.execute(f"SELECT {checks} FROM read_parquet('{parquet_path}')").fetchone()
    bad = {c: n for c, n in zip(NUMERIC_ENGINEERED_FEATURES, row) if n > 0}
    assert not bad, f"{split}: non-finite (inf/-inf) values found in engineered features: {bad}"


def main() -> None:
    import pandas as pd

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    select_clause = build_select_clause()
    drop_cols = derivation_only_columns()

    labelled_query = f"SELECT {select_clause} FROM ({labelled_population_query()})"
    replay_query = f"SELECT {select_clause} FROM ({replay_population_query()})"

    # ORDER BY id on every query that actually gets materialized to parquet:
    # without it, row order is an unspecified side effect of the query plan,
    # and anything downstream that relies on "the first N rows" (a
    # signature-inference sample, a quick preview) would see a different
    # sample than intended whenever the query plan changes, even with the
    # same rows and values. id is unique per loan, so this is a total order.
    summary = {}
    for split, query in [
        ("train", f"{labelled_query} WHERE split = 'train' AND target IS NOT NULL ORDER BY id"),
        ("validation", f"{labelled_query} WHERE split = 'validation' AND target IS NOT NULL ORDER BY id"),
        ("test", f"{labelled_query} WHERE split = 'test' AND target IS NOT NULL ORDER BY id"),
        ("replay", f"{replay_query} ORDER BY id"),
    ]:
        out_path = PROCESSED_DIR / f"{split}.parquet"

        raw_df = con.execute(query).fetchdf()
        # engineer_features() is the SAME function serving imports (src/features/
        # feature_engineering.py) - no copied logic between training and serving.
        engineered = engineer_features(raw_df)
        final_df = pd.concat([raw_df.drop(columns=drop_cols), engineered], axis=1)
        final_df.to_parquet(out_path, index=False)

        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_path.as_posix()}')").fetchall()]
        assert_no_banned_columns(set(cols))
        assert_engineered_features_finite(con, out_path.as_posix(), split)

        n_rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        summary[split] = {"rows": n_rows, "columns": len(cols)}
        if split != "replay":
            default_rate = con.execute(
                f"SELECT round(100.0 * avg(target), 2) FROM read_parquet('{out_path.as_posix()}')"
            ).fetchone()[0]
            summary[split]["default_rate_pct"] = default_rate
        else:
            n_non_null_target = con.execute(
                f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}') WHERE target IS NOT NULL"
            ).fetchone()[0]
            assert n_non_null_target == 0, (
                f"replay target must be entirely NULL by design (no labels used), "
                f"found {n_non_null_target} non-null values"
            )
            summary[split]["target_entirely_null"] = True
        print(f"{split}: {n_rows} rows, {len(cols)} columns -> {out_path}")

    summary["n_raw_features"] = len(feature_columns())
    summary["n_engineered_features"] = len(engineered_feature_names())
    summary["n_hard_excluded_columns_defined"] = len(hard_excluded_columns())

    with open("reports/metrics/build_features_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
