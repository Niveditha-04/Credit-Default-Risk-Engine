"""Build the final modelling tables (train/validation/test/replay) from the
cleaned, typed LendingClub table: apply the allow-list, engineer features,
and assert no banned column ever reaches the output.

Engineered features (all computed from origination-time data only):
    fico_mean               = (fico_range_low + fico_range_high) / 2
    credit_history_months   = months between earliest_cr_line and issue_d
    log_annual_inc          = log1p(annual_inc)
    loan_to_income          = loan_amnt / annual_inc
    revol_bal_to_income     = revol_bal / annual_inc
    never_flag__<source>    = 1 if <source> is missing, 0 if present, for the
                               6 mths_since_* columns.

Four of those six source columns (mths_since_recent_bc_dlq,
mths_since_recent_inq, mths_since_recent_revol_delinq,
mths_since_last_major_derog) are 100% missing in January 2012: the field
did not exist yet in LendingClub's schema, confirmed at month granularity
in reports/metrics/field_rollout.csv. For loans issued before Aug 2012
(the first month any of them has a non-null value), a missing value does
not mean "this never happened" - it means "not recorded yet". The flag is
NULL for those loans rather than 1. The other two source columns
(mths_since_last_delinq, mths_since_last_record) were already non-null in
January 2012, so their missingness is true MNAR throughout and the flag
stays a plain 1/0 for every row.
"""

import json
from pathlib import Path

import duckdb

from src.data.target import labelled_population_query, replay_population_query
from src.features.allow_list import ALLOW_LIST, check_completeness, feature_columns

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
PROCESSED_DIR = Path("data/processed")

NEVER_FLAG_SOURCE_COLUMNS = [
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_inq",
]

# Source columns that were 100% missing in January 2012 (field not yet
# collected), verified at month granularity in field_rollout.py. For loans
# issued before this cutoff, "missing" is unknown, not "never happened".
ROLLOUT_AFFECTED_FLAG_SOURCES = {
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "mths_since_last_major_derog",
}
ROLLOUT_CUTOFF_DATE = "2012-08-01"  # first month any of the above has a non-null value


def engineered_feature_exprs() -> list[str]:
    exprs = [
        "(fico_range_low + fico_range_high) / 2.0 AS fico_mean",
        "date_diff('month', earliest_cr_line, issue_d) AS credit_history_months",
        "log(1 + annual_inc) AS log_annual_inc",
        "CASE WHEN annual_inc > 0 THEN loan_amnt / annual_inc ELSE NULL END AS loan_to_income",
        "CASE WHEN annual_inc > 0 THEN revol_bal / annual_inc ELSE NULL END AS revol_bal_to_income",
    ]
    for src in NEVER_FLAG_SOURCE_COLUMNS:
        flag_name = f"never_flag__{src}"
        if src in ROLLOUT_AFFECTED_FLAG_SOURCES:
            expr = (
                f"CASE WHEN issue_d < DATE '{ROLLOUT_CUTOFF_DATE}' THEN NULL "
                f'WHEN "{src}" IS NULL THEN 1 ELSE 0 END AS "{flag_name}"'
            )
        else:
            expr = f'CASE WHEN "{src}" IS NULL THEN 1 ELSE 0 END AS "{flag_name}"'
        exprs.append(expr)
    return exprs


def engineered_feature_names() -> list[str]:
    names = ["fico_mean", "credit_history_months", "log_annual_inc", "loan_to_income", "revol_bal_to_income"]
    names += [f"never_flag__{src}" for src in NEVER_FLAG_SOURCE_COLUMNS]
    return names


def build_select_clause() -> str:
    check_completeness()
    raw_feats = feature_columns()
    raw_select = ", ".join(f'"{c}"' for c in raw_feats)
    eng_select = ", ".join(engineered_feature_exprs())
    return f"id, issue_d, split, target, {raw_select}, {eng_select}"


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
    "log_annual_inc",
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    select_clause = build_select_clause()

    labelled_query = f"SELECT {select_clause} FROM ({labelled_population_query()})"
    replay_query = f"SELECT {select_clause} FROM ({replay_population_query()})"

    summary = {}
    for split, query in [
        ("train", f"{labelled_query} WHERE split = 'train' AND target IS NOT NULL"),
        ("validation", f"{labelled_query} WHERE split = 'validation' AND target IS NOT NULL"),
        ("test", f"{labelled_query} WHERE split = 'test' AND target IS NOT NULL"),
        ("replay", replay_query),
    ]:
        out_path = PROCESSED_DIR / f"{split}.parquet"
        con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")

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
