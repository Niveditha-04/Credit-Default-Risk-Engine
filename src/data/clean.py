"""Remove footer rows and type every column of the raw LendingClub file.

Input: data/interim/accepted_2007_to_2018Q4_raw.parquet (all columns text).
Output: data/interim/accepted_2007_to_2018Q4_cleaned.parquet (same 151
columns, each cast to its natural type, footer rows removed).
"""

import json
from pathlib import Path

import duckdb

from src.data.columns import (
    ALL_GROUPED_COLUMNS,
    DATE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TEXT_COLUMNS,
)

RAW_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_raw.parquet")
OUT_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
OUT_METRICS = Path("reports/metrics/phase2_cleaning.json")


def check_column_coverage(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Every raw column must appear in exactly one typing group. Returns the
    raw column list (so callers don't re-query it)."""
    raw_cols = [
        r[0]
        for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{RAW_PARQUET.as_posix()}')").fetchall()
    ]
    raw_set = set(raw_cols)
    grouped_set = set(ALL_GROUPED_COLUMNS)

    missing_from_groups = raw_set - grouped_set
    extra_in_groups = grouped_set - raw_set
    assert not missing_from_groups, f"Raw columns not assigned a type group: {missing_from_groups}"
    assert not extra_in_groups, f"Typing groups reference columns not in raw file: {extra_in_groups}"
    assert len(raw_cols) == len(raw_set), "Raw file has duplicate column names"
    assert len(ALL_GROUPED_COLUMNS) == len(set(ALL_GROUPED_COLUMNS)), "A column is listed in two type groups"
    return raw_cols


def build_select_expr(col: str) -> str:
    """One SELECT expression per raw column, per its typing group."""
    if col == ID_COLUMN:
        return f'TRY_CAST("{col}" AS BIGINT) AS "{col}"'
    if col in DATE_COLUMNS:
        return f"TRY_STRPTIME(\"{col}\", '%b-%Y')::DATE AS \"{col}\""
    if col == "term":
        return f'TRY_CAST(regexp_extract(trim("{col}"), \'(\\d+)\') AS INTEGER) AS "{col}"'
    if col == "emp_length":
        return f"""
        CASE
            WHEN "{col}" IS NULL THEN NULL
            WHEN "{col}" = '< 1 year' THEN 0
            WHEN "{col}" = '10+ years' THEN 10
            ELSE TRY_CAST(regexp_extract("{col}", '(\\d+)') AS INTEGER)
        END AS "{col}"
        """
    if col in ("int_rate", "revol_util"):
        # Defensive strip: this Kaggle copy has no "%" sign, but the
        # original LendingClub export does, so handle both.
        return f'TRY_CAST(replace("{col}", \'%\', \'\') AS DOUBLE) AS "{col}"'
    if col == "dti":
        # A debt-to-income ratio cannot be negative. 2 rows in the raw file
        # have dti = -1 (likely a sentinel), 1 of which is in our 2012-2018
        # term=36/individual population (logged in phase2_cleaning.json).
        # dti > 100 is treated as impossible too: 2,561 rows in the raw
        # file, dominated by a dti=999 sentinel cluster (135 rows, by far
        # the single most common value above 100) - none fall in our
        # 2012-2018 term=36/individual population, but nulled defensively
        # since this cleaning runs on the full table (including replay).
        return (
            f'CASE WHEN TRY_CAST("{col}" AS DOUBLE) < 0 OR TRY_CAST("{col}" AS DOUBLE) > 100 '
            f'THEN NULL ELSE TRY_CAST("{col}" AS DOUBLE) END AS "{col}"'
        )
    if col in NUMERIC_COLUMNS:
        return f'TRY_CAST("{col}" AS DOUBLE) AS "{col}"'
    if col in TEXT_COLUMNS:
        return f'"{col}"'
    raise ValueError(f"Column {col} not in any typing group")


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    raw_cols = check_column_coverage(con)
    print(f"Column coverage check passed: all {len(raw_cols)} raw columns assigned a type.")

    n_raw_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{RAW_PARQUET.as_posix()}')"
    ).fetchone()[0]
    n_footer = con.execute(
        f"SELECT count(*) FROM read_parquet('{RAW_PARQUET.as_posix()}') WHERE issue_d IS NULL"
    ).fetchone()[0]

    n_negative_dti = con.execute(
        f"""
        SELECT count(*) FROM read_parquet('{RAW_PARQUET.as_posix()}')
        WHERE issue_d IS NOT NULL AND TRY_CAST(dti AS DOUBLE) < 0
        """
    ).fetchone()[0]
    n_dti_over_100 = con.execute(
        f"""
        SELECT count(*) FROM read_parquet('{RAW_PARQUET.as_posix()}')
        WHERE issue_d IS NOT NULL AND TRY_CAST(dti AS DOUBLE) > 100
        """
    ).fetchone()[0]

    select_clause = ",\n    ".join(build_select_expr(c) for c in raw_cols)
    query = f"""
        SELECT
        {select_clause}
        FROM read_parquet('{RAW_PARQUET.as_posix()}')
        WHERE issue_d IS NOT NULL
        ORDER BY id
    """

    con.execute(f"COPY ({query}) TO '{OUT_PARQUET.as_posix()}' (FORMAT PARQUET)")

    n_cleaned = con.execute(
        f"SELECT count(*) FROM read_parquet('{OUT_PARQUET.as_posix()}')"
    ).fetchone()[0]
    assert n_cleaned == n_raw_total - n_footer, (
        f"Row count after footer removal is wrong: {n_cleaned} != {n_raw_total} - {n_footer}"
    )

    n_distinct_ids = con.execute(
        f"SELECT count(DISTINCT id) FROM read_parquet('{OUT_PARQUET.as_posix()}')"
    ).fetchone()[0]
    n_null_ids = con.execute(
        f"SELECT count(*) FROM read_parquet('{OUT_PARQUET.as_posix()}') WHERE id IS NULL"
    ).fetchone()[0]

    # Sanity check on a known-good conversion: term parses cleanly to 36/60 only.
    term_values = con.execute(
        f"SELECT DISTINCT term FROM read_parquet('{OUT_PARQUET.as_posix()}') ORDER BY 1"
    ).fetchall()

    metrics = {
        "raw_row_count": n_raw_total,
        "footer_rows_removed": n_footer,
        "cleaned_row_count": n_cleaned,
        "distinct_ids": n_distinct_ids,
        "null_ids": n_null_ids,
        "duplicate_ids": n_cleaned - n_distinct_ids,
        "term_distinct_values_after_parsing": [t[0] for t in term_values],
        "n_columns": len(raw_cols),
        "negative_dti_rows_set_to_null": n_negative_dti,
        "dti_over_100_rows_set_to_null": n_dti_over_100,
    }

    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved cleaned parquet to {OUT_PARQUET}")
    print(f"Saved metrics to {OUT_METRICS}")


if __name__ == "__main__":
    main()
