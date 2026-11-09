"""Missing-value percentage by issue year, for every raw column, within the
2012-2015 individual/36-month modelling population. This is the evidence
behind the allow_list.py drop/keep decisions for columns that rolled out
partway through the data's history.
"""

from pathlib import Path

import duckdb

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
OUT_CSV = Path("reports/metrics/missing_by_year.csv")


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    all_cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{CLEANED_PARQUET.as_posix()}')"
        ).fetchall()
    ]

    base = f"""
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        WHERE term = 36
          AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2015
    """

    selects = [
        f'round(100.0 * sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) / count(*), 2) AS "{c}"'
        for c in all_cols
        if c != "issue_d"
    ]

    query = f"""
        SELECT EXTRACT(year FROM issue_d) AS issue_year,
        {', '.join(selects)}
        {base}
        GROUP BY 1
        ORDER BY 1
    """
    wide = con.execute(query).fetchdf()

    long = wide.melt(id_vars="issue_year", var_name="column", value_name="missing_pct")
    long = long.sort_values(["column", "issue_year"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(long)} rows, {len(all_cols) - 1} columns x {wide.shape[0]} years)")


if __name__ == "__main__":
    main()
