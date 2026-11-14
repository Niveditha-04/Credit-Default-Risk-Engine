"""Data audits on the cleaned, typed LendingClub table (all years):
status by year/term, maturity (resolved share), and payment-timing audit.
"""

import json
from pathlib import Path

import duckdb

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")

RESOLVED_STATUSES = ("Fully Paid", "Charged Off", "Default")


def status_by_year(con: duckdb.DuckDBPyConnection) -> None:
    """Counts by issue year, term, loan_status, across all years in the file."""
    df = con.execute(
        f"""
        SELECT
            EXTRACT(year FROM issue_d) AS issue_year,
            term,
            loan_status,
            count(*) AS n
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchdf()
    df.to_csv("reports/metrics/status_by_year.csv", index=False)
    print(f"status_by_year.csv: {len(df)} rows")


def maturity_audit(con: duckdb.DuckDBPyConnection) -> None:
    """Share of loans resolved (Fully Paid/Charged Off/Default) vs still
    open, by issue year and term. Justifies the 2012-2015, 36-month choice:
    a loan needs its full term to elapse before we can call it resolved."""
    resolved_list = ", ".join(f"'{s}'" for s in RESOLVED_STATUSES)
    df = con.execute(
        f"""
        SELECT
            EXTRACT(year FROM issue_d) AS issue_year,
            term,
            count(*) AS n_total,
            sum(CASE WHEN loan_status IN ({resolved_list}) THEN 1 ELSE 0 END) AS n_resolved,
            round(
                100.0 * sum(CASE WHEN loan_status IN ({resolved_list}) THEN 1 ELSE 0 END) / count(*),
                2
            ) AS pct_resolved
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()
    df.to_csv("reports/metrics/maturity_audit.csv", index=False)
    print(f"maturity_audit.csv: {len(df)} rows")


def event_timing_audit(con: duckdb.DuckDBPyConnection) -> None:
    """Months from issue to last payment, by status, within the 36-month
    individual-application population across all years. Checks how many
    loans show payments after month 36 (late payments can stretch the
    term), and documents that last_pymnt_d only approximates default
    timing (it is the last payment seen, not necessarily the exact
    default event date)."""
    base_pop = f"""
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        WHERE term = 36 AND application_type = 'Individual'
    """

    months_expr = "date_diff('month', issue_d, last_pymnt_d)"

    summary_df = con.execute(
        f"""
        SELECT
            loan_status,
            count(*) AS n,
            sum(CASE WHEN last_pymnt_d IS NULL THEN 1 ELSE 0 END) AS n_no_last_pymnt_d,
            round(avg({months_expr}), 2) AS mean_months_to_last_pymnt,
            median({months_expr}) AS median_months_to_last_pymnt,
            min({months_expr}) AS min_months_to_last_pymnt,
            max({months_expr}) AS max_months_to_last_pymnt,
            sum(CASE WHEN {months_expr} > 36 THEN 1 ELSE 0 END) AS n_payments_after_month_36
        {base_pop}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    summary_df.to_csv("reports/metrics/event_timing_audit.csv", index=False)

    total_after_36 = con.execute(
        f"SELECT sum(CASE WHEN {months_expr} > 36 THEN 1 ELSE 0 END) {base_pop}"
    ).fetchone()[0]
    total_pop = con.execute(f"SELECT count(*) {base_pop}").fetchone()[0]

    summary = {
        "population": "term=36, application_type=Individual, all issue years",
        "total_rows": total_pop,
        "rows_with_payment_after_month_36": int(total_after_36),
        "pct_with_payment_after_month_36": round(100.0 * total_after_36 / total_pop, 2),
        "note": (
            "last_pymnt_d is the date of the last payment LendingClub recorded, "
            "not a recorded default event date. It only approximates when a "
            "charged-off loan actually stopped paying."
        ),
    }
    with open("reports/metrics/event_timing_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def eligibility_caps_by_year(con: duckdb.DuckDBPyConnection) -> None:
    """LendingClub's own origination-time eligibility caps, by issue
    year, for the 36-month individual population used throughout this
    project: FICO floor and DTI ceiling. Checks whether either cap moved
    over time, since a claim like "every loan has DTI <= X" is only as
    good as the years it was actually checked against."""
    df = con.execute(
        f"""
        SELECT
            EXTRACT(year FROM issue_d) AS issue_year,
            count(*) AS n,
            min(fico_range_low) AS min_fico_range_low,
            sum(CASE WHEN fico_range_low < 660 THEN 1 ELSE 0 END) AS n_fico_lt_660,
            max(dti) AS max_dti,
            sum(CASE WHEN dti > 35 THEN 1 ELSE 0 END) AS n_dti_gt_35,
            round(100.0 * sum(CASE WHEN dti > 35 THEN 1 ELSE 0 END) / count(*), 3) AS pct_dti_gt_35
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2018
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    df.to_csv("reports/metrics/eligibility_caps_by_year.csv", index=False)
    print("eligibility_caps_by_year.csv:")
    print(df.to_string(index=False))


def duplicate_id_check(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(
        f"""
        SELECT count(*) AS n_rows, count(DISTINCT id) AS n_distinct_ids
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        """
    ).fetchone()
    result = {"n_rows": row[0], "n_distinct_ids": row[1], "n_duplicate_ids": row[0] - row[1]}
    print(result)
    return result


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    status_by_year(con)
    maturity_audit(con)
    event_timing_audit(con)
    eligibility_caps_by_year(con)
    duplicate_id_check(con)


if __name__ == "__main__":
    main()
