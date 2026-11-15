"""Data audits on the cleaned, typed LendingClub table (all years):
status by year/term, maturity (resolved share), payment-timing, field
-rollout timing, and missing-value rate by issue year.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
FIELD_ROLLOUT_OUT_CSV = Path("reports/metrics/data/field_rollout.csv")
MISSING_BY_YEAR_OUT_CSV = Path("reports/metrics/data/missing_by_year.csv")

RESOLVED_STATUSES = ("Fully Paid", "Charged Off", "Default")

# Columns kept as features whose train-year-average missingness is driven by
# a 2012 collection rollout (identified in allow_list.py).
ROLLOUT_KEEP_FEATURE_COLUMNS = [
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "total_rev_hi_lim",
    "avg_cur_bal",
    "tot_coll_amt",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "acc_open_past_24mths",
    "bc_open_to_buy",
    "bc_util",
    "percent_bc_gt_75",
    "mths_since_recent_bc",
]

# never_flag source columns confirmed to show the same 100%-missing-in-
# Jan-2012 rollout signature (checked at month granularity).
ROLLOUT_NEVER_FLAG_SOURCE_COLUMNS = [
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "mths_since_last_major_derog",
]

ALL_ROLLOUT_COLUMNS = ROLLOUT_KEEP_FEATURE_COLUMNS + ROLLOUT_NEVER_FLAG_SOURCE_COLUMNS


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
    df.to_csv("reports/metrics/data/status_by_year.csv", index=False)
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
    df.to_csv("reports/metrics/data/maturity_audit.csv", index=False)
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
    Path("reports/extra").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv("reports/extra/event_timing_audit.csv", index=False)

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
    with open("reports/metrics/data/event_timing_audit_summary.json", "w") as f:
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
    df.to_csv("reports/metrics/data/eligibility_caps_by_year.csv", index=False)
    print("eligibility_caps_by_year.csv:")
    print(df.to_string(index=False))


def field_rollout(con: duckdb.DuckDBPyConnection) -> None:
    """For every column identified as having a 2012 data-collection
    rollout (100% missing in January 2012, meaning the field did not
    exist yet in LendingClub's schema), find the first issue month it
    has any non-null value. Computed across the full cleaned dataset
    (all years, all terms, all application types) since field
    availability is a LendingClub-wide schema decision, not specific
    to this project's modelling population."""
    rows = []
    for col in ALL_ROLLOUT_COLUMNS:
        first_month = con.execute(
            f"""
            SELECT strftime(min(issue_d), '%Y-%m')
            FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
            WHERE "{col}" IS NOT NULL
            """
        ).fetchone()[0]

        jan_2012_missing_pct = con.execute(
            f"""
            SELECT round(100.0 * sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) / count(*), 2)
            FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
            WHERE term = 36 AND application_type = 'Individual'
              AND EXTRACT(year FROM issue_d) = 2012 AND EXTRACT(month FROM issue_d) = 1
            """
        ).fetchone()[0]

        aug_2012_missing_pct = con.execute(
            f"""
            SELECT round(100.0 * sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) / count(*), 2)
            FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
            WHERE term = 36 AND application_type = 'Individual'
              AND EXTRACT(year FROM issue_d) = 2012 AND EXTRACT(month FROM issue_d) = 8
            """
        ).fetchone()[0]

        rows.append(
            {
                "column": col,
                "first_issue_month_non_null": first_month,
                "jan_2012_missing_pct": jan_2012_missing_pct,
                "aug_2012_missing_pct": aug_2012_missing_pct,
                "used_as_never_flag_source": col in ROLLOUT_NEVER_FLAG_SOURCE_COLUMNS,
            }
        )

    df = pd.DataFrame(rows).sort_values("first_issue_month_non_null")
    FIELD_ROLLOUT_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIELD_ROLLOUT_OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {FIELD_ROLLOUT_OUT_CSV} ({len(df)} rows)")


def missing_by_year(con: duckdb.DuckDBPyConnection) -> None:
    """Missing-value percentage by issue year, for every raw column,
    within the 2012-2015 individual/36-month modelling population.
    This is the evidence behind the allow_list.py drop/keep decisions
    for columns that rolled out partway through the data's history."""
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
        {", ".join(selects)}
        {base}
        GROUP BY 1
        ORDER BY 1
    """
    wide = con.execute(query).fetchdf()

    long = wide.melt(id_vars="issue_year", var_name="column", value_name="missing_pct")
    long = long.sort_values(["column", "issue_year"]).reset_index(drop=True)

    MISSING_BY_YEAR_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(MISSING_BY_YEAR_OUT_CSV, index=False)
    print(f"Wrote {MISSING_BY_YEAR_OUT_CSV} ({len(long)} rows, {len(all_cols) - 1} columns x {wide.shape[0]} years)")


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
    Path("reports/metrics/data").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    missing_by_year(con)
    field_rollout(con)
    status_by_year(con)
    maturity_audit(con)
    event_timing_audit(con)
    eligibility_caps_by_year(con)
    duplicate_id_check(con)


if __name__ == "__main__":
    main()
