"""For every column identified as having a 2012 data-collection rollout
(100% missing in January 2012, meaning the field did not exist yet in
LendingClub's schema), find the first issue month it has any non-null
value. Computed across the full cleaned dataset (all years, all terms,
all application types) since field availability is a LendingClub-wide
schema decision, not specific to our modelling population.
"""

from pathlib import Path

import duckdb

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")
OUT_CSV = Path("reports/metrics/field_rollout.csv")

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


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

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

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("first_issue_month_non_null")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
