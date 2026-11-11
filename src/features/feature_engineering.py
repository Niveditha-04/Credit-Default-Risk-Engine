"""Engineered-feature computation, imported verbatim by BOTH training
(build_features.py) and serving (src/api/main.py) - one function per
feature, never a second hand-written copy, so the two paths cannot
silently drift apart.

This was originally inline DuckDB SQL in build_features.py. It has been
verified to reproduce that SQL's output exactly, row for row, on every
split (see reports/metrics/feature_engineering_refactor_verification.json)
before build_features.py was switched over to call these functions.

Required raw input columns: fico_range_low, fico_range_high, earliest_cr_line,
issue_d, loan_amnt, annual_inc, revol_bal, tot_cur_bal, and the 6
mths_since_* columns.

Four of the 6 never_flag__* source columns (mths_since_recent_bc_dlq,
mths_since_recent_inq, mths_since_recent_revol_delinq,
mths_since_last_major_derog) are gated because they are 100% missing in
LendingClub's schema before a certain point in 2012 (see
reports/metrics/field_rollout.csv) - a missing value before that point
means "not recorded yet", not "never happened". This gating is harmless
for a live application scored today:
issue_d will always be well after the 2012 cutoffs and tot_cur_bal will
always be populated, so the gate always evaluates to "known" and the flag
reduces to a plain 1/0 on whether the source field itself is missing.
"""

import numpy as np
import pandas as pd

NEVER_FLAG_SOURCE_COLUMNS = [
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_inq",
]

ROW_LEVEL_GATED_FLAG_SOURCES = {
    "mths_since_recent_bc_dlq": "tot_cur_bal",
    "mths_since_last_major_derog": "tot_cur_bal",
}

DATE_GATED_FLAG_SOURCES = {
    "mths_since_recent_inq": "2012-04-01",
    "mths_since_recent_revol_delinq": "2012-04-01",
}

ENGINEERED_NUMERIC_NAMES = ["fico_mean", "credit_history_months", "loan_to_income", "revol_bal_to_income"]
ENGINEERED_FLAG_NAMES = [f"never_flag__{src}" for src in NEVER_FLAG_SOURCE_COLUMNS]
ENGINEERED_FEATURE_NAMES = ENGINEERED_NUMERIC_NAMES + ENGINEERED_FLAG_NAMES

# Raw columns engineer_features() reads, beyond the ones it passes through
# unchanged as model features.
REQUIRED_RAW_COLUMNS = [
    "fico_range_low", "fico_range_high", "earliest_cr_line", "issue_d",
    "loan_amnt", "annual_inc", "revol_bal", "tot_cur_bal",
] + NEVER_FLAG_SOURCE_COLUMNS


def compute_fico_mean(df: pd.DataFrame) -> pd.Series:
    return (df["fico_range_low"] + df["fico_range_high"]) / 2.0


def compute_credit_history_months(df: pd.DataFrame) -> pd.Series:
    """Matches DuckDB's date_diff('month', earliest_cr_line, issue_d):
    (year_b - year_a)*12 + (month_b - month_a), ignoring day-of-month
    entirely (verified against DuckDB directly before this was written)."""
    issue = pd.to_datetime(df["issue_d"])
    earliest = pd.to_datetime(df["earliest_cr_line"])
    return (issue.dt.year - earliest.dt.year) * 12 + (issue.dt.month - earliest.dt.month)


def compute_loan_to_income(df: pd.DataFrame) -> pd.Series:
    annual_inc = df["annual_inc"]
    return pd.Series(
        np.where(annual_inc > 0, df["loan_amnt"] / annual_inc, np.nan), index=df.index
    )


def compute_revol_bal_to_income(df: pd.DataFrame) -> pd.Series:
    annual_inc = df["annual_inc"]
    return pd.Series(
        np.where(annual_inc > 0, df["revol_bal"] / annual_inc, np.nan), index=df.index
    )


def compute_never_flags(df: pd.DataFrame) -> pd.DataFrame:
    issue_d = pd.to_datetime(df["issue_d"])
    out = {}
    for src in NEVER_FLAG_SOURCE_COLUMNS:
        flag_name = f"never_flag__{src}"
        src_missing = df[src].isna()
        if src in ROW_LEVEL_GATED_FLAG_SOURCES:
            gate_col = ROW_LEVEL_GATED_FLAG_SOURCES[src]
            gate_missing = df[gate_col].isna()
            flag = np.where(gate_missing, np.nan, np.where(src_missing, 1.0, 0.0))
        elif src in DATE_GATED_FLAG_SOURCES:
            cutoff = pd.Timestamp(DATE_GATED_FLAG_SOURCES[src])
            before_cutoff = issue_d < cutoff
            flag = np.where(before_cutoff, np.nan, np.where(src_missing, 1.0, 0.0))
        else:
            flag = np.where(src_missing, 1.0, 0.0)
        out[flag_name] = flag
    return pd.DataFrame(out, index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with exactly the 10 engineered feature columns,
    same row index as df. Does not touch or return any other column."""
    out = pd.DataFrame(index=df.index)
    out["fico_mean"] = compute_fico_mean(df)
    out["credit_history_months"] = compute_credit_history_months(df)
    out["loan_to_income"] = compute_loan_to_income(df)
    out["revol_bal_to_income"] = compute_revol_bal_to_income(df)
    flags = compute_never_flags(df)
    return pd.concat([out, flags], axis=1)
