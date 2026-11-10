"""never_flag__* rollout gating: the two gating mechanisms
(row-level, via tot_cur_bal; date-level, via a 2012-04-01 cutoff) must
null out the flag for pre-rollout rows instead of marking them "1"
(which would wrongly claim "never happened" for a field that simply
didn't exist in the data yet)."""

import numpy as np
import pandas as pd

from src.features.feature_engineering import compute_never_flags


def base_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "issue_d": pd.to_datetime(["2012-01-01"] * n),
        "tot_cur_bal": [np.nan] * n,
        "mths_since_recent_bc_dlq": [np.nan] * n,
        "mths_since_last_major_derog": [np.nan] * n,
        "mths_since_recent_inq": [np.nan] * n,
        "mths_since_recent_revol_delinq": [np.nan] * n,
        "mths_since_last_delinq": [np.nan] * n,
        "mths_since_last_record": [np.nan] * n,
    })


def test_row_level_gate_nulls_flag_when_gate_column_missing():
    """mths_since_recent_bc_dlq and mths_since_last_major_derog are
    gated on tot_cur_bal: if tot_cur_bal itself is missing (pre-rollout
    row), the flag must be NULL, not 1."""
    df = base_df(2)
    df.loc[0, "tot_cur_bal"] = np.nan  # pre-rollout: gate missing
    df.loc[1, "tot_cur_bal"] = 5000.0  # post-rollout: gate present, source still missing -> flag = 1
    flags = compute_never_flags(df)
    assert pd.isna(flags["never_flag__mths_since_recent_bc_dlq"].iloc[0])
    assert flags["never_flag__mths_since_recent_bc_dlq"].iloc[1] == 1.0
    assert pd.isna(flags["never_flag__mths_since_last_major_derog"].iloc[0])
    assert flags["never_flag__mths_since_last_major_derog"].iloc[1] == 1.0


def test_row_level_gate_flag_is_0_when_source_present():
    df = base_df(1)
    df.loc[0, "tot_cur_bal"] = 5000.0
    df.loc[0, "mths_since_recent_bc_dlq"] = 10.0  # source present
    flags = compute_never_flags(df)
    assert flags["never_flag__mths_since_recent_bc_dlq"].iloc[0] == 0.0


def test_date_gate_nulls_flag_before_cutoff():
    """mths_since_recent_inq and mths_since_recent_revol_delinq are
    gated on issue_d < 2012-04-01."""
    df = base_df(2)
    df.loc[0, "issue_d"] = pd.Timestamp("2012-02-01")  # before cutoff
    df.loc[1, "issue_d"] = pd.Timestamp("2012-05-01")  # after cutoff, source still missing -> flag = 1
    flags = compute_never_flags(df)
    assert pd.isna(flags["never_flag__mths_since_recent_inq"].iloc[0])
    assert flags["never_flag__mths_since_recent_inq"].iloc[1] == 1.0
    assert pd.isna(flags["never_flag__mths_since_recent_revol_delinq"].iloc[0])
    assert flags["never_flag__mths_since_recent_revol_delinq"].iloc[1] == 1.0


def test_date_gate_flag_is_0_when_source_present_after_cutoff():
    df = base_df(1)
    df.loc[0, "issue_d"] = pd.Timestamp("2012-05-01")
    df.loc[0, "mths_since_recent_inq"] = 2.0
    flags = compute_never_flags(df)
    assert flags["never_flag__mths_since_recent_inq"].iloc[0] == 0.0


def test_date_gate_boundary_is_inclusive_of_cutoff_date():
    df = base_df(1)
    df.loc[0, "issue_d"] = pd.Timestamp("2012-04-01")  # exactly the cutoff - not "before" it
    flags = compute_never_flags(df)
    # issue_d < cutoff is False when issue_d == cutoff, so the gate does NOT null the flag here
    assert flags["never_flag__mths_since_recent_inq"].iloc[0] == 1.0


def test_live_application_today_always_passes_both_gates():
    """A live API request always has issue_d defaulted to today and a
    populated tot_cur_bal in the normal case, so both gates should
    reduce to a plain 1/0 flag, never NULL, for any current application."""
    df = base_df(1)
    df.loc[0, "issue_d"] = pd.Timestamp.today()
    df.loc[0, "tot_cur_bal"] = 8000.0
    flags = compute_never_flags(df)
    for col in flags.columns:
        assert flags[col].iloc[0] in (0.0, 1.0), f"{col} should never be NULL for a current-dated application"
