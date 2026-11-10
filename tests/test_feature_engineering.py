"""Engineered feature functions against a tiny synthetic DataFrame - no
real data needed, pure functions of src/features/feature_engineering.py."""

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    compute_credit_history_months,
    compute_fico_mean,
    compute_loan_to_income,
    compute_never_flags,
    compute_revol_bal_to_income,
    engineer_features,
)


def synthetic_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "fico_range_low": [680.0, 700.0, 660.0],
        "fico_range_high": [684.0, 704.0, 664.0],
        "earliest_cr_line": pd.to_datetime(["2005-03-01", "2010-01-01", "2000-06-15"]),
        "issue_d": pd.to_datetime(["2013-06-01", "2013-06-01", "2013-06-01"]),
        "loan_amnt": [10000.0, 20000.0, 5000.0],
        "annual_inc": [50000.0, 0.0, 40000.0],  # row 1 has zero income - division edge case
        "revol_bal": [2000.0, 1000.0, 500.0],
        "tot_cur_bal": [10000.0, np.nan, 5000.0],
        "mths_since_last_delinq": [np.nan, 12.0, np.nan],
        "mths_since_last_major_derog": [np.nan, np.nan, 24.0],
        "mths_since_last_record": [np.nan, np.nan, np.nan],
        "mths_since_recent_bc_dlq": [np.nan, np.nan, np.nan],
        "mths_since_recent_revol_delinq": [np.nan, np.nan, np.nan],
        "mths_since_recent_inq": [3.0, np.nan, np.nan],
    })


def test_fico_mean_is_average_of_low_and_high():
    df = synthetic_raw_df()
    result = compute_fico_mean(df)
    assert list(result) == [682.0, 702.0, 662.0]


def test_credit_history_months_matches_year_month_difference():
    df = synthetic_raw_df()
    result = compute_credit_history_months(df)
    # 2013-06 minus 2005-03 = (2013-2005)*12 + (6-3) = 96 + 3 = 99
    assert result.iloc[0] == 99
    # 2013-06 minus 2010-01 = (3)*12 + 5 = 41
    assert result.iloc[1] == 41


def test_loan_to_income_handles_zero_income_without_dividing_by_zero():
    df = synthetic_raw_df()
    result = compute_loan_to_income(df)
    assert result.iloc[0] == 10000.0 / 50000.0
    assert pd.isna(result.iloc[1]), "zero annual_inc must produce NaN, not inf or a ZeroDivisionError"
    assert np.isfinite(result.iloc[0]) and np.isfinite(result.iloc[2])


def test_revol_bal_to_income_handles_zero_income():
    df = synthetic_raw_df()
    result = compute_revol_bal_to_income(df)
    assert pd.isna(result.iloc[1])
    assert np.isfinite(result.iloc[0]) and np.isfinite(result.iloc[2])


def test_engineered_numeric_features_are_all_finite_or_nan_never_inf():
    """Finite-values check: no engineered column may contain +-inf, even
    with a zero-income row in the input."""
    df = synthetic_raw_df()
    out = engineer_features(df)
    numeric_cols = [c for c in out.columns if out[c].dtype.kind == "f"]
    for col in numeric_cols:
        values = out[col].dropna().to_numpy()
        assert np.all(np.isfinite(values)), f"{col} contains a non-finite value (inf or -inf)"


def test_engineer_features_returns_exactly_the_expected_columns():
    df = synthetic_raw_df()
    out = engineer_features(df)
    assert list(out.columns) == ENGINEERED_FEATURE_NAMES
    assert len(out) == len(df)


def test_never_flag_plain_mnar_column_is_1_0_regardless_of_date():
    """mths_since_last_delinq and mths_since_last_record are not
    row/date-gated - missing means missing, always."""
    df = synthetic_raw_df()
    flags = compute_never_flags(df)
    assert list(flags["never_flag__mths_since_last_delinq"]) == [1.0, 0.0, 1.0]
    assert list(flags["never_flag__mths_since_last_record"]) == [1.0, 1.0, 1.0]
