"""src/features/build_features.py and src/features/feature_engineering.py:
pure helpers, an end-to-end main() run against a tiny synthetic
cleaned-parquet fixture, engineered-feature functions, and the
never_flag__* rollout gating - no real data needed anywhere in this file."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data import target as target_mod
from src.features import build_features as bf_mod
from src.features.allow_list import feature_columns
from src.features.build_features import (
    ALWAYS_SELECTED_COLUMNS,
    assert_no_banned_columns,
    build_select_clause,
    derivation_only_columns,
    engineered_feature_names,
    hard_excluded_columns,
)
from src.features.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    compute_credit_history_months,
    compute_fico_mean,
    compute_loan_to_income,
    compute_never_flags,
    compute_revol_bal_to_income,
    engineer_features,
)

# ---------------------------------------------------------------------
# build_features.py pure helpers
# ---------------------------------------------------------------------


def test_engineered_feature_names_has_the_10_expected_names():
    names = engineered_feature_names()
    assert len(names) == 10
    assert "fico_mean" in names
    assert "credit_history_months" in names


def test_derivation_only_columns_excludes_always_selected_columns():
    cols = derivation_only_columns()
    assert ALWAYS_SELECTED_COLUMNS.isdisjoint(cols)


def test_derivation_only_columns_includes_fico_range_fields():
    cols = derivation_only_columns()
    assert "fico_range_low" in cols
    assert "fico_range_high" in cols
    assert "earliest_cr_line" in cols


def test_build_select_clause_includes_id_issue_d_split_target_prefix():
    clause = build_select_clause()
    assert clause.startswith("id, issue_d, split, target, ")


def test_build_select_clause_has_no_duplicate_columns():
    clause = build_select_clause()
    cols = [c.strip().strip('"') for c in clause.split(",")]
    assert len(cols) == len(set(cols)), f"duplicate column in select clause: {cols}"


def test_hard_excluded_columns_includes_target_source_and_drop_decisions():
    excluded = hard_excluded_columns()
    assert "loan_status" in excluded  # target_source
    assert "total_pymnt" in excluded  # drop (post-origination)


def test_assert_no_banned_columns_passes_for_clean_set():
    assert_no_banned_columns({"fico_mean", "dti", "annual_inc"})


def test_assert_no_banned_columns_raises_for_a_banned_column():
    with pytest.raises(AssertionError, match="loan_status"):
        assert_no_banned_columns({"fico_mean", "loan_status"})


# ---------------------------------------------------------------------
# build_features.py main(), end to end against a synthetic fixture
# ---------------------------------------------------------------------

NON_NULL_NUMERIC_COLUMNS = sorted(
    set(feature_columns()) | {"fico_range_low", "fico_range_high"}
    - {"home_ownership", "purpose", "verification_status"}
)


def _base_row(row_id: int, issue_d: str, loan_status: str) -> dict:
    row = {
        "id": row_id,
        "term": 36,
        "application_type": "Individual",
        "issue_d": issue_d,
        "loan_status": loan_status,
        "earliest_cr_line": "2005-01-01",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "verification_status": "Verified",
        "annual_inc": 50000.0,
        "loan_amnt": 10000.0,
        "fico_range_low": 680.0,
        "fico_range_high": 684.0,
        "revol_bal": 5000.0,
        "tot_cur_bal": 20000.0,
        "mths_since_last_delinq": 12.0,
        "mths_since_last_major_derog": 24.0,
        "mths_since_last_record": 36.0,
        "mths_since_recent_bc_dlq": 6.0,
        "mths_since_recent_revol_delinq": 6.0,
        "mths_since_recent_inq": 3.0,
    }
    for col in NON_NULL_NUMERIC_COLUMNS:
        row.setdefault(col, 1.0)
    return row


@pytest.fixture
def synthetic_cleaned_full(tmp_path, monkeypatch) -> Path:
    rows = [
        _base_row(1, "2012-06-01", "Charged Off"),       # train, target=1
        _base_row(2, "2013-06-01", "Fully Paid"),         # train, target=0
        _base_row(3, "2014-06-01", "Charged Off"),        # validation, target=1
        _base_row(4, "2015-06-01", "Fully Paid"),         # test, target=0
        _base_row(5, "2012-06-01", "Current"),            # excluded, target NULL
        _base_row(6, "2017-06-01", "Charged Off"),         # replay, target forced NULL
    ]
    df = pd.DataFrame(rows)
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
    path = tmp_path / "cleaned.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(target_mod, "CLEANED_PARQUET", path)
    return path


def test_main_writes_all_four_splits_with_no_banned_columns(synthetic_cleaned_full, tmp_path, monkeypatch):
    monkeypatch.setattr(bf_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "metrics").mkdir(parents=True)

    bf_mod.main()

    train = pd.read_parquet(tmp_path / "processed" / "train.parquet")
    validation = pd.read_parquet(tmp_path / "processed" / "validation.parquet")
    test = pd.read_parquet(tmp_path / "processed" / "test.parquet")
    replay = pd.read_parquet(tmp_path / "processed" / "replay.parquet")

    assert list(train["id"]) == [1, 2]
    assert list(validation["id"]) == [3]
    assert list(test["id"]) == [4]
    assert list(replay["id"]) == [6]
    assert replay["target"].isna().all()

    banned = bf_mod.hard_excluded_columns()
    for df in (train, validation, test, replay):
        assert banned.isdisjoint(df.columns)
        assert "fico_mean" in df.columns  # engineered feature made it through


def test_main_writes_summary_json(synthetic_cleaned_full, tmp_path, monkeypatch):
    monkeypatch.setattr(bf_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "metrics").mkdir(parents=True)

    bf_mod.main()

    import json

    with open(tmp_path / "reports" / "metrics" / "data" / "build_features_summary.json") as f:
        summary = json.load(f)
    assert summary["train"]["rows"] == 2
    assert summary["validation"]["rows"] == 1
    assert summary["test"]["rows"] == 1
    assert summary["replay"]["target_entirely_null"] is True


def test_assert_engineered_features_finite_passes_on_clean_data(tmp_path):
    con = duckdb.connect()
    df = pd.DataFrame({
        "fico_mean": [680.0, 700.0],
        "credit_history_months": [100.0, 150.0],
        "loan_to_income": [0.2, 0.3],
        "revol_bal_to_income": [0.1, 0.05],
    })
    path = tmp_path / "clean.parquet"
    df.to_parquet(path, index=False)
    bf_mod.assert_engineered_features_finite(con, str(path), "train")  # must not raise


def test_assert_engineered_features_finite_raises_on_infinite_value(tmp_path):
    con = duckdb.connect()
    df = pd.DataFrame({
        "fico_mean": [680.0, 700.0],
        "credit_history_months": [100.0, 150.0],
        "loan_to_income": [float("inf"), 0.3],
        "revol_bal_to_income": [0.1, 0.05],
    })
    path = tmp_path / "bad.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(AssertionError, match="non-finite"):
        bf_mod.assert_engineered_features_finite(con, str(path), "train")


# ---------------------------------------------------------------------
# feature_engineering.py: engineered feature functions, synthetic DataFrame
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# never_flag__* rollout gating: the two gating mechanisms (row-level,
# via tot_cur_bal; date-level, via a 2012-04-01 cutoff) must null out
# the flag for pre-rollout rows instead of marking it "1" (which would
# wrongly claim "never happened" for a field that simply didn't exist
# in the data yet).
# ---------------------------------------------------------------------


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
