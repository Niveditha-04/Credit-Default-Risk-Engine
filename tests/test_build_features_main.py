"""End-to-end run of src/features/build_features.py's main() against a
tiny synthetic cleaned-parquet fixture - no real data needed. Covers the
one-time data-build path (write train/validation/test/replay parquet,
the banned-column assertion, the finite-value assertion, the replay
target-is-null assertion, and the summary json) that the pure-helper
tests in test_build_features_helpers.py don't exercise."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data import target as target_mod
from src.features import build_features as bf_mod
from src.features.allow_list import feature_columns

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

    with open(tmp_path / "reports" / "metrics" / "build_features_summary.json") as f:
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
