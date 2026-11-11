"""ORDER BY determinism: every query that gets materialized to a
parquet file must return rows in the same order every time it runs, and
that order must be sorted by id (see src/models/stability.py). An
unordered query lets row order become an unspecified side effect of the
query plan, which LightGBM's seeded bagging/feature subsampling is
sensitive to even with a fixed seed."""

import duckdb
import pandas as pd
import pytest

from src.data import target as target_mod


@pytest.fixture
def synthetic_cleaned(tmp_path, monkeypatch) -> None:
    # Deliberately inserted out of id order, so a query with no ORDER BY
    # would have no reason to come back sorted.
    rows = [
        (5, 36, "Individual", "2012-06-01", "Charged Off"),
        (1, 36, "Individual", "2013-06-01", "Fully Paid"),
        (9, 36, "Individual", "2014-06-01", "Fully Paid"),
        (3, 36, "Individual", "2015-06-01", "Charged Off"),
        (7, 36, "Individual", "2012-06-01", "Fully Paid"),
    ]
    df = pd.DataFrame(rows, columns=["id", "term", "application_type", "issue_d", "loan_status"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    path = tmp_path / "cleaned.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(target_mod, "CLEANED_PARQUET", path)


def test_labelled_population_query_is_sorted_by_id(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    assert list(result["id"]) == sorted(result["id"]), "labelled_population_query() must return rows sorted by id"


def test_replay_population_query_is_sorted_by_id(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.replay_population_query()})").fetchdf()
    assert list(result["id"]) == sorted(result["id"]), "replay_population_query() must return rows sorted by id"


def test_labelled_population_query_is_repeatable(synthetic_cleaned):
    con = duckdb.connect()
    first = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    second = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    assert list(first["id"]) == list(second["id"]), "the same query run twice must return the same row order"
