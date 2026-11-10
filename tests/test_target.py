"""Target mapping and population filters, against a tiny synthetic
table - no real data needed."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data import target as target_mod


@pytest.fixture
def synthetic_cleaned(tmp_path, monkeypatch) -> Path:
    rows = [
        # id, term, application_type, issue_d, loan_status -> expected target, expected split
        (1, 36, "Individual", "2012-06-01", "Charged Off"),       # target=1, split=train
        (2, 36, "Individual", "2013-06-01", "Default"),            # target=1, split=train
        (3, 36, "Individual", "2014-06-01", "Fully Paid"),         # target=0, split=validation
        (4, 36, "Individual", "2015-06-01", "Fully Paid"),         # target=0, split=test
        (5, 36, "Individual", "2012-06-01", "Current"),            # excluded (target NULL)
        (6, 36, "Individual", "2012-06-01", "In Grace Period"),    # excluded
        (7, 36, "Individual", "2012-06-01", "Late (16-30 days)"),  # excluded
        (8, 36, "Individual", "2012-06-01", "Late (31-120 days)"),  # excluded
        (9, 36, "Individual", "2012-06-01", "Does not meet the credit policy. Status:Fully Paid"),  # excluded
        (10, 60, "Individual", "2012-06-01", "Charged Off"),       # wrong term, excluded from population
        (11, 36, "Joint App", "2012-06-01", "Charged Off"),        # wrong application_type, excluded
        (12, 36, "Individual", "2011-06-01", "Charged Off"),       # outside 2012-2015, excluded from labelled pop
        (13, 36, "Individual", "2016-06-01", "Charged Off"),       # replay year
        (14, 36, "Individual", "2017-06-01", "Fully Paid"),        # replay year, resolved status
    ]
    df = pd.DataFrame(rows, columns=["id", "term", "application_type", "issue_d", "loan_status"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    path = tmp_path / "cleaned.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(target_mod, "CLEANED_PARQUET", path)
    return path


def test_target_mapping(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id, target FROM ({target_mod.labelled_population_query()}) ORDER BY id").fetchdf()
    mapping = dict(zip(result["id"], result["target"]))
    assert mapping[1] == 1  # Charged Off
    assert mapping[2] == 1  # Default
    assert mapping[3] == 0  # Fully Paid
    assert mapping[4] == 0  # Fully Paid


def test_excluded_statuses_get_null_target_and_drop_out(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()}) WHERE target IS NOT NULL").fetchdf()
    present_ids = set(result["id"])
    for excluded_id in [5, 6, 7, 8, 9]:
        assert excluded_id not in present_ids, f"id {excluded_id} (an excluded status) leaked into the labelled population"


def test_population_filter_term_36_only(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    assert 10 not in set(result["id"]), "a 60-month loan leaked into the population"


def test_population_filter_individual_only(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    assert 11 not in set(result["id"]), "a non-Individual application leaked into the population"


def test_population_filter_year_range_2012_2015(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()
    ids = set(result["id"])
    assert 12 not in ids, "a 2011 loan leaked into the 2012-2015 labelled population"
    assert 13 not in ids and 14 not in ids, "2016-2018 loans leaked into the labelled (not replay) population"


def test_replay_population_is_2016_2018_with_null_target(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id, target, split FROM ({target_mod.replay_population_query()})").fetchdf()
    ids = set(result["id"])
    assert ids == {13, 14}
    assert result["target"].isna().all(), "replay target must be NULL for every row, even a resolved 2017 loan"
    assert (result["split"] == "replay").all()


def test_split_years_assigned_correctly(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id, split FROM ({target_mod.labelled_population_query()})").fetchdf()
    split_by_id = dict(zip(result["id"], result["split"]))
    assert split_by_id[1] == "train"       # 2012
    assert split_by_id[2] == "train"       # 2013
    assert split_by_id[3] == "validation"  # 2014
    assert split_by_id[4] == "test"        # 2015
