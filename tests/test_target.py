"""Target mapping and population filters, against a tiny synthetic
table - no real data needed."""

import itertools
import json
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


def test_population_and_target_report_counts(synthetic_cleaned):
    con = duckdb.connect()
    report = target_mod.population_and_target_report(con)
    assert report["splits"]["train"]["n_default"] == 2  # ids 1, 2
    assert report["splits"]["train"]["n_excluded_unresolved"] == 5  # ids 5-9
    assert report["splits"]["validation"]["n_fully_paid"] == 1  # id 3
    assert report["splits"]["test"]["n_fully_paid"] == 1  # id 4
    assert report["replay"]["n_rows"] == 2  # ids 13, 14
    assert "In Grace Period" in report["exclusions_by_status"]


def test_split_years_assigned_correctly(synthetic_cleaned):
    con = duckdb.connect()
    result = con.execute(f"SELECT id, split FROM ({target_mod.labelled_population_query()})").fetchdf()
    split_by_id = dict(zip(result["id"], result["split"]))
    assert split_by_id[1] == "train"       # 2012
    assert split_by_id[2] == "train"       # 2013
    assert split_by_id[3] == "validation"  # 2014
    assert split_by_id[4] == "test"        # 2015


def test_split_year_ranges_never_overlap():
    """train/validation/test/replay must partition issue years with no
    year assigned to two splits - a date-range bug here would silently
    leak rows across the time-based split the whole evaluation depends on."""
    ranges = [target_mod.SPLIT_YEARS[s] for s in ("train", "validation", "test", "replay")]
    seen_years: set[int] = set()
    for y0, y1 in ranges:
        assert y0 <= y1
        years = set(range(y0, y1 + 1))
        assert seen_years.isdisjoint(years), f"years {seen_years & years} assigned to more than one split"
        seen_years |= years
    # and no gap between the end of one split and the start of the next
    boundaries = sorted(ranges)
    for (_, prev_end), (next_start, _) in itertools.pairwise(boundaries):
        assert next_start == prev_end + 1


def test_split_boundary_years_land_in_the_right_split(synthetic_cleaned, monkeypatch, tmp_path):
    rows = [
        (101, 36, "Individual", "2013-12-31", "Fully Paid"),  # last day of train
        (102, 36, "Individual", "2014-01-01", "Fully Paid"),  # first day of validation
        (103, 36, "Individual", "2015-12-31", "Fully Paid"),  # last day of test
        (104, 36, "Individual", "2016-01-01", "Fully Paid"),  # first day of replay (no labelled split)
    ]
    df = pd.DataFrame(rows, columns=["id", "term", "application_type", "issue_d", "loan_status"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    path = tmp_path / "boundary_cleaned.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(target_mod, "CLEANED_PARQUET", path)

    con = duckdb.connect()
    result = con.execute(f"SELECT id, split FROM ({target_mod.labelled_population_query()})").fetchdf()
    split_by_id = dict(zip(result["id"], result["split"]))
    assert split_by_id[101] == "train"
    assert split_by_id[102] == "validation"
    assert split_by_id[103] == "test"
    assert 104 not in split_by_id  # 2016 is replay-only, not part of the labelled population


def test_unrecognized_loan_status_gets_null_target_not_a_crash(synthetic_cleaned, monkeypatch, tmp_path):
    """A loan_status this project has never seen (e.g. a new LendingClub
    status) must fall through to NULL (excluded), not raise or silently
    get mapped to 0/1."""
    rows = [(201, 36, "Individual", "2012-06-01", "Some New Status Nobody Has Seen")]
    df = pd.DataFrame(rows, columns=["id", "term", "application_type", "issue_d", "loan_status"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    path = tmp_path / "unknown_status_cleaned.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(target_mod, "CLEANED_PARQUET", path)

    con = duckdb.connect()
    result = con.execute(f"SELECT id, target FROM ({target_mod.labelled_population_query()})").fetchdf()
    assert pd.isna(result.loc[result["id"] == 201, "target"].iloc[0])


def test_issue_year_outside_every_defined_split_is_excluded_entirely(synthetic_cleaned):
    """id 12 (issued 2011) is outside both the labelled (2012-2015) and
    replay (2016-2018) populations - it must appear in neither."""
    con = duckdb.connect()
    labelled_ids = set(
        con.execute(f"SELECT id FROM ({target_mod.labelled_population_query()})").fetchdf()["id"]
    )
    replay_ids = set(
        con.execute(f"SELECT id FROM ({target_mod.replay_population_query()})").fetchdf()["id"]
    )
    assert 12 not in labelled_ids
    assert 12 not in replay_ids


def test_main_writes_report_to_given_path(synthetic_cleaned, tmp_path):
    out_path = tmp_path / "out" / "population_and_target.json"
    report = target_mod.main(out_path=str(out_path))
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["splits"]["train"]["n_default"] == report["splits"]["train"]["n_default"] == 2
