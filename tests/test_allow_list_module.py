"""src/features/allow_list.py: the allow-list declaration is pure,
in-memory data (built from the 151 raw columns), so these functions need
no real data file to test."""

import csv

import pytest

from src.features.allow_list import (
    ALLOW_LIST,
    _load_allow_list,
    _parse_known_at_origination,
    banned_columns,
    check_completeness,
    feature_columns,
    main,
    write_feature_audit_csv,
)


def test_check_completeness_passes_for_the_real_allow_list():
    check_completeness()  # raises AssertionError if any column is missing a decision


def test_banned_columns_excludes_keep_feature_columns():
    banned = banned_columns()
    kept = set(feature_columns())
    assert banned.isdisjoint(kept)
    assert banned | kept == set(ALLOW_LIST.keys())


def test_feature_columns_is_sorted_and_matches_keep_feature_decisions():
    feats = feature_columns()
    assert feats == sorted(feats)
    assert all(ALLOW_LIST[c].decision == "keep_feature" for c in feats)


def test_known_banned_columns_are_excluded_from_features():
    feats = set(feature_columns())
    for banned in ("id", "target", "loan_status", "grade", "sub_grade", "int_rate", "installment"):
        assert banned not in feats


def test_write_feature_audit_csv_writes_one_row_per_column(tmp_path):
    out = tmp_path / "feature_audit.csv"
    write_feature_audit_csv(str(out))
    lines = out.read_text().splitlines()
    assert len(lines) == len(ALLOW_LIST) + 1  # header + one row per column
    assert lines[0] == "column,decision,known_at_origination,reason"


def test_parse_known_at_origination_values():
    assert _parse_known_at_origination("") is None
    assert _parse_known_at_origination("True") is True
    assert _parse_known_at_origination("False") is False


def test_parse_known_at_origination_rejects_unexpected_value():
    with pytest.raises(ValueError, match="unexpected known_at_origination value"):
        _parse_known_at_origination("maybe")


def test_load_allow_list_rejects_duplicate_column(tmp_path):
    path = tmp_path / "dup_allow_list.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "decision", "known_at_origination", "reason"])
        writer.writerow(["loan_amnt", "keep_feature", "True", "first"])
        writer.writerow(["loan_amnt", "drop", "", "second, conflicting row"])
    with pytest.raises(ValueError, match="loan_amnt already has a decision"):
        _load_allow_list(path)


def test_main_runs_against_the_real_allow_list(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    main()
    out = capsys.readouterr().out
    assert f"All {len(ALLOW_LIST)} columns have a decision." in out
    assert (tmp_path / "reports" / "metrics" / "feature_audit.csv").exists()
