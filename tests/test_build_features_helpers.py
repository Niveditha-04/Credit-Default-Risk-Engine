"""src/features/build_features.py's pure helper functions (operating on
the in-memory allow-list, no real data file needed)."""

import pytest

from src.features.build_features import (
    ALWAYS_SELECTED_COLUMNS,
    assert_no_banned_columns,
    build_select_clause,
    derivation_only_columns,
    engineered_feature_names,
    hard_excluded_columns,
)


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
