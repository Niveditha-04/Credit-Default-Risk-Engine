"""Exact allow-list assertion (src.models.gbm.feature_list): the model's
feature list must equal the frozen allow-list exactly, not just avoid a
handful of specifically-named banned columns. frozen_model_config.json
is a small, committed metrics file (not real training data), so this
runs in CI."""

import json

import pytest

from src.models.gbm import feature_list

with open("reports/metrics/frozen_model_config.json") as f:
    FROZEN_FEATURES = json.load(f)["lightgbm_chosen"]["features"]


def test_real_table_columns_pass():
    train_columns = ["id", "issue_d", "split", "target"] + FROZEN_FEATURES
    assert feature_list(train_columns) == FROZEN_FEATURES


def test_injected_extra_column_is_caught():
    """Prove the exact allow-list would have caught the issue_d_1 bug:
    inject a stray extra column once, confirm the assertion fires, then
    (implicitly, since nothing is mutated on disk) the check reverts -
    this test never writes to any file."""
    train_columns = ["id", "issue_d", "split", "target"] + FROZEN_FEATURES + ["issue_d_1"]
    with pytest.raises(AssertionError, match="issue_d_1"):
        feature_list(train_columns)


def test_injected_banned_column_is_caught():
    """A column that predates origination (not even on the old deny-list)
    must also be caught - this is the entire point of an allow-list over
    a deny-list."""
    train_columns = ["id", "issue_d", "split", "target"] + FROZEN_FEATURES + ["grade"]
    with pytest.raises(AssertionError, match="grade"):
        feature_list(train_columns)


def test_missing_column_is_caught():
    train_columns = ["id", "issue_d", "split", "target"] + FROZEN_FEATURES[1:]
    with pytest.raises(AssertionError, match=FROZEN_FEATURES[0]):
        feature_list(train_columns)
