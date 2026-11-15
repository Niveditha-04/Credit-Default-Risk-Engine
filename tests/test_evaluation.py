"""decision_quality_row on a small synthetic fixture: no file I/O,
no real model. Checks the identities the real function must hold
regardless of input - capture and miss rates must sum to 100%, and
every count must add back up to N.
"""

import numpy as np
import pandas as pd

from src.evaluation.decision_quality import decision_quality_row, rank_based_approval


def make_fixture(n=20, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "target": rng.randint(0, 2, n),
        "pd_lightgbm": rng.uniform(0, 1, n),
        "total_pymnt": rng.uniform(5000, 15000, n),
        "collection_recovery_fee": rng.uniform(0, 200, n),
        "funded_amnt": rng.uniform(8000, 12000, n),
    })
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    return df


def test_capture_and_miss_rate_sum_to_100():
    df = make_fixture()
    mask, cutoff = rank_based_approval(df, 60)
    row = decision_quality_row(df, mask, "60% (test)", cutoff)
    assert abs(row["default_capture_rate_pct"] + row["default_miss_rate_pct"] - 100.0) < 1e-6


def test_counts_sum_to_n():
    df = make_fixture()
    mask, cutoff = rank_based_approval(df, 40)
    row = decision_quality_row(df, mask, "40% (test)", cutoff)
    assert row["n_approved"] + row["n_declined"] == row["n_total"] == len(df)
    assert row["n_defaults_captured_by_decline"] + row["n_defaults_missed_still_approved"] == row["n_defaults_total"]


def test_all_approved_captures_nothing():
    df = make_fixture()
    mask = pd.Series(True, index=df.index)
    row = decision_quality_row(df, mask, "100% (test)", float("inf"))
    assert row["n_declined"] == 0
    assert row["default_capture_rate_pct"] == 0.0
    assert row["default_miss_rate_pct"] == 100.0
    assert np.isnan(row["precision_of_declines_pct"])


def test_all_declined_captures_everything():
    df = make_fixture()
    mask = pd.Series(False, index=df.index)
    row = decision_quality_row(df, mask, "0% (test)", float("-inf"))
    assert row["n_approved"] == 0
    assert row["default_capture_rate_pct"] == 100.0
    assert row["default_miss_rate_pct"] == 0.0


def test_rank_based_approval_approves_lowest_pd_rows():
    df = make_fixture()
    mask, _ = rank_based_approval(df, 50)
    assert mask.sum() == len(df) // 2
    approved_max_pd = df.loc[mask, "pd_lightgbm"].max()
    declined_min_pd = df.loc[~mask, "pd_lightgbm"].min()
    assert approved_max_pd <= declined_min_pd
