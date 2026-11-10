"""PSI function vs a hand calculation, plus the edge cases the real
pipeline actually hits (empty bins, missing values, unseen categories).
"""

import numpy as np
import pandas as pd

from src.monitoring.psi import (
    EPSILON,
    bin_counts_categorical,
    bin_counts_numeric,
    fit_categorical_bins,
    fit_numeric_bins,
    psi_band,
    psi_categorical,
    psi_from_counts,
    psi_numeric,
)


def test_psi_matches_hand_calculation():
    # Two bins: train 50/50, compare 40/60.
    # PSI = (0.4-0.5)*ln(0.4/0.5) + (0.6-0.5)*ln(0.6/0.5)
    #     = -0.1 * ln(0.8) + 0.1 * ln(1.2)
    #     = 0.0223143551 + 0.0182321557 = 0.0405465108
    train_counts = np.array([50, 50])
    compare_counts = np.array([40, 60])
    expected = 0.0405465108
    result = psi_from_counts(train_counts, compare_counts)
    assert np.isclose(result, expected, atol=1e-8)


def test_psi_zero_for_identical_distributions():
    counts = np.array([30, 30, 40])
    assert np.isclose(psi_from_counts(counts, counts), 0.0, atol=1e-12)


def test_psi_handles_empty_bin_without_crashing():
    # compare has zero count in the second bin - would be log(0) without
    # the epsilon substitution.
    train_counts = np.array([50, 50])
    compare_counts = np.array([100, 0])
    result = psi_from_counts(train_counts, compare_counts)
    assert np.isfinite(result)
    assert result > 0
    # Hand-check using the documented epsilon substitution directly.
    t = train_counts / train_counts.sum()
    c = np.array([1.0, EPSILON])
    expected = np.sum((c - t) * np.log(c / t))
    assert np.isclose(result, expected, atol=1e-8)


def test_fit_numeric_bins_deciles_on_train_only():
    train = pd.Series(range(1, 101), dtype=float)  # 1..100
    edges = fit_numeric_bins(train, n_bins=10)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    # 9 interior decile edges + 2 infinite caps = 11 edges for 10 bins
    assert len(edges) == 11


def test_fit_numeric_bins_constant_column():
    train = pd.Series([5.0] * 20)
    edges = fit_numeric_bins(train)
    assert list(edges) == [-np.inf, np.inf]


def test_numeric_psi_same_population_is_zero():
    rng = np.random.RandomState(0)
    train = pd.Series(rng.normal(size=5000))
    assert np.isclose(psi_numeric(train, train), 0.0, atol=1e-9)


def test_numeric_psi_missing_values_get_own_bin():
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    compare = pd.Series([np.nan] * 100)  # entirely missing
    edges = fit_numeric_bins(train, n_bins=4)
    compare_counts = bin_counts_numeric(compare, edges)
    assert compare_counts[-1] == 100  # all 100 landed in the missing bin
    assert compare_counts[:-1].sum() == 0
    result = psi_numeric(train, compare, edges)
    assert np.isfinite(result) and result > 0


def test_categorical_psi_unseen_category_counted_not_dropped():
    train = pd.Series(["a", "b", "a", "b", "c"] * 20)
    categories = fit_categorical_bins(train)
    assert categories == ["a", "b", "c"]
    compare = pd.Series(["a"] * 50 + ["totally_new_category"] * 50)
    counts = bin_counts_categorical(compare, categories)
    # order: a, b, c, missing, unseen
    assert counts[0] == 50  # a
    assert counts[1] == 0 and counts[2] == 0  # b, c
    assert counts[3] == 0  # missing
    assert counts[4] == 50  # unseen
    result = psi_categorical(train, compare, categories)
    assert np.isfinite(result) and result > 0


def test_psi_band_thresholds():
    assert psi_band(0.05) == "stable"
    assert psi_band(0.0999) == "stable"
    assert psi_band(0.10) == "watch"
    assert psi_band(0.25) == "watch"
    assert psi_band(0.2501) == "alert"
    assert psi_band(1.0) == "alert"
