"""Population Stability Index (PSI). This is a HISTORICAL REPLAY tool,
not a live monitoring pipeline: it compares already-closed months of
origination data against a fixed reference, it does not watch a live
feed.

Bins are fit on TRAIN ONLY and reused unchanged for every comparison
population (validation, test, replay). Numeric bins are train deciles;
categorical bins are train's observed categories. Missing values get
their own separate bin in both cases, rather than being dropped or
folded into another bin.

Thresholds (industry convention, not derived from this data):
    PSI < 0.10            stable
    0.10 <= PSI <= 0.25   watch
    PSI > 0.25            alert

EPSILON = 1e-4 replaces a zero bin proportion before the ratio/log step.
An empty bin would otherwise produce log(0) (minus infinity) or a
division by zero; 1e-4 is small enough to barely perturb a real
non-empty bin's PSI contribution, but keeps the number finite and
comparable across features.
"""


import numpy as np
import pandas as pd

EPSILON = 1e-4
N_NUMERIC_BINS = 10


def fit_numeric_bins(train_values: pd.Series, n_bins: int = N_NUMERIC_BINS) -> np.ndarray:
    """Decile edges fit on train only. The first and last edges are
    widened to -inf/+inf so a value outside train's observed range (a
    replay-month loan larger than anything seen in training) still falls
    into the nearest bin instead of being silently dropped."""
    clean = train_values.dropna()
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(clean.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        # A constant (zero-variance) column in train: everything correctly
        # falls in one bin by construction.
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_counts_numeric(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Counts per numeric bin, with a trailing bin for missing values."""
    non_missing = values.dropna()
    n_missing = int(values.isna().sum())
    binned = pd.cut(non_missing, bins=edges, include_lowest=True)
    counts = binned.value_counts(sort=False).to_numpy()
    return np.append(counts, n_missing)


def fit_categorical_bins(train_values: pd.Series) -> list:
    """Train's observed categories, sorted for a stable bin order."""
    return sorted(train_values.dropna().unique().tolist())


def bin_counts_categorical(values: pd.Series, categories: list) -> np.ndarray:
    """Counts per known category, then a missing bin, then an "unseen"
    bin for any value present in `values` that was never seen in train -
    this keeps that mass visible in the PSI calculation instead of
    silently dropping it, which would understate drift."""
    known = set(categories)
    counts = [int((values == cat).sum()) for cat in categories]
    is_missing = values.isna()
    is_unseen = (~is_missing) & (~values.isin(known))
    counts.append(int(is_missing.sum()))
    counts.append(int(is_unseen.sum()))
    return np.array(counts)


def psi_from_counts(train_counts: np.ndarray, compare_counts: np.ndarray, epsilon: float = EPSILON) -> float:
    train_total = train_counts.sum()
    compare_total = compare_counts.sum()
    assert train_total > 0 and compare_total > 0, "PSI needs at least one row on each side"
    train_pct = train_counts / train_total
    compare_pct = compare_counts / compare_total
    train_pct = np.where(train_pct == 0, epsilon, train_pct)
    compare_pct = np.where(compare_pct == 0, epsilon, compare_pct)
    return float(np.sum((compare_pct - train_pct) * np.log(compare_pct / train_pct)))


def psi_numeric(train_values: pd.Series, compare_values: pd.Series, edges: np.ndarray | None = None) -> float:
    if edges is None:
        edges = fit_numeric_bins(train_values)
    train_counts = bin_counts_numeric(train_values, edges)
    compare_counts = bin_counts_numeric(compare_values, edges)
    return psi_from_counts(train_counts, compare_counts)


def psi_categorical(train_values: pd.Series, compare_values: pd.Series, categories: list | None = None) -> float:
    if categories is None:
        categories = fit_categorical_bins(train_values)
    train_counts = bin_counts_categorical(train_values, categories)
    compare_counts = bin_counts_categorical(compare_values, categories)
    return psi_from_counts(train_counts, compare_counts)


def psi_band(psi_value: float) -> str:
    if psi_value < 0.10:
        return "stable"
    if psi_value <= 0.25:
        return "watch"
    return "alert"
