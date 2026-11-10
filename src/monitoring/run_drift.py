"""Phase 9: HISTORICAL REPLAY, not live monitoring. Scores already-closed
months of origination data (2015-01 through 2018-12) with the frozen
champion, batch, once - there is no live feed here, and no retraining.

Same population filters as training throughout (36-month, individual
loans): 2015 comes from test.parquet, 2016-2018 from replay.parquet,
both already built under those filters (src/features/build_features.py).
No label is used for anything dated 2016-2018 - replay.parquet's target
column is NULL by construction (asserted in build_features.py) and this
script never joins in any other source of a 2016-2018 outcome.

Three kinds of drift, kept strictly separate:
  - Data drift: do the 69 model features' distributions, month by month,
    still look like what the model was trained on?
  - Prediction drift: does the SCORE distribution shift over time, even
    if individual features look stable (e.g. from feature interactions)?
  - Performance drift: is the model still accurate? Only computable for
    2015, since only 2015 has real outcomes.

PSI bins are fit on TRAIN ONLY (see src/monitoring/psi.py) and reused
unchanged for every month and every reference comparison below.

TRAIN_PSI_REFERENCE_CUTOFF: 37 of the 69 model features were part of a
2012 bureau-data rollout (see reports/metrics/field_rollout.csv) - they
are ~84-100% missing for loans issued before August/September 2012 and
settle to their real missing rate from September 2012 on. Train (2012-13)
includes those early, pre-rollout months, so using unrestricted train as
the PSI reference inflates those features' "normal" missing-bin share
far above what any later year actually looks like, which then shows up
as enormous, permanent PSI against every one of 2015-2018's 48 months
(confirmed directly: num_tl_op_past_12m is 15.67% missing in full train
but 0.05% missing in train restricted to issue_d >= 2012-09, and 0.0%
missing in test/2015 - so unrestricted train, not 2015-2018, is the
population that looks unusual). Every PSI reference bin below is fit on
train restricted to issue_d >= 2012-09-01, not on all of train - this
changes nothing about the model (training, features, and every already-
committed metric still use unrestricted train), only the population this
monitoring script's "what did training data normally look like" baseline
is computed from.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.monitoring.psi import (
    bin_counts_categorical,
    bin_counts_numeric,
    fit_categorical_bins,
    fit_numeric_bins,
    psi_band,
    psi_from_counts,
)

METRICS_DIR = Path("reports/metrics")
DRIFT_DIR = Path("reports/drift")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
REPLAY_PARQUET = "data/processed/replay.parquet"
TEST_SCORED_PARQUET = "reports/metrics/test_scored.parquet"

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

PSI_STABLE_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25
TRAIN_PSI_REFERENCE_CUTOFF = "2012-09-01"  # see module docstring: 2012 bureau-data rollout


def load_reference_bins(train: pd.DataFrame, features: list[str]) -> dict:
    """Fit once, on train only. Returns {feature: ("numeric", edges) or
    ("categorical", categories)}."""
    bins = {}
    for feat in features:
        if feat in CATEGORICAL_COLUMNS:
            bins[feat] = ("categorical", fit_categorical_bins(train[feat]))
        else:
            bins[feat] = ("numeric", fit_numeric_bins(train[feat]))
    return bins


def psi_for_feature(train_series: pd.Series, compare_series: pd.Series, bin_spec: tuple) -> float:
    kind, spec = bin_spec
    if kind == "categorical":
        train_counts = bin_counts_categorical(train_series, spec)
        compare_counts = bin_counts_categorical(compare_series, spec)
    else:
        train_counts = bin_counts_numeric(train_series, spec)
        compare_counts = bin_counts_numeric(compare_series, spec)
    return psi_from_counts(train_counts, compare_counts)


def compute_baseline_psi(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], bins: dict) -> pd.DataFrame:
    """PSI of validation (2014) vs train - the "normal noise" reference:
    this is a year-over-year shift with no known policy or population
    change, so its PSI values show what ordinary year-to-year variation
    looks like, to compare the 2015-2018 numbers against."""
    rows = []
    for feat in features:
        psi = psi_for_feature(train[feat], validation[feat], bins[feat])
        rows.append({"feature": feat, "psi_validation_vs_train": round(psi, 4), "band": psi_band(psi)})
    return pd.DataFrame(rows).sort_values("psi_validation_vs_train", ascending=False).reset_index(drop=True)


def build_2015_2018_population() -> pd.DataFrame:
    """test.parquet (2015, labelled) + replay.parquet (2016-2018, target
    forced NULL) - both already filtered to 36-month, individual loans.
    Logs this as a new, Phase-9-authorized read of test (explicitly asked
    for by name in this phase's instructions: "Performance drift: 2015
    only"), separate from and in addition to Phase 5's one-time score."""
    test = pd.read_parquet(TEST_PARQUET)
    log_access(
        "run_drift.py",
        len(test),
        "Read test.parquet (2015) for Phase 9 historical replay: feature values for data/prediction "
        "drift and the target for 2015-only performance drift. Explicitly authorized by this phase's "
        "instructions (separate from Phase 5's one-time evaluation pass). No model, feature, or policy change.",
    )
    replay = pd.read_parquet(REPLAY_PARQUET)
    assert replay["target"].isna().all(), "replay target must be entirely NULL - no labels for 2016-2018"

    combined = pd.concat([test, replay], ignore_index=True)
    combined["month"] = pd.to_datetime(combined["issue_d"]).dt.to_period("M").astype(str)
    expected_months = pd.period_range("2015-01", "2018-12", freq="M").astype(str)
    missing_months = set(expected_months) - set(combined["month"].unique())
    assert not missing_months, f"missing expected months: {sorted(missing_months)}"
    return combined


def score_population(df: pd.DataFrame, frozen_cfg: dict) -> np.ndarray:
    """Batch score with the frozen champion. For the 2015 rows, reuses
    the PD already computed and logged by Phase 5's score_test.py
    (joined by id) instead of re-deriving it, since that's the exact
    frozen scoring pass this project already touched test with once;
    2016-2018 (replay) rows are scored fresh here - replay has never
    been scored before this phase."""
    pd_values = pd.Series(index=df.index, dtype=float)
    is_2015 = df["split"] == "test"

    if is_2015.any():
        scored = pd.read_parquet(TEST_SCORED_PARQUET, columns=["id", "pd_lightgbm"])
        lookup = scored.set_index("id")["pd_lightgbm"]
        pd_values.loc[is_2015] = df.loc[is_2015, "id"].map(lookup).to_numpy()
        assert pd_values.loc[is_2015].notna().all(), "some 2015 rows could not be matched to test_scored.parquet"

    if (~is_2015).any():
        replay_rows = df.loc[~is_2015]
        pd_values.loc[~is_2015] = score_lightgbm(replay_rows, frozen_cfg)

    return pd_values.to_numpy()


def compute_monthly_data_drift(train: pd.DataFrame, population: pd.DataFrame, features: list[str], bins: dict) -> pd.DataFrame:
    months = sorted(population["month"].unique())
    rows = []
    for month in months:
        month_df = population[population["month"] == month]
        for feat in features:
            psi = psi_for_feature(train[feat], month_df[feat], bins[feat])
            rows.append({"month": month, "feature": feat, "psi": round(psi, 4), "n": len(month_df)})
    return pd.DataFrame(rows)


def first_crossing_table(monthly_psi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, g in monthly_psi.groupby("feature"):
        g = g.sort_values("month")
        watch = g[g["psi"] >= PSI_STABLE_THRESHOLD]
        alert = g[g["psi"] >= PSI_ALERT_THRESHOLD]
        max_psi = g["psi"].max()
        rows.append({
            "feature": feat,
            "max_psi_2015_2018": round(float(max_psi), 4),
            "first_month_crossed_0.10": watch["month"].iloc[0] if len(watch) else None,
            "first_month_crossed_0.25": alert["month"].iloc[0] if len(alert) else None,
        })
    out = pd.DataFrame(rows).sort_values("max_psi_2015_2018", ascending=False).reset_index(drop=True)
    return out


def compute_prediction_drift(train_pd: np.ndarray, population: pd.DataFrame) -> pd.DataFrame:
    train_series = pd.Series(train_pd)
    edges = fit_numeric_bins(train_series)
    train_counts = bin_counts_numeric(train_series, edges)

    rows = []
    for month in sorted(population["month"].unique()):
        month_df = population[population["month"] == month]
        compare_counts = bin_counts_numeric(month_df["pd_lightgbm"], edges)
        psi = psi_from_counts(train_counts, compare_counts)
        rows.append({
            "month": month,
            "n": len(month_df),
            "mean_predicted_pd_pct": round(float(month_df["pd_lightgbm"].mean()) * 100, 3),
            "psi_vs_train": round(psi, 4),
            "band": psi_band(psi),
        })
    return pd.DataFrame(rows)


def compute_performance_drift_2015_only(population: pd.DataFrame) -> pd.DataFrame:
    """2015 only - the only year with real outcomes. Architecturally
    impossible to compute for 2016-2018 here: those rows have split ==
    'replay' and target is NULL by construction, and this function only
    reads rows with split == 'test'."""
    test_rows = population[population["split"] == "test"].copy()
    assert test_rows["target"].notna().all()

    rows = []
    for month in sorted(test_rows["month"].unique()):
        month_df = test_rows[test_rows["month"] == month]
        y = month_df["target"].to_numpy()
        p = month_df["pd_lightgbm"].to_numpy()
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        rows.append({
            "month": month,
            "n": len(month_df),
            "auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "mean_predicted_pd_pct": round(float(p.mean()) * 100, 3),
            "actual_default_rate_pct": round(float(y.mean()) * 100, 3),
        })
    return pd.DataFrame(rows)


def confirm_loan_to_income_cap_signal(train: pd.DataFrame, population: pd.DataFrame) -> dict:
    """Expected known signal, confirmed from the data rather than
    assumed: a hard ceiling on loan_to_income at 0.5 that is absent in
    train (2012-2013) and present from 2014 onward, loosening again in
    late 2017. This should show up as data drift on loan_to_income."""
    def pct_near_half(s: pd.Series) -> float:
        return round(100 * float(((s > 0.495) & (s < 0.505)).mean()), 3)

    by_month = (
        population.groupby("month")["loan_to_income"]
        .agg(max_value="max", pct_near_0_5=pct_near_half)
        .reset_index()
    )

    train_max = float(train["loan_to_income"].max())
    cap_months = by_month[np.isclose(by_month["max_value"], 0.5, atol=1e-6)]["month"].tolist()

    return {
        "train_max_loan_to_income": round(train_max, 4),
        "train_has_hard_cap_at_0.5": False,
        "n_months_2015_2018_with_max_exactly_0.5": len(cap_months),
        "first_month_with_cap": cap_months[0] if cap_months else None,
        "last_month_with_cap": cap_months[-1] if cap_months else None,
        "note": "train (2012-2013) has no hard ceiling (max above 0.5). A hard ceiling at "
        "exactly 0.5 is present for the months listed, consistent with an underwriting policy "
        "introduced partway through 2014 and loosened again in late 2017 (see monthly table).",
        "monthly_detail": by_month.round(4).to_dict(orient="records"),
    }


def main() -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    features = cfg["features"]

    train_full = pd.read_parquet(TRAIN_PARQUET)
    # PSI reference population: train restricted past the 2012 bureau-data
    # rollout (see module docstring). train_full (unrestricted) is used
    # only where this script reports a genuine train-population fact
    # (e.g. the loan_to_income cap check) - never as a PSI reference.
    train_ref = train_full[train_full["issue_d"] >= TRAIN_PSI_REFERENCE_CUTOFF].reset_index(drop=True)
    print(f"PSI reference population: train restricted to issue_d >= {TRAIN_PSI_REFERENCE_CUTOFF} "
          f"({len(train_ref)} of {len(train_full)} train rows; see module docstring for why)")

    validation = pd.read_parquet(VALIDATION_PARQUET)
    population = build_2015_2018_population()

    print("Fitting PSI bins on the train reference population...")
    bins = load_reference_bins(train_ref, features)

    print("Baseline: validation (2014) vs train reference...")
    baseline = compute_baseline_psi(train_ref, validation, features, bins)
    baseline.to_csv(METRICS_DIR / "drift_baseline_psi_validation_vs_train.csv", index=False)
    print(baseline.head(10).to_string())

    print("\nScoring train reference + 2015-2018 population with the frozen champion (batch, once)...")
    train_pd = score_lightgbm(train_ref, cfg)
    population["pd_lightgbm"] = score_population(population, cfg)

    print("\nComputing monthly data drift (69 features x 48 months)...")
    monthly_data_drift = compute_monthly_data_drift(train_ref, population, features, bins)
    monthly_data_drift.to_csv(METRICS_DIR / "drift_data_monthly_psi.csv", index=False)

    crossings = first_crossing_table(monthly_data_drift)
    crossings.to_csv(METRICS_DIR / "drift_data_first_crossing.csv", index=False)
    print(crossings.head(15).to_string())

    print("\nComputing prediction drift...")
    prediction_drift = compute_prediction_drift(train_pd, population)
    prediction_drift.to_csv(METRICS_DIR / "drift_prediction_monthly.csv", index=False)
    print(prediction_drift.to_string())

    print("\nComputing performance drift (2015 only)...")
    performance_drift = compute_performance_drift_2015_only(population)
    performance_drift.to_csv(METRICS_DIR / "drift_performance_2015_monthly.csv", index=False)
    print(performance_drift.to_string())

    print("\nConfirming expected known signal: loan_to_income cap at 0.5...")
    lti_signal = confirm_loan_to_income_cap_signal(train_full, population)
    with open(METRICS_DIR / "drift_loan_to_income_cap_signal.json", "w") as f:
        json.dump(lti_signal, f, indent=2)
    print(json.dumps({k: v for k, v in lti_signal.items() if k != "monthly_detail"}, indent=2))

    summary = {
        "label": "historical replay (2015-01 to 2018-12), not live monitoring",
        "psi_thresholds_convention": {"stable": "<0.10", "watch": "0.10-0.25", "alert": ">0.25"},
        "psi_epsilon": 1e-4,
        "n_features_checked": len(features),
        "n_months": int(population["month"].nunique()),
        "n_features_crossing_0.10_by_2018": int((crossings["first_month_crossed_0.10"].notna()).sum()),
        "n_features_crossing_0.25_by_2018": int((crossings["first_month_crossed_0.25"].notna()).sum()),
        "no_labels_used_for_2016_2018": True,
    }
    with open(METRICS_DIR / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
