"""Post-hoc, descriptive only: did training on a DTI-capped population
cause under- or over-prediction once the cap moved? v1 trained on
train (2012-2013, max DTI 34.99); v2 trained on train plus validation
(2012-2014, max DTI 39.99, reports/metrics/eligibility_caps_by_year.csv).
Both models still score any DTI value fed to them, so a loan above the
training cap is being scored outside the range the model ever saw - a
form of extrapolation, not interpolation.

v1 is checked on test (2015), the population it was actually evaluated
on once, against the same lifetime `target` label used throughout that
evaluation - the gap is comparable in absolute terms across buckets
because the label is identical for both.

v2 is checked on the 2016 holdout (24-month label, the same post-hoc
population used for the pre-registered decision rule), since 2016 is
where DTI briefly reached 49.96. v2's own calibrator targets lifetime
default, not a 24-month outcome, so every bucket's absolute gap is
biased negative by that horizon mismatch alone, not comparable to v1's
absolute numbers or to each other in absolute terms. What is comparable
is each bucket's gap relative to the in-range (DTI <= 35) bucket: the
horizon-mismatch bias should apply roughly equally to every bucket, so
differencing it out isolates whatever extra effect DTI extrapolation
itself has. That relative difference is reported with a seeded
bootstrap 95% CI (resampling each bucket's rows independently, with
replacement).

No model, feature, or policy change. No new model is fit: v2 is scored
with the real, currently-served booster and calibrator
(reports/metrics/frozen_model_config.json), the same artifacts the API
serves predictions from today.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list
from src.models.posthoc_v2 import build_eval_population

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
TEST_PARQUET = "data/processed/test.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
Z_95 = 1.959964

V1_TRAIN_MAX_DTI = 34.99
V2_TRAIN_MAX_DTI = 39.99


def bucket_stats(df: pd.DataFrame, pd_col: str, label_col: str, bucket_label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"bucket": bucket_label, "n": 0}
    mean_pd = float(df[pd_col].mean())
    actual_rate = float(df[label_col].mean())
    gap = actual_rate - mean_pd
    se = np.sqrt(actual_rate * (1 - actual_rate) / n)
    return {
        "bucket": bucket_label,
        "n": n,
        "mean_predicted_pd_pct": round(100 * mean_pd, 3),
        "actual_default_rate_pct": round(100 * actual_rate, 3),
        "gap_actual_minus_predicted_pp": round(100 * gap, 3),
        "gap_ci95_lower_pp": round(100 * (gap - Z_95 * se), 3),
        "gap_ci95_upper_pp": round(100 * (gap + Z_95 * se), 3),
        "gap_exceeds_noise": (gap - Z_95 * se > 0) or (gap + Z_95 * se < 0),
    }


def bootstrap_gap_difference(
    ref_df: pd.DataFrame, other_df: pd.DataFrame, pd_col: str, label_col: str,
    ref_label: str, other_label: str, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """Seeded bootstrap 95% CI on (other bucket's gap) minus (reference
    bucket's gap), each bucket resampled independently, with
    replacement. Isolates a bucket-specific effect from a bias (such as
    a label-horizon mismatch) that would otherwise shift every bucket's
    absolute gap by roughly the same amount."""
    rng = np.random.RandomState(seed)
    ref_pd, ref_y = ref_df[pd_col].to_numpy(), ref_df[label_col].to_numpy()
    other_pd, other_y = other_df[pd_col].to_numpy(), other_df[label_col].to_numpy()
    n_ref, n_other = len(ref_pd), len(other_pd)

    point_estimate = (other_y.mean() - other_pd.mean()) - (ref_y.mean() - ref_pd.mean())

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_ref = rng.randint(0, n_ref, n_ref)
        idx_other = rng.randint(0, n_other, n_other)
        gap_ref = ref_y[idx_ref].mean() - ref_pd[idx_ref].mean()
        gap_other = other_y[idx_other].mean() - other_pd[idx_other].mean()
        diffs[i] = gap_other - gap_ref

    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": f"{other_label} minus {ref_label}",
        "n_other": n_other,
        "n_ref": n_ref,
        "diff_in_gap_pp": round(100 * point_estimate, 3),
        "ci95_lower_pp": round(100 * float(ci_lower), 3),
        "ci95_upper_pp": round(100 * float(ci_upper), 3),
        "n_boot": n_boot,
        "seed": seed,
        "exceeds_noise": (ci_lower > 0) or (ci_upper < 0),
    }


def v1_on_test() -> pd.DataFrame:
    scored = pd.read_parquet(TEST_SCORED)
    log_access(
        "dti_cap_extrapolation.py",
        len(scored),
        "Post-hoc, descriptive only: re-read saved test_scored.parquet (plus test.parquet for "
        "dti, not stored in test_scored.parquet) to check whether v1's calibration gap differs "
        "for loans above its own train-time DTI cap (34.99). No model, feature, or policy change.",
    )
    dti = pd.read_parquet(TEST_PARQUET, columns=["id", "dti"])
    df = scored.merge(dti, on="id", how="left")
    assert df["dti"].isna().sum() == 0

    rows = []
    rows.append(bucket_stats(df[df["dti"] <= V1_TRAIN_MAX_DTI], "pd_lightgbm", "target", f"DTI <= {V1_TRAIN_MAX_DTI} (within v1's train range)"))
    rows.append(bucket_stats(df[df["dti"] > V1_TRAIN_MAX_DTI], "pd_lightgbm", "target", f"DTI > {V1_TRAIN_MAX_DTI} (above v1's train range)"))
    return pd.DataFrame(rows)


def v2_on_2016_holdout() -> pd.DataFrame:
    population = build_eval_population()
    log_access(
        "dti_cap_extrapolation.py",
        len(population),
        "Post-hoc, descriptive only: re-scores the served v2 model on the 2016 holdout "
        "(already used for the pre-registered decision rule) to check whether its calibration "
        "gap differs for loans above its own train-time DTI cap (39.99). No model, feature, "
        "or policy change.",
    )
    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))
    assert champion_feats == cfg["features"]
    population["pd_v2"] = score_lightgbm(population, cfg)

    ref_df = population[population["dti"] <= 35]
    mid_df = population[(population["dti"] > 35) & (population["dti"] <= 40)]
    high_df = population[population["dti"] > 40]

    rows = []
    rows.append(bucket_stats(ref_df, "pd_v2", "label_24m", "DTI <= 35 (reference)"))
    rows.append(bucket_stats(mid_df, "pd_v2", "label_24m", "35 < DTI <= 40"))
    rows.append(bucket_stats(high_df, "pd_v2", "label_24m", f"DTI > 40 (above v2's train range, {V2_TRAIN_MAX_DTI})"))
    bucket_table = pd.DataFrame(rows)

    diffs = [
        bootstrap_gap_difference(ref_df, mid_df, "pd_v2", "label_24m", "DTI <= 35", "35 < DTI <= 40"),
        bootstrap_gap_difference(ref_df, high_df, "pd_v2", "label_24m", "DTI <= 35", "DTI > 40"),
    ]
    diff_table = pd.DataFrame(diffs)
    return bucket_table, diff_table


def main() -> None:
    print("=== v1 on test (2015): DTI <= 34.99 vs. above ===")
    v1_table = v1_on_test()
    v1_table.to_csv(METRICS_DIR / "dti_cap_extrapolation_v1_test.csv", index=False)
    print(v1_table.to_string(index=False))

    print("\n=== v2 on 2016 holdout (24-month label): DTI buckets ===")
    print("(absolute gaps are biased negative by the lifetime-vs-24-month horizon mismatch, not comparable across buckets)")
    v2_table, v2_diff_table = v2_on_2016_holdout()
    v2_table.to_csv(METRICS_DIR / "dti_cap_extrapolation_v2_2016.csv", index=False)
    print(v2_table.to_string(index=False))

    print("\n=== v2: each bucket's gap relative to the in-range (DTI <= 35) bucket, bootstrap 95% CI ===")
    v2_diff_table.to_csv(METRICS_DIR / "dti_cap_extrapolation_v2_2016_relative.csv", index=False)
    print(v2_diff_table.to_string(index=False))

    summary = {
        "label": "post-hoc, descriptive only - no model, feature, or policy change",
        "v1_train_max_dti": V1_TRAIN_MAX_DTI,
        "v2_train_max_dti": V2_TRAIN_MAX_DTI,
        "v1_on_test": v1_table.to_dict("records"),
        "v2_on_2016_holdout_absolute_gaps_not_cross_bucket_comparable": v2_table.to_dict("records"),
        "v2_on_2016_holdout_relative_to_in_range": v2_diff_table.to_dict("records"),
    }
    with open(METRICS_DIR / "dti_cap_extrapolation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {METRICS_DIR / 'dti_cap_extrapolation_summary.json'}")


if __name__ == "__main__":
    main()
