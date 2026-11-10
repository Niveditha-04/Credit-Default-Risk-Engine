"""Phase 6: explain the FROZEN LightGBM champion only. No model changes.

SHAP TreeExplainer on a seeded 10,000-row test sample. TreeExplainer's
output here is on the raw model's log-odds (margin) scale, verified by
matching booster.predict(..., raw_score=True) exactly (additivity check
below). The champion's actual deployed output passes this log-odds value
through a Platt calibrator, which is a strictly monotonic transform - it
rescales every log-odds value through the same increasing function, so it
can shift the numeric SHAP values but cannot change which direction any
feature pushes a prediction, or the RANK order of driver importance for a
given applicant. All SHAP analysis below is therefore done in log-odds
space and its directional conclusions (positive/negative, feature
ranking) carry over unchanged to the calibrated probability the model
actually serves.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explain.labels import PLAIN_ENGLISH

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_PARQUET = "data/processed/test.parquet"
SAMPLE_SIZE = 10_000
SEED = 42
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

ROLLOUT_FLAG_COLUMNS = [
    "never_flag__mths_since_recent_bc_dlq",
    "never_flag__mths_since_last_major_derog",
    "never_flag__mths_since_recent_inq",
    "never_flag__mths_since_recent_revol_delinq",
]


def nan_safe_corr(values: np.ndarray, shap_vals: np.ndarray) -> float:
    """Pearson correlation on the rows where the feature is non-null.
    np.corrcoef propagates a single NaN into the whole result, which
    silently breaks for any feature with missing values - several of the
    rollout-era and bankcard features have a small residual missing rate
    even on test (see Phase 2/4 missingness notes)."""
    mask = ~pd.isna(values)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(values)[mask], shap_vals[mask])[0, 1])


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]
    feats = cfg["features"]

    test = pd.read_parquet(TEST_PARQUET)
    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")
    sample = test.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    return sample, sample[feats], feats


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    booster = joblib.load(frozen["lightgbm_chosen"]["model_path"])

    sample, X, feats = load_sample()
    print(f"Sample: {len(sample)} rows (seed={SEED}), {len(feats)} features")

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value

    # --- Additivity check ---
    raw_pred = booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)
    reconstructed = shap_values.sum(axis=1) + base_value
    max_diff = float(np.max(np.abs(reconstructed - raw_pred)))
    additivity_ok = max_diff < 1e-6
    print(f"\n=== Additivity check: max |SHAP sum + base - raw output| = {max_diff:.2e} "
          f"(tolerance 1e-6: {'PASS' if additivity_ok else 'FAIL'}) ===")
    assert additivity_ok, "Additivity check failed"

    # --- Global importance ---
    print("\n=== Global importance: mean |SHAP|, top 15 ===")
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
    gain_importance = pd.read_csv(METRICS_DIR / "gbm_feature_importance.csv").set_index("feature")["gain"]

    global_importance = pd.DataFrame({"mean_abs_shap": mean_abs_shap})
    global_importance["mean_abs_shap_rank"] = range(1, len(global_importance) + 1)
    global_importance["gain"] = gain_importance.reindex(global_importance.index)
    global_importance["gain_rank"] = global_importance["gain"].rank(ascending=False).astype(int)
    global_importance.to_csv(METRICS_DIR / "shap_global_importance.csv")
    print(global_importance.head(15).to_string())

    top6 = mean_abs_shap.head(6).index.tolist()
    print(f"\nTop 6 for dependence plots: {top6}")

    # --- Dependence plots + direction check ---
    print("\n=== Dependence check: direction per feature ===")
    direction_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, feat in zip(axes.flat, top6):
        vals = X[feat]
        sv = shap_values[:, feats.index(feat)]
        if pd.api.types.is_numeric_dtype(vals):
            corr = nan_safe_corr(vals, sv)
            ax.scatter(vals, sv, s=4, alpha=0.3, color="#4C72B0")
            # annual_inc is heavily right-skewed (a handful of very high
            # earners stretch the axis); log scale makes the shape legible
            # without changing any underlying SHAP value.
            if vals.min() > 0 and (vals.max() / vals.median()) > 20:
                ax.set_xscale("log")
            direction = "positive (higher value -> higher predicted risk)" if corr > 0 else "negative (higher value -> lower predicted risk)"
        else:
            corr = np.nan
            cats = vals.astype(str)
            for i, cat in enumerate(sorted(cats.unique())):
                ax.scatter(np.full((cats == cat).sum(), i), sv[cats == cat], s=4, alpha=0.3)
            ax.set_xticks(range(len(sorted(cats.unique()))))
            ax.set_xticklabels(sorted(cats.unique()), rotation=45, ha="right", fontsize=8)
            direction = "categorical - see plot"
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value (log-odds)")
        direction_rows.append({"feature": feat, "pearson_corr_value_vs_shap": round(float(corr), 3) if not np.isnan(corr) else None, "direction": direction})
    fig.suptitle("SHAP dependence, top 6 features (log-odds scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_dependence_top6.png", dpi=150)
    plt.close(fig)

    direction_df = pd.DataFrame(direction_rows)
    print(direction_df.to_string(index=False))

    expected_directions = {
        "fico_mean": "negative", "dti": "positive", "loan_to_income": "positive", "inq_last_6mths": "positive",
    }
    direction_flags = []
    for _, row in direction_df.iterrows():
        feat = row["feature"]
        if feat in expected_directions and row["pearson_corr_value_vs_shap"] is not None:
            actual = "positive" if row["pearson_corr_value_vs_shap"] > 0 else "negative"
            if actual != expected_directions[feat]:
                direction_flags.append(f"{feat}: expected {expected_directions[feat]}, found {actual}")
    print(f"\nCounterintuitive vs the stated expectations: {direction_flags or 'none'}")
    direction_df.to_csv(METRICS_DIR / "shap_dependence_direction_top6.csv", index=False)

    # --- Local non-monotonicity check: a simple linear correlation can
    # miss a dip or reversal that still exists within an overall trend.
    # Bucket means for the two features where the scatter plot showed one.
    print("\n=== Local non-monotonicity: bucketed mean SHAP ===")
    non_monotonic_notes = []
    bucket_defs = {
        "dti": [-0.01, 1, 3, 5, 8, 12, 16, 20, 25, 30, 100],
        "annual_inc": [0, 30000, 50000, 70000, 90000, 110000, 150000, 200000, 300000, 500000, 1e7],
    }
    for feat, bins in bucket_defs.items():
        if feat not in feats:
            continue
        sv = shap_values[:, feats.index(feat)]
        bdf = pd.DataFrame({feat: X[feat].values, "shap": sv})
        bdf["bucket"] = pd.cut(bdf[feat], bins=bins)
        bucket_means = bdf.groupby("bucket", observed=True)["shap"].agg(["mean", "count"])
        bucket_means.to_csv(METRICS_DIR / f"shap_bucket_means_{feat}.csv")
        print(f"\n{feat}:")
        print(bucket_means.to_string())
        means = bucket_means["mean"].values
        is_monotonic = np.all(np.diff(means) >= 0) or np.all(np.diff(means) <= 0)
        if not is_monotonic:
            non_monotonic_notes.append(feat)
    print(f"\nFeatures with a local (bucket-level) non-monotonic pattern: {non_monotonic_notes or 'none'}")

    # --- SHAP vs scorecard WOE direction, shared features ---
    print("\n=== SHAP direction vs scorecard WOE direction, shared features ===")
    coef_df = pd.read_csv(METRICS_DIR / "logistic_regression_coefficients.csv")
    final_scorecard_feats = set(coef_df["feature"])
    shared_final = [f for f in feats if f in final_scorecard_feats]
    print(f"Shared with scorecard's final 17 features: {shared_final}")

    binning_process = joblib.load(frozen["scorecard"]["binning_process_path"])

    def woe_direction(feat: str) -> tuple[str, float | None]:
        """Correlation between bin order and WoE for the ordered interval
        bins only (excludes Special/Missing/Totals). Positive correlation
        means WoE rises as the feature value rises, i.e. the convention
        established in Phase 3 (higher WoE = lower event rate) implies
        higher feature value -> lower risk."""
        table = binning_process.get_binned_variable(feat).binning_table.build()
        # "Totals" row has Bin="" (empty string, not NaN), which silently
        # casts the whole WoE column to object dtype - filter by row
        # position (exclude the last "Totals" row) rather than by value,
        # and coerce WoE to numeric only after that.
        ordered = table.iloc[:-1]
        # Special/Missing rows always have a plain string Bin value;
        # interval/category bins have a string (numeric) or an array-like
        # (categorical) - check type directly rather than .isin(), which
        # breaks on array-valued cells.
        is_special_or_missing = ordered["Bin"].apply(lambda b: isinstance(b, str) and b in ("Special", "Missing"))
        ordered = ordered[~is_special_or_missing]
        woe_numeric = pd.to_numeric(ordered["WoE"], errors="coerce")
        if len(ordered) < 2 or woe_numeric.isna().any():
            return "not determinable (categorical or too few bins)", None
        bin_rank = np.arange(len(ordered))
        corr = float(np.corrcoef(bin_rank, woe_numeric.values)[0, 1])
        direction = "higher value -> lower risk" if corr > 0 else "higher value -> higher risk"
        return direction, corr

    woe_comparison_rows = []
    disagreements = []
    for feat in shared_final:
        if feat not in binning_process.variable_names or not pd.api.types.is_numeric_dtype(X[feat]):
            continue
        sv = shap_values[:, feats.index(feat)]
        shap_corr = nan_safe_corr(X[feat], sv)
        shap_direction = "higher value -> higher risk" if shap_corr > 0 else "higher value -> lower risk"
        w_direction, w_corr = woe_direction(feat)
        agrees = (w_corr is not None) and (shap_direction == w_direction)
        woe_comparison_rows.append({
            "feature": feat, "shap_direction": shap_direction, "shap_corr": round(float(shap_corr), 3),
            "woe_direction": w_direction, "woe_bin_rank_corr": round(w_corr, 3) if w_corr is not None else None,
            "agrees": agrees,
        })
        if w_corr is not None and not agrees:
            disagreements.append(feat)
    woe_df = pd.DataFrame(woe_comparison_rows)
    woe_df.to_csv(METRICS_DIR / "shap_vs_woe_direction.csv", index=False)
    print(woe_df.to_string(index=False))
    print(f"\nDisagreements: {disagreements or 'none'}")

    # --- Per-applicant reason codes ---
    print("\n=== Per-applicant illustrative reason codes (NOT legal adverse action notices) ===")
    from src.evaluation.score_models import score_lightgbm

    calibrated_pd = score_lightgbm(sample, frozen["lightgbm_chosen"])
    percentiles = [5, 25, 50, 75, 95]
    reason_codes = []
    for p in percentiles:
        target_pd = np.percentile(calibrated_pd, p)
        idx = int(np.argmin(np.abs(calibrated_pd - target_pd)))
        row_shap = shap_values[idx]
        top3_idx = np.argsort(row_shap)[::-1][:3]  # most risk-increasing
        drivers = []
        for i in top3_idx:
            feat = feats[i]
            label = PLAIN_ENGLISH.get(feat, feat)
            drivers.append({"feature": feat, "plain_english": label, "shap_value_log_odds": round(float(row_shap[i]), 4),
                             "feature_value": str(X.iloc[idx][feat])})
        reason_codes.append({
            "percentile": p, "calibrated_pd_pct": round(float(calibrated_pd[idx]) * 100, 2),
            "top_3_risk_increasing_drivers_illustrative_only": drivers,
        })
    reason_codes_output = {
        "pd_note": (
            "calibrated_pd_pct is the CALIBRATED probability (raw LightGBM log-odds passed through the "
            "frozen Platt calibrator, via score_lightgbm() - the exact same function used everywhere else "
            "in this project to score a loan). It is NOT the raw/uncalibrated model output. The SHAP values "
            "and 'drivers' themselves are on the raw log-odds scale (see module docstring); only the PD shown "
            "per applicant is calibrated."
        ),
        "examples": reason_codes,
    }
    with open(METRICS_DIR / "shap_reason_codes_examples.json", "w") as f:
        json.dump(reason_codes_output, f, indent=2)
    print(json.dumps(reason_codes_output, indent=2))

    # --- Missing values in rollout-era features ---
    print("\n=== Missing values in rollout-era features, test sample ===")
    missing_check = {}
    for col in ROLLOUT_FLAG_COLUMNS:
        if col in X.columns:
            missing_check[col] = int(X[col].isna().sum())
    print(json.dumps(missing_check, indent=2))
    with open(METRICS_DIR / "shap_rollout_missing_check.json", "w") as f:
        json.dump(missing_check, f, indent=2)

    summary = {
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "shap_scale": "raw model log-odds (margin), not calibrated probability",
        "additivity_max_abs_diff": max_diff,
        "additivity_tolerance": 1e-6,
        "additivity_check_passed": additivity_ok,
        "counterintuitive_directions": direction_flags,
        "local_non_monotonic_features": non_monotonic_notes,
        "shap_vs_woe_disagreements": disagreements,
        "rollout_feature_missing_count_in_sample": missing_check,
    }
    with open(METRICS_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
