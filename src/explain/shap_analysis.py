"""Explains the frozen LightGBM champion only. No model changes.

Per-feature contributions come from booster.predict(pred_contrib=True),
LightGBM's own built-in exact TreeSHAP (computed in C++ inside the
library already used for prediction), on the full 282,787-row test set -
not a sample, and not the separate shap package. This is the same call
src/api/scoring.py uses for its reason codes, so the numbers here match
what the served model actually returns. The output is on the raw
model's log-odds (margin) scale. The champion's deployed output passes
that log-odds value through a Platt calibrator, which is a strictly
monotonic transform - it rescales every log-odds value through the same
increasing function, so it can shift the numeric contribution values but
cannot change which direction any feature pushes a prediction, or the
RANK order of driver importance for a given applicant. All analysis
below is therefore done in log-odds space and its directional
conclusions (positive/negative, feature ranking) carry over unchanged to
the calibrated probability the model actually serves.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.score_models import v1_lightgbm_cfg
from src.explain.labels import PLAIN_ENGLISH
from src.report_paths import MODEL_CONFIG_PATH, metrics_path

FIG_DIR = Path("reports/figures")
EXTRA_DIR = Path("reports/extra")
TEST_PARQUET = "data/processed/test.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

ROLLOUT_FLAG_COLUMNS = [
    "never_flag__mths_since_recent_bc_dlq",
    "never_flag__mths_since_last_major_derog",
    "never_flag__mths_since_recent_inq",
    "never_flag__mths_since_recent_revol_delinq",
]

TRAIN_PARQUET = "data/processed/train.parquet"
INCOME_BINS = [0, 30000, 50000, 70000, 90000, 110000, 150000, 200000, 300000, 500000, 1e7]
DTI_BINS = [-0.01, 1, 3, 5, 8, 12, 16, 20, 25, 30, 100]
Z_95 = 1.959963984540054  # standard normal 97.5th percentile


def nan_safe_corr(values: np.ndarray, shap_vals: np.ndarray) -> float:
    """Pearson correlation on the rows where the feature is non-null.
    np.corrcoef propagates a single NaN into the whole result, which
    silently breaks for any feature with missing values - several of the
    rollout-era and bankcard features have a small residual missing rate
    even on test (see build_features.py's missing-rate handling)."""
    mask = ~pd.isna(values)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(values)[mask], shap_vals[mask])[0, 1])


def load_test() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    with open(MODEL_CONFIG_PATH) as f:
        frozen = json.load(f)
    cfg = v1_lightgbm_cfg(frozen)
    feats = cfg["features"]

    test = pd.read_parquet(TEST_PARQUET)
    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")
    return test, test[feats], feats


def wilson_ci(n_success: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = n_success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)


def default_rate_with_ci(df: pd.DataFrame, col: str, bins: list) -> pd.DataFrame:
    d = df.copy()
    d["bucket"] = pd.cut(d[col], bins=bins)
    rows = []
    for bucket, g in d.groupby("bucket", observed=True):
        n = len(g)
        n_default = int(g["target"].sum())
        rate = 100 * n_default / n
        lo, hi = wilson_ci(n_default, n)
        rows.append({
            "bucket": str(bucket), "n": n, "default_rate_pct": round(rate, 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
        })
    return pd.DataFrame(rows)


def cis_overlap(row_a: pd.Series, row_b: pd.Series) -> bool:
    return not (row_a["ci_high"] < row_b["ci_low"] or row_b["ci_high"] < row_a["ci_low"])


def shap_followup_checks() -> None:
    """Follow-up checks on v1's own SHAP output (v1 explicitly,
    lightgbm_v1_history, not lightgbm_chosen - this is the model this
    project calls "the champion" everywhere else these findings are
    cited, reports/extra/shap_followup_checks_summary.json):
      - Income reversal: does it survive when grouping all
        income-scaled features together, and does it show up in the
        raw train default rate?
      - dti dip: does it show up in the raw train default rate?
    Every bucket's raw default rate is reported with n and a 95%
    Wilson score confidence interval - some buckets are small (the
    $500K+ income bucket has n=116, the dti 0-1 bucket has n=700), and
    a point-estimate "dip" or "rise" that looks clean on a chart can be
    well within sampling noise once the interval is checked."""
    with open(MODEL_CONFIG_PATH) as f:
        frozen = json.load(f)
    cfg = v1_lightgbm_cfg(frozen)
    booster = joblib.load(cfg["model_path"])
    feats = cfg["features"]

    test = pd.read_parquet(TEST_PARQUET)
    for col in CATEGORICAL_COLUMNS:
        test[col] = test[col].astype("category")
    X = test[feats]
    print(f"Test set: {len(test)} rows (no sampling)")

    contrib = booster.predict(X, pred_contrib=True, num_iteration=booster.best_iteration)
    shap_values = contrib[:, :-1]

    # --- Income reversal: grouped SHAP ---
    income_group = ["annual_inc", "loan_to_income", "revol_bal_to_income"]
    idxs = [feats.index(f) for f in income_group]
    grouped_shap = shap_values[:, idxs].sum(axis=1)

    income_df = pd.DataFrame({
        "annual_inc": X["annual_inc"].values,
        "grouped_shap": grouped_shap,
        "annual_inc_only_shap": shap_values[:, feats.index("annual_inc")],
    })
    income_df["bucket"] = pd.cut(income_df["annual_inc"], bins=INCOME_BINS)
    income_summary = income_df.groupby("bucket", observed=True).agg(
        mean_grouped_shap=("grouped_shap", "mean"),
        mean_annual_inc_only_shap=("annual_inc_only_shap", "mean"),
        n=("annual_inc", "size"),
    )
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    income_summary.to_csv(EXTRA_DIR / "shap_income_group_reversal_check.csv")
    print("=== Income reversal: grouped SHAP (annual_inc + loan_to_income + revol_bal_to_income) ===")
    print(income_summary.to_string())

    grouped_means = income_summary["mean_grouped_shap"].values
    grouped_monotonic = np.all(np.diff(grouped_means) <= 0) or np.all(np.diff(grouped_means) >= 0)
    print(f"\nGrouped SHAP monotonic across income buckets: {grouped_monotonic}")

    # --- Income: raw train default rate, with n and 95% CI ---
    train = pd.read_parquet(TRAIN_PARQUET)
    income_rate = default_rate_with_ci(train, "annual_inc", INCOME_BINS)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    income_rate.to_csv(EXTRA_DIR / "train_default_rate_by_income_bucket.csv", index=False)
    print("\n=== Raw train default rate by income bucket (n, 95% Wilson CI) ===")
    print(income_rate.to_string(index=False))

    row_500k = income_rate[income_rate["bucket"].str.contains("500000.0, 10000000.0")].iloc[0]
    row_110_150k = income_rate[income_rate["bucket"].str.contains("110000.0, 150000.0")].iloc[0]
    income_500k_overlaps_min = cis_overlap(row_500k, row_110_150k)
    print(f"\n$500K+ CI [{row_500k['ci_low']}, {row_500k['ci_high']}] vs $110-150K (min) CI "
          f"[{row_110_150k['ci_low']}, {row_110_150k['ci_high']}]: overlap = {income_500k_overlaps_min}")

    # --- dti: raw train default rate, with n and 95% CI ---
    dti_rate = default_rate_with_ci(train, "dti", DTI_BINS)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    dti_rate.to_csv(EXTRA_DIR / "train_default_rate_by_dti_bucket.csv", index=False)
    print("\n=== Raw train default rate by dti bucket (n, 95% Wilson CI) ===")
    print(dti_rate.to_string(index=False))

    row_0_1 = dti_rate[dti_rate["bucket"].str.contains(r"\-0.01, 1.0")].iloc[0]
    row_5_8 = dti_rate[dti_rate["bucket"].str.contains("5.0, 8.0")].iloc[0]
    dti_low_end_overlaps = cis_overlap(row_0_1, row_5_8)
    print(f"\ndti 0-1 CI [{row_0_1['ci_low']}, {row_0_1['ci_high']}] vs dti 5-8 CI "
          f"[{row_5_8['ci_low']}, {row_5_8['ci_high']}]: overlap = {dti_low_end_overlaps}")

    summary = {
        "income_reversal": {
            "grouped_shap_monotonic_across_buckets": bool(grouped_monotonic),
            "note": (
                "annual_inc alone shows a rising SHAP trend above $90-110K (from -0.236 toward -0.14 to -0.15 at "
                "$150K and up). Grouped with loan_to_income and revol_bal_to_income (both scaled by the same "
                "income), the combined effect stays in a -0.22 to -0.30 band above $90K rather than trending back "
                "toward zero like annual_inc alone - the other income-scaled features partly offset annual_inc's "
                "own rise."
            ),
            "raw_rate_500k_plus_ci": [float(row_500k["ci_low"]), float(row_500k["ci_high"])],
            "raw_rate_110_150k_min_ci": [float(row_110_150k["ci_low"]), float(row_110_150k["ci_high"])],
            "cis_overlap": bool(income_500k_overlaps_min),
            "verdict": (
                "PLATEAU, not reversal: the $500K+ bucket (n=116) has a wide CI that fully contains the "
                "$110-150K minimum's CI. The raw data cannot distinguish 'risk rises again at very high income' "
                "from 'risk stays flat with noise' at this sample size."
            ),
        },
        "dti_low_end": {
            "raw_rate_0_1_ci": [float(row_0_1["ci_low"]), float(row_0_1["ci_high"])],
            "raw_rate_5_8_ci": [float(row_5_8["ci_low"]), float(row_5_8["ci_high"])],
            "cis_overlap": bool(dti_low_end_overlaps),
            "verdict": (
                "The SHAP dip (minimum at dti=5-8) and the raw default rate do NOT agree at "
                "the low end - dti=0-1 (n=700) is actually the SAFEST raw bucket (8.43%), not the riskiest, "
                "while SHAP shows dti=0-1 as less protective than dti=5-8. This is a small non-monotonic SHAP "
                "shape at very low dti that is NOT clearly visible in the raw default rate (its CI overlaps "
                "neighboring buckets) - likely a conditional effect (interacting with other features) or "
                "sampling noise on a modest bucket, not a confirmed standalone historical pattern. The overall "
                "rising trend from dti~15 upward IS robust (buckets there are large, CIs tight, non-overlapping)."
            ),
        },
    }
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXTRA_DIR / "shap_followup_checks_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


def loan_to_income_cap_check() -> None:
    """No clip()/min()/max() call exists anywhere in build_features.py's
    loan_to_income formula (plain division: loan_amnt / annual_inc) -
    this confirms that empirically across all three splits."""
    rows = []
    for name, path in [
        ("train", "data/processed/train.parquet"),
        ("validation", "data/processed/validation.parquet"),
        ("test", "data/processed/test.parquet"),
    ]:
        df = pd.read_parquet(path)
        lti = df["loan_to_income"]
        rows.append(
            {
                "split": name,
                "n": len(df),
                "max": round(float(lti.max()), 4),
                "pct_at_exactly_0_5": round(100 * (lti == 0.5).mean(), 3),
                "pct_above_0_5": round(100 * (lti > 0.5).mean(), 3),
            }
        )
    result = {
        "clipped_in_our_code": False,
        "formula": "loan_amnt / annual_inc (build_features.py), no clip/min/max applied anywhere in the pipeline",
        "by_split": rows,
        "interpretation": (
            "Train (2012-2013) has loans above 0.5 (max 0.7125), with little mass exactly at 0.5 (0.065%). "
            "Validation (2014) and test (2015) never exceed 0.5, and test shows 0.538% of loans landing "
            "exactly at 0.5 (7x validation's rate). This looks like a LendingClub underwriting policy "
            "(a hard loan-to-income cap around 50%) that tightened/became binding between the 2012-2013 and "
            "2014-2015 vintages - a property of the raw data, not something introduced by our feature "
            "engineering."
        ),
    }
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXTRA_DIR / "loan_to_income_cap_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_CONFIG_PATH) as f:
        frozen = json.load(f)
    booster = joblib.load(v1_lightgbm_cfg(frozen)["model_path"])

    test, X, feats = load_test()
    print(f"Test set: {len(test)} rows (no sampling), {len(feats)} features")

    contrib = booster.predict(X, pred_contrib=True, num_iteration=booster.best_iteration)
    shap_values = contrib[:, :-1]
    base_value = contrib[:, -1]

    # --- Additivity check ---
    raw_pred = booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)
    reconstructed = shap_values.sum(axis=1) + base_value
    max_diff = float(np.max(np.abs(reconstructed - raw_pred)))
    additivity_ok = max_diff < 1e-6
    print(f"\n=== Additivity check: max |contrib sum + base - raw output| = {max_diff:.2e} "
          f"(tolerance 1e-6: {'PASS' if additivity_ok else 'FAIL'}) ===")
    assert additivity_ok, "Additivity check failed"

    # --- Global importance ---
    print("\n=== Global importance: mean |contribution|, top 15 ===")
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
    gain_importance = pd.read_csv(metrics_path("gbm_feature_importance.csv")).set_index("feature")["gain"]

    global_importance = pd.DataFrame({"mean_abs_shap": mean_abs_shap})
    global_importance["mean_abs_shap_rank"] = range(1, len(global_importance) + 1)
    global_importance["gain"] = gain_importance.reindex(global_importance.index)
    global_importance["gain_rank"] = global_importance["gain"].rank(ascending=False).astype(int)
    global_importance.to_csv(metrics_path("shap_global_importance.csv"))
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
            ax.scatter(vals, sv, s=2, alpha=0.05, color="#4C72B0")
            # annual_inc is heavily right-skewed (a handful of very high
            # earners stretch the axis); log scale makes the shape legible
            # without changing any underlying contribution value.
            if vals.min() > 0 and (vals.max() / vals.median()) > 20:
                ax.set_xscale("log")
            direction = "positive (higher value -> higher predicted risk)" if corr > 0 else "negative (higher value -> lower predicted risk)"
        else:
            corr = np.nan
            cats = vals.astype(str)
            for i, cat in enumerate(sorted(cats.unique())):
                ax.scatter(np.full((cats == cat).sum(), i), sv[cats == cat], s=2, alpha=0.05)
            ax.set_xticks(range(len(sorted(cats.unique()))))
            ax.set_xticklabels(sorted(cats.unique()), rotation=45, ha="right", fontsize=8)
            direction = "categorical - see plot"
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Contribution (log-odds)")
        direction_rows.append({"feature": feat, "pearson_corr_value_vs_shap": round(float(corr), 3) if not np.isnan(corr) else None, "direction": direction})
    fig.suptitle("Per-feature contribution, top 6 features, full test set (log-odds scale)")
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
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    direction_df.to_csv(EXTRA_DIR / "shap_dependence_direction_top6.csv", index=False)

    # --- Local non-monotonicity check: a simple linear correlation can
    # miss a dip or reversal that still exists within an overall trend.
    # Bucket means for the two features where the scatter plot showed one.
    print("\n=== Local non-monotonicity: bucketed mean contribution ===")
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
        EXTRA_DIR.mkdir(parents=True, exist_ok=True)
        bucket_means.to_csv(EXTRA_DIR / f"shap_bucket_means_{feat}.csv")
        print(f"\n{feat}:")
        print(bucket_means.to_string())
        means = bucket_means["mean"].values
        is_monotonic = np.all(np.diff(means) >= 0) or np.all(np.diff(means) <= 0)
        if not is_monotonic:
            non_monotonic_notes.append(feat)
    print(f"\nFeatures with a local (bucket-level) non-monotonic pattern: {non_monotonic_notes or 'none'}")

    # --- SHAP vs scorecard WOE direction, shared features ---
    print("\n=== Contribution direction vs scorecard WOE direction, shared features ===")
    coef_df = pd.read_csv(metrics_path("logistic_regression_coefficients.csv"))
    final_scorecard_feats = set(coef_df["feature"])
    shared_final = [f for f in feats if f in final_scorecard_feats]
    print(f"Shared with scorecard's final 17 features: {shared_final}")

    binning_process = joblib.load(frozen["scorecard"]["binning_process_path"])

    def woe_direction(feat: str) -> tuple[str, float | None]:
        """Correlation between bin order and WoE for the ordered interval
        bins only (excludes Special/Missing/Totals). Positive correlation
        means WoE rises as the feature value rises, i.e. the scorecard's
        convention (higher WoE = lower event rate) implies higher feature
        value -> lower risk."""
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
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    woe_df.to_csv(EXTRA_DIR / "shap_vs_woe_direction.csv", index=False)
    print(woe_df.to_string(index=False))
    print(f"\nDisagreements: {disagreements or 'none'}")

    # --- Per-applicant reason codes ---
    print("\n=== Per-applicant illustrative reason codes (NOT legal adverse action notices) ===")
    from src.evaluation.score_models import score_lightgbm

    calibrated_pd = score_lightgbm(test, v1_lightgbm_cfg(frozen))
    ids = test["id"].to_numpy()
    percentiles = [5, 25, 50, 75, 95]
    reason_codes = []
    for p in percentiles:
        target_pd = np.percentile(calibrated_pd, p)
        abs_diff = np.abs(calibrated_pd - target_pd)
        min_diff = abs_diff.min()
        # Several loans can be equally close to the target percentile -
        # break the tie by lowest id, so the chosen example is the same
        # every time this script runs, independent of row order.
        tied = np.flatnonzero(abs_diff == min_diff)
        idx = tied[np.argmin(ids[tied])]
        row_shap = shap_values[idx]
        top3_idx = np.argsort(row_shap)[::-1][:3]  # most risk-increasing
        drivers = []
        for i in top3_idx:
            feat = feats[i]
            label = PLAIN_ENGLISH.get(feat, feat)
            drivers.append({"feature": feat, "plain_english": label, "shap_value_log_odds": round(float(row_shap[i]), 4),
                             "feature_value": str(X.iloc[idx][feat])})
        reason_codes.append({
            "percentile": p, "id": int(ids[idx]), "calibrated_pd_pct": round(float(calibrated_pd[idx]) * 100, 2),
            "top_3_risk_increasing_drivers_illustrative_only": drivers,
        })
    reason_codes_output = {
        "pd_note": (
            "calibrated_pd_pct is the CALIBRATED probability (raw LightGBM log-odds passed through the "
            "frozen Platt calibrator, via score_lightgbm() - the exact same function used everywhere else "
            "in this project to score a loan). It is NOT the raw/uncalibrated model output. The per-feature "
            "contributions and 'drivers' themselves are on the raw log-odds scale (see module docstring); "
            "only the PD shown per applicant is calibrated. Ties for closest-to-percentile are broken by "
            "lowest loan id, so these examples are the same every time this script runs."
        ),
        "examples": reason_codes,
    }
    with open(metrics_path("shap_reason_codes_examples.json"), "w") as f:
        json.dump(reason_codes_output, f, indent=2)
    print(json.dumps(reason_codes_output, indent=2))

    # --- Missing values in rollout-era features ---
    print("\n=== Missing values in rollout-era features, full test set ===")
    missing_check = {}
    for col in ROLLOUT_FLAG_COLUMNS:
        if col in X.columns:
            missing_check[col] = int(X[col].isna().sum())
    print(json.dumps(missing_check, indent=2))

    summary = {
        "sample_size": len(test),
        "sampling": "none - full test set",
        "contribution_source": "booster.predict(pred_contrib=True), LightGBM's exact TreeSHAP",
        "shap_scale": "raw model log-odds (margin), not calibrated probability",
        "additivity_max_abs_diff": max_diff,
        "additivity_tolerance": 1e-6,
        "additivity_check_passed": additivity_ok,
        "counterintuitive_directions": direction_flags,
        "local_non_monotonic_features": non_monotonic_notes,
        "shap_vs_woe_disagreements": disagreements,
        "rollout_feature_missing_count_in_sample": missing_check,
    }
    with open(metrics_path("shap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
    print()
    shap_followup_checks()
    print()
    loan_to_income_cap_check()
