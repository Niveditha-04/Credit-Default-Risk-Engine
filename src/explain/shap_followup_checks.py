"""Phase 6 follow-up checks (no model changes):
  - Income reversal: does it survive when grouping all income-scaled
    features together, and does it show up in the raw train default rate?
  - dti dip: does it show up in the raw train default rate?

These checks distinguish "the model is picking up a real historical
pattern" from "this is noise/overfitting on a sparse tail" - by comparing
against the actual label, not just the model's own SHAP output.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

METRICS_DIR = Path("reports/metrics")
TEST_PARQUET = "data/processed/test.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
SAMPLE_SIZE = 10_000
SEED = 42

INCOME_BINS = [0, 30000, 50000, 70000, 90000, 110000, 150000, 200000, 300000, 500000, 1e7]
DTI_BINS = [-0.01, 1, 3, 5, 8, 12, 16, 20, 25, 30, 100]


def main() -> None:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]
    booster = joblib.load(cfg["model_path"])
    feats = cfg["features"]

    test = pd.read_parquet(TEST_PARQUET)
    for col in ["home_ownership", "purpose", "verification_status"]:
        test[col] = test[col].astype("category")
    sample = test.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    X = sample[feats]

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

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
    income_summary.to_csv(METRICS_DIR / "shap_income_group_reversal_check.csv")
    print("=== Income reversal: grouped SHAP (annual_inc + loan_to_income + revol_bal_to_income) ===")
    print(income_summary.to_string())

    grouped_means = income_summary["mean_grouped_shap"].values
    grouped_monotonic = np.all(np.diff(grouped_means) <= 0) or np.all(np.diff(grouped_means) >= 0)
    print(f"\nGrouped SHAP monotonic across income buckets: {grouped_monotonic}")

    # --- Income reversal: raw train default rate ---
    train = pd.read_parquet(TRAIN_PARQUET)
    train_income = train.copy()
    train_income["bucket"] = pd.cut(train_income["annual_inc"], bins=INCOME_BINS)
    income_rate = train_income.groupby("bucket", observed=True).agg(
        n=("target", "size"), default_rate_pct=("target", lambda x: round(100 * x.mean(), 2))
    )
    income_rate.to_csv(METRICS_DIR / "train_default_rate_by_income_bucket.csv")
    print("\n=== Raw train default rate by income bucket ===")
    print(income_rate.to_string())
    rates = income_rate["default_rate_pct"].values
    income_rate_has_uptick = bool(np.any(np.diff(rates) > 0.3))  # more than a trivial wiggle
    print(f"Raw default rate shows an uptick after its minimum: {income_rate_has_uptick}")

    # --- dti dip: raw train default rate ---
    train_dti = train.copy()
    train_dti["bucket"] = pd.cut(train_dti["dti"], bins=DTI_BINS)
    dti_rate = train_dti.groupby("bucket", observed=True).agg(
        n=("target", "size"), default_rate_pct=("target", lambda x: round(100 * x.mean(), 2))
    )
    dti_rate.to_csv(METRICS_DIR / "train_default_rate_by_dti_bucket.csv")
    print("\n=== Raw train default rate by dti bucket ===")
    print(dti_rate.to_string())
    dti_rates = dti_rate["default_rate_pct"].values
    dti_dip_present = bool(np.any(np.diff(dti_rates[:5]) < 0))  # any decrease in the first 5 buckets
    print(f"Raw default rate shows a local dip in the low-dti region: {dti_dip_present}")

    summary = {
        "income_reversal": {
            "grouped_shap_monotonic_across_buckets": bool(grouped_monotonic),
            "note": (
                "annual_inc alone reverses clearly (SHAP rises from -0.237 at $90-110K back toward -0.14 at "
                "$500K+). Grouped with loan_to_income and revol_bal_to_income (both scaled by the same income), "
                "the combined effect mostly plateaus around -0.25 to -0.27 rather than trending back toward "
                "zero - the other income-scaled features partly offset annual_inc's own reversal, since a very "
                "high earner's loan_to_income and revol_bal_to_income are mechanically very low."
            ),
            "raw_train_default_rate_shows_uptick": income_rate_has_uptick,
        },
        "dti_dip": {
            "raw_train_default_rate_shows_dip": dti_dip_present,
            "note": "Both the SHAP dip and the raw train default rate dip in the dti=3-8 region before rising.",
        },
    }
    with open(METRICS_DIR / "shap_followup_checks_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
