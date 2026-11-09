"""Choose the approval cut-off and every economic assumption on VALIDATION
(2014) and TRAIN (2012-2013) only. This script never reads test (2015).
Run this BEFORE score_test.py, and never change reports/metrics/
decision_policy.json after test has been scored.

LGD (loss given default): estimated from TRAIN-year charged-off 36-month
individual loans only.
    LGD_i = 1 - clip((total_pymnt_i + recoveries_i - collection_recovery_fee_i) / funded_amnt_i, 0, 1)
    LGD   = mean(LGD_i) across that population
Both total_pymnt and funded_amnt are banned model features (post-origination
leakage) but are fine to use here: this is an economics calculation, not a
model input.

Interest income (approximate, for a loan that performs):
    interest_income_i = installment_i * 36 - loan_amnt_i
(the lender's own computed monthly payment over the 36-month term, minus
principal - a standard approximation, not exact amortized interest).

Expected net value per loan at a given PD:
    EV_i = (1 - PD_i) * interest_income_i - PD_i * loan_amnt_i * LGD

Cut-off: swept over the champion's (LightGBM) calibrated PD on validation,
the PD value that maximizes TOTAL expected net value across the approved
population is chosen. That PD value is then applied, unchanged, as a
single fixed threshold to all three models on test.
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.models.gbm import logit

METRICS_DIR = Path("reports/metrics")
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
MODELS_DIR = Path("data/processed/models")

LGD_SENSITIVITY_RANGE = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]


def estimate_lgd_from_train(con: duckdb.DuckDBPyConnection) -> dict:
    df = con.execute(
        f"""
        SELECT funded_amnt, total_pymnt, recoveries, collection_recovery_fee
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
          AND loan_status = 'Charged Off'
        """
    ).fetchdf()
    recovered_fraction = (df["total_pymnt"] + df["recoveries"] - df["collection_recovery_fee"]) / df["funded_amnt"]
    lgd_per_loan = 1 - recovered_fraction.clip(0, 1)
    return {
        "n_charged_off_train_loans": len(df),
        "lgd_mean": round(float(lgd_per_loan.mean()), 4),
        "lgd_median": round(float(lgd_per_loan.median()), 4),
        "lgd_std": round(float(lgd_per_loan.std()), 4),
        "formula": (
            "LGD_i = 1 - clip((total_pymnt_i + recoveries_i - collection_recovery_fee_i) / funded_amnt_i, 0, 1); "
            "LGD = mean(LGD_i) across train-year (2012-2013) charged-off, 36-month, individual loans"
        ),
    }


def load_champion_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    lgb_cfg = frozen["lightgbm_chosen"]

    booster = joblib.load(lgb_cfg["model_path"])
    calibrator = joblib.load(lgb_cfg["calibrator"]["path"])
    feats = lgb_cfg["features"]

    validation = pd.read_parquet(VALIDATION_PARQUET)
    for col in ["home_ownership", "purpose", "verification_status"]:
        validation[col] = validation[col].astype("category")

    raw_pred = booster.predict(validation[feats], num_iteration=booster.best_iteration)
    calibrated_pd = calibrator.predict_proba(logit(raw_pred).reshape(-1, 1))[:, 1]

    # Query by the same population filter used to build validation.parquet
    # (term=36, individual, issue year 2014), then merge on id in pandas -
    # avoids building a 162,570-id SQL IN clause.
    econ = con.execute(
        f"""
        SELECT id, installment, loan_amnt
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) = 2014
        """
    ).fetchdf()
    out = validation[["id", "target"]].merge(econ, on="id", how="left")
    assert out["installment"].isna().sum() == 0, "Some validation ids did not match the cleaned table"
    out["pd"] = calibrated_pd
    return out


def sweep_threshold_table(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    df = df.sort_values("pd").reset_index(drop=True)
    df["interest_income"] = df["installment"] * 36 - df["loan_amnt"]
    df["ev_per_loan"] = (1 - df["pd"]) * df["interest_income"] - df["pd"] * df["loan_amnt"] * lgd

    pd_candidates = np.unique(df["pd"].values)
    # Thin to ~300 candidate cutoffs for a manageable sweep.
    if len(pd_candidates) > 300:
        pd_candidates = pd_candidates[:: len(pd_candidates) // 300]

    rows = []
    for cutoff in pd_candidates:
        approved = df[df["pd"] <= cutoff]
        if len(approved) == 0:
            continue
        rows.append(
            {
                "pd_cutoff": cutoff,
                "approval_rate_pct": round(100 * len(approved) / len(df), 2),
                "n_approved": len(approved),
                "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                "total_expected_net_value": round(float(approved["ev_per_loan"].sum()), 2),
                "mean_expected_net_value_per_loan": round(float(approved["ev_per_loan"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    print("=== LGD estimate (train 2012-2013 charged-off loans only) ===")
    lgd_info = estimate_lgd_from_train(con)
    print(json.dumps(lgd_info, indent=2))
    lgd = lgd_info["lgd_mean"]

    print("\n=== Champion (LightGBM) calibrated PD on validation ===")
    val_df = load_champion_predictions(con)
    print(f"n validation rows: {len(val_df)}")

    print("\n=== Threshold sweep on validation ===")
    threshold_table = sweep_threshold_table(val_df, lgd)
    threshold_table.to_csv(METRICS_DIR / "decision_policy_validation_threshold_sweep.csv", index=False)

    best_row = threshold_table.loc[threshold_table["total_expected_net_value"].idxmax()]
    print(f"Chosen cut-off: PD <= {best_row['pd_cutoff']:.4f}")
    print(best_row.to_string())

    policy = {
        "methodology": (
            "Swept the champion model's (LightGBM) calibrated PD on validation (2014) over its empirical "
            "distribution. For each candidate PD cut-off, approved = PD <= cut-off. Picked the cut-off that "
            "maximizes TOTAL expected net value (sum of (1-PD)*interest_income - PD*loan_amnt*LGD across "
            "approved loans) on validation. This fixed PD value is then applied unchanged to all three models "
            "on test - each model approves test loans with its OWN calibrated PD <= this value."
        ),
        "lgd": lgd_info,
        "lgd_sensitivity_range": LGD_SENSITIVITY_RANGE,
        "interest_income_formula": "interest_income_i = installment_i * 36 - loan_amnt_i",
        "expected_net_value_formula": "EV_i = (1 - PD_i) * interest_income_i - PD_i * loan_amnt_i * LGD",
        "chosen_pd_cutoff": round(float(best_row["pd_cutoff"]), 4),
        "validation_at_chosen_cutoff": {
            "approval_rate_pct": float(best_row["approval_rate_pct"]),
            "n_approved": int(best_row["n_approved"]),
            "actual_default_rate_approved_pct": float(best_row["actual_default_rate_approved_pct"]),
            "total_expected_net_value": float(best_row["total_expected_net_value"]),
            "mean_expected_net_value_per_loan": float(best_row["mean_expected_net_value_per_loan"]),
        },
        "note": (
            "Chosen entirely from validation (2014) and train (2012-2013). Test (2015) has not been read "
            "anywhere in this script. This cutoff, LGD, and interest-income formula are now FIXED and will be "
            "applied identically to test with no further adjustment."
        ),
    }
    with open(METRICS_DIR / "decision_policy.json", "w") as f:
        json.dump(policy, f, indent=2, default=str)
    print(f"\nWrote {METRICS_DIR / 'decision_policy.json'}")


if __name__ == "__main__":
    main()
