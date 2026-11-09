"""Choose the approval cut-off and every economic assumption on VALIDATION
(2014) and TRAIN (2012-2013) only. This script never reads test (2015).
Run this BEFORE score_test.py, and never change reports/metrics/
decision_policy.json after test has been scored.

v2 (see reports/metrics/decision_policy_changelog.json for what was wrong
with v1 and why):

LGD, two forms, both estimated from TRAIN-year (2012-2013) charged-off
36-month individual loans only:
  (a) cash-basis loss, used for the EV model:
      LGD_i = 1 - clip((total_pymnt_i - collection_recovery_fee_i) / funded_amnt_i, 0, 1)
      total_pymnt already includes recoveries (verified: total_pymnt =
      total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries,
      exact match on every v1 charged-off train row) - do not add
      recoveries again.
  (b) standard LGD on outstanding principal at default, reported for
      reference only, not used in the EV model:
      LGD_i = 1 - clip((recoveries_i - collection_recovery_fee_i) / (funded_amnt_i - total_rec_prncp_i), 0, 1)

Interest income, prepayment-adjusted, estimated from TRAIN Fully Paid
loans only:
    scheduled_interest_i = installment_i * 36 - loan_amnt_i
    prepayment_ratio      = sum(total_rec_int) / sum(scheduled_interest), train Fully Paid loans
    interest_income_i     = scheduled_interest_i * prepayment_ratio
(scheduled_interest assumes the full 36-month term with no early payoff;
the ratio corrects for loans that pay off early and therefore earn less
interest than the full schedule implies.)

Expected net value per loan at a given PD:
    EV_i = (1 - PD_i) * interest_income_i - PD_i * loan_amnt_i * LGD_cash_basis

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

LGD_SENSITIVITY_RANGE = [0.25, 0.30, 0.35, 0.3803, 0.40, 0.45, 0.50, 0.60]
FIXED_APPROVAL_RATES_PCT = [70, 80, 90]


def estimate_lgd_from_train(con: duckdb.DuckDBPyConnection) -> dict:
    df = con.execute(
        f"""
        SELECT funded_amnt, total_pymnt, total_rec_prncp, recoveries, collection_recovery_fee
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
          AND loan_status = 'Charged Off'
        """
    ).fetchdf()

    cash_loss = 1 - ((df["total_pymnt"] - df["collection_recovery_fee"]) / df["funded_amnt"]).clip(0, 1)

    outstanding = df["funded_amnt"] - df["total_rec_prncp"]
    valid = outstanding > 0
    standard_lgd = 1 - (
        (df["recoveries"][valid] - df["collection_recovery_fee"][valid]) / outstanding[valid]
    ).clip(0, 1)

    return {
        "n_charged_off_train_loans": len(df),
        "cash_basis": {
            "formula": "1 - clip((total_pymnt - collection_recovery_fee) / funded_amnt, 0, 1)",
            "used_for": "EV model",
            "mean": round(float(cash_loss.mean()), 4),
            "median": round(float(cash_loss.median()), 4),
            "std": round(float(cash_loss.std()), 4),
        },
        "standard_outstanding_principal_basis": {
            "formula": "1 - clip((recoveries - collection_recovery_fee) / (funded_amnt - total_rec_prncp), 0, 1)",
            "used_for": "reference only, not used in EV model",
            "n_valid": int(valid.sum()),
            "mean": round(float(standard_lgd.mean()), 4),
            "median": round(float(standard_lgd.median()), 4),
            "std": round(float(standard_lgd.std()), 4),
        },
    }


def estimate_prepayment_ratio_from_train(con: duckdb.DuckDBPyConnection) -> dict:
    df = con.execute(
        f"""
        SELECT installment, loan_amnt, total_rec_int
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
          AND loan_status = 'Fully Paid'
        """
    ).fetchdf()
    df["scheduled_interest"] = df["installment"] * 36 - df["loan_amnt"]
    df = df[df["scheduled_interest"] > 0]
    ratio = df["total_rec_int"].sum() / df["scheduled_interest"].sum()
    return {
        "n_fully_paid_train_loans": len(df),
        "formula": "sum(total_rec_int) / sum(installment*36 - loan_amnt), train Fully Paid loans only",
        "ratio": round(float(ratio), 4),
        "interpretation": (
            f"Fully paid train loans realized {ratio*100:.1f}% of the interest a full 36-month "
            "schedule would imply, due to early payoff."
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


def add_ev_columns(df: pd.DataFrame, lgd: float, prepayment_ratio: float) -> pd.DataFrame:
    df = df.copy()
    df["scheduled_interest"] = df["installment"] * 36 - df["loan_amnt"]
    df["interest_income"] = df["scheduled_interest"] * prepayment_ratio
    df["ev_per_loan"] = (1 - df["pd"]) * df["interest_income"] - df["pd"] * df["loan_amnt"] * lgd
    return df


def sweep_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("pd").reset_index(drop=True)
    pd_candidates = np.unique(df["pd"].values)
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

    print("=== v2 LGD estimate (train 2012-2013 charged-off loans only) ===")
    lgd_info = estimate_lgd_from_train(con)
    print(json.dumps(lgd_info, indent=2))
    lgd = lgd_info["cash_basis"]["mean"]

    print("\n=== v2 prepayment-adjusted interest ratio (train Fully Paid loans only) ===")
    prepay_info = estimate_prepayment_ratio_from_train(con)
    print(json.dumps(prepay_info, indent=2))
    prepayment_ratio = prepay_info["ratio"]

    print("\n=== Champion (LightGBM) calibrated PD on validation ===")
    val_df = load_champion_predictions(con)
    val_df = add_ev_columns(val_df, lgd, prepayment_ratio)
    print(f"n validation rows: {len(val_df)}")

    print("\n=== Threshold sweep on validation (v2) ===")
    threshold_table = sweep_threshold_table(val_df)
    threshold_table.to_csv(METRICS_DIR / "decision_policy_validation_threshold_sweep.csv", index=False)

    best_row = threshold_table.loc[threshold_table["total_expected_net_value"].idxmax()]
    print(f"Chosen cut-off: PD <= {best_row['pd_cutoff']:.4f}")
    print(best_row.to_string())

    print("\n=== Fixed approval-rate views on validation (champion only, for reference) ===")
    fixed_rate_rows = []
    val_sorted = val_df.sort_values("pd").reset_index(drop=True)
    for rate_pct in FIXED_APPROVAL_RATES_PCT:
        n_approve = int(len(val_sorted) * rate_pct / 100)
        approved = val_sorted.iloc[:n_approve]
        fixed_rate_rows.append(
            {
                "approval_rate_pct": rate_pct,
                "pd_cutoff": round(float(approved["pd"].max()), 4),
                "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                "total_expected_net_value": round(float(approved["ev_per_loan"].sum()), 2),
            }
        )
    fixed_rate_df = pd.DataFrame(fixed_rate_rows)
    print(fixed_rate_df.to_string(index=False))

    policy = {
        "version": "v2",
        "superseded_version": "v1 (reports/metrics/decision_policy_v1_superseded.json) - see decision_policy_changelog.json",
        "methodology": (
            "Swept the champion model's (LightGBM) calibrated PD on validation (2014) over its empirical "
            "distribution. For each candidate PD cut-off, approved = PD <= cut-off. Picked the cut-off that "
            "maximizes TOTAL expected net value (sum of (1-PD)*interest_income - PD*loan_amnt*LGD_cash_basis "
            "across approved loans) on validation. This fixed PD value is then applied unchanged to all three "
            "models on test - each model approves test loans with its OWN calibrated PD <= this value."
        ),
        "lgd": lgd_info,
        "lgd_used_in_ev_model": lgd,
        "lgd_sensitivity_range": LGD_SENSITIVITY_RANGE,
        "prepayment_adjusted_interest": prepay_info,
        "interest_income_formula": "interest_income_i = (installment_i * 36 - loan_amnt_i) * prepayment_ratio",
        "expected_net_value_formula": "EV_i = (1 - PD_i) * interest_income_i - PD_i * loan_amnt_i * LGD_cash_basis",
        "chosen_pd_cutoff": round(float(best_row["pd_cutoff"]), 4),
        "validation_at_chosen_cutoff": {
            "approval_rate_pct": float(best_row["approval_rate_pct"]),
            "n_approved": int(best_row["n_approved"]),
            "actual_default_rate_approved_pct": float(best_row["actual_default_rate_approved_pct"]),
            "total_expected_net_value": float(best_row["total_expected_net_value"]),
            "mean_expected_net_value_per_loan": float(best_row["mean_expected_net_value_per_loan"]),
        },
        "fixed_approval_rate_views_validation": fixed_rate_rows,
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
