"""Apply the corrected (v2) economics to test, once. Reads ONLY from the
already-saved reports/metrics/test_scored.parquet (no new read of raw
test data - model scores are unchanged from the original Test evaluation
commit) and reports/metrics/decision_policy.json (the v2 cutoff/LGD/
interest formula, fixed on validation+train before this script runs).

See reports/metrics/decision_policy_changelog.json for what was wrong
with v1 and why.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
MODEL_COLS = {"lightgbm": "pd_lightgbm", "xgboost": "pd_xgboost", "scorecard": "pd_scorecard"}
FIXED_APPROVAL_RATES_PCT = [70, 80, 90]


def add_v2_ev_columns(df: pd.DataFrame, lgd: float, prepayment_ratio: float) -> pd.DataFrame:
    df = df.copy()
    df["scheduled_interest"] = df["installment"] * 36 - df["loan_amnt"]
    df["interest_income"] = df["scheduled_interest"] * prepayment_ratio
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    return df


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    print(f"Loaded {len(df)} rows from {TEST_SCORED} (already scored and saved, no new test read)")

    with open(METRICS_DIR / "decision_policy.json") as f:
        policy = json.load(f)
    lgd = policy["lgd_used_in_ev_model"]
    prepayment_ratio = policy["prepayment_adjusted_interest"]["ratio"]
    cutoff = policy["chosen_pd_cutoff"]

    df = add_v2_ev_columns(df, lgd, prepayment_ratio)

    print(f"\n=== v2 economics at fixed cutoff (PD <= {cutoff}) ===")
    econ_rows = []
    for name, col in MODEL_COLS.items():
        ev_per_loan = (1 - df[col]) * df["interest_income"] - df[col] * df["loan_amnt"] * lgd
        approved = df[df[col] <= cutoff]
        approved_ev = ev_per_loan[df[col] <= cutoff]
        ex_ante = float(approved_ev.sum())
        ex_post = float(approved["realized_net_cash"].sum())
        econ_rows.append(
            {
                "model": name,
                "pd_cutoff": cutoff,
                "approval_rate_pct": round(100 * len(approved) / len(df), 2),
                "n_approved": len(approved),
                "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                "ex_ante_expected_loss_total": round(
                    float((approved[col] * approved["loan_amnt"] * lgd).sum()), 2
                ),
                "ex_ante_total_expected_net_value": round(ex_ante, 2),
                "ex_post_realized_net_cash_total": round(ex_post, 2),
                "ex_post_realized_net_cash_per_loan": round(ex_post / len(approved), 2),
                "ex_ante_minus_ex_post_gap": round(ex_ante - ex_post, 2),
                "ex_ante_minus_ex_post_gap_pct_of_ex_ante": round(100 * (ex_ante - ex_post) / ex_ante, 2),
            }
        )
    econ_summary = pd.DataFrame(econ_rows)
    econ_summary.to_csv(METRICS_DIR / "test_economics_at_fixed_cutoff.csv", index=False)
    print(econ_summary.to_string(index=False))

    print("\n=== v2 LGD sensitivity (champion model, fixed PD cutoff) ===")
    lgd_sens_rows = []
    champion_approved = df[df["pd_lightgbm"] <= cutoff]
    for lgd_val in policy["lgd_sensitivity_range"]:
        expected_loss = float((champion_approved["pd_lightgbm"] * champion_approved["loan_amnt"] * lgd_val).sum())
        ev = float(
            ((1 - champion_approved["pd_lightgbm"]) * champion_approved["interest_income"]
             - champion_approved["pd_lightgbm"] * champion_approved["loan_amnt"] * lgd_val).sum()
        )
        lgd_sens_rows.append({"lgd": lgd_val, "ex_ante_expected_loss_total": round(expected_loss, 2),
                               "ex_ante_total_expected_net_value": round(ev, 2)})
    lgd_sens_df = pd.DataFrame(lgd_sens_rows)
    lgd_sens_df.to_csv(METRICS_DIR / "test_lgd_sensitivity.csv", index=False)
    print(lgd_sens_df.to_string(index=False))

    print("\n=== v2 threshold table (champion model, test) ===")
    threshold_rows = []
    pd_candidates = np.percentile(df["pd_lightgbm"], np.arange(10, 101, 10))
    for cut in pd_candidates:
        approved = df[df["pd_lightgbm"] <= cut]
        ev = float(((1 - approved["pd_lightgbm"]) * approved["interest_income"]
                     - approved["pd_lightgbm"] * approved["loan_amnt"] * lgd).sum())
        threshold_rows.append(
            {
                "pd_cutoff": round(float(cut), 4),
                "approval_rate_pct": round(100 * len(approved) / len(df), 2),
                "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                "ex_ante_total_expected_net_value": round(ev, 2),
                "ex_post_realized_net_cash_total": round(float(approved["realized_net_cash"].sum()), 2),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(METRICS_DIR / "test_threshold_table.csv", index=False)
    print(threshold_df.to_string(index=False))

    print("\n=== Fixed approval-rate views (70/80/90%), all 3 models, test (v2) ===")
    fixed_rows = []
    for name, col in MODEL_COLS.items():
        sorted_df = df.sort_values(col).reset_index(drop=True)
        for rate_pct in FIXED_APPROVAL_RATES_PCT:
            n_approve = int(len(sorted_df) * rate_pct / 100)
            approved = sorted_df.iloc[:n_approve]
            ev = float(((1 - approved[col]) * approved["interest_income"] - approved[col] * approved["loan_amnt"] * lgd).sum())
            ex_post = float(approved["realized_net_cash"].sum())
            fixed_rows.append(
                {
                    "model": name,
                    "approval_rate_pct": rate_pct,
                    "pd_cutoff": round(float(approved[col].max()), 4),
                    "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                    "ex_ante_total_expected_net_value": round(ev, 2),
                    "ex_post_realized_net_cash_total": round(ex_post, 2),
                }
            )
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(METRICS_DIR / "test_fixed_approval_rate_views.csv", index=False)
    print(fixed_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
