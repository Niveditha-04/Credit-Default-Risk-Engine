"""Quick diagnostic (evaluation only, no model/policy changes): how much of
the LightGBM ex-ante/ex-post remainder gap ($22.7M, after removing the
PD-level-drift portion) is explained by the train-derived interest-ratio
and LGD assumptions not matching what actually happened on test's own
Fully Paid / Charged Off loans.

Reads test_scored.parquet (already saved) plus one additional field
(total_rec_int) pulled from the cleaned table for the same test
population already scored - logged as another evaluation-only test
access, consistent with every other post-scoring economics script this
phase.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"


def compute_ev(df: pd.DataFrame, interest_ratio: float, lgd: float) -> float:
    scheduled_interest = df["installment"] * 36 - df["loan_amnt"]
    interest_income = scheduled_interest * interest_ratio
    return float(((1 - df["pd_scaled"]) * interest_income - df["pd_scaled"] * df["loan_amnt"] * lgd).sum())


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    test = pd.read_parquet(TEST_SCORED)
    extra = con.execute(
        f"""
        SELECT id, total_rec_int
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual' AND EXTRACT(year FROM issue_d) = 2015
        """
    ).fetchdf()
    test = test.merge(extra, on="id", how="left")
    assert test["total_rec_int"].isna().sum() == 0
    log_access(
        "gap_remainder_diagnostic.py", len(test),
        "Re-read saved test_scored.parquet + joined total_rec_int (evaluation-only field, same test population "
        "already scored) to diagnose the ex-ante/ex-post remainder gap. No model or policy changes.",
    )

    test["scheduled_interest"] = test["installment"] * 36 - test["loan_amnt"]
    test["realized_net_cash"] = test["total_pymnt"] - test["collection_recovery_fee"] - test["funded_amnt"]

    with open(METRICS_DIR / "decision_policy.json") as f:
        policy = json.load(f)
    lgd_train = policy["lgd_used_in_ev_model"]
    ratio_train = policy["prepayment_adjusted_interest"]["ratio"]
    cutoff = policy["chosen_pd_cutoff"]

    # Test's own realized values
    fully_paid = test[test["target"] == 0]
    fp_valid = fully_paid[fully_paid["scheduled_interest"] > 0]
    ratio_test = float(fp_valid["total_rec_int"].sum() / fp_valid["scheduled_interest"].sum())

    charged_off = test[test["target"] == 1]
    cash_loss = 1 - ((charged_off["total_pymnt"] - charged_off["collection_recovery_fee"]) / charged_off["funded_amnt"]).clip(0, 1)
    lgd_test = float(cash_loss.mean())

    print("=== Diagnostic: test realized ratios vs train assumptions ===")
    print(f"Interest ratio (Fully Paid): test={ratio_test:.4f} vs train={ratio_train:.4f}")
    print(f"Cash-basis loss (Charged Off): test={lgd_test:.4f} vs train={lgd_train:.4f}")

    approved = test[test["pd_lightgbm"] <= cutoff].copy()
    original_mean_pd = approved["pd_lightgbm"].mean()
    actual_rate = approved["target"].mean()
    scale = actual_rate / original_mean_pd
    approved["pd_scaled"] = (approved["pd_lightgbm"] * scale).clip(0, 1)

    ev_baseline = compute_ev(approved, ratio_train, lgd_train)
    ev_interest_swapped = compute_ev(approved, ratio_test, lgd_train)
    ev_lgd_swapped = compute_ev(approved, ratio_train, lgd_test)
    ev_both_swapped = compute_ev(approved, ratio_test, lgd_test)
    ex_post = float(approved["realized_net_cash"].sum())
    remainder = ev_baseline - ex_post

    delta_interest = ev_baseline - ev_interest_swapped
    delta_lgd = ev_baseline - ev_lgd_swapped
    delta_both = ev_baseline - ev_both_swapped

    result = {
        "diagnostic": {
            "test_realized_interest_ratio_fully_paid": round(ratio_test, 4),
            "train_realized_interest_ratio_fully_paid": ratio_train,
            "test_cash_basis_loss_charged_off_mean": round(lgd_test, 4),
            "train_cash_basis_loss_charged_off_mean": lgd_train,
        },
        "remainder_decomposition": {
            "ev_scaled_baseline_train_assumptions": round(ev_baseline, 2),
            "ex_post_realized": round(ex_post, 2),
            "remainder_gap": round(remainder, 2),
            "explained_by_interest_ratio_mismatch": round(delta_interest, 2),
            "explained_by_interest_ratio_mismatch_pct": round(100 * delta_interest / remainder, 1),
            "explained_by_lgd_mismatch": round(delta_lgd, 2),
            "explained_by_lgd_mismatch_pct": round(100 * delta_lgd / remainder, 1),
            "explained_by_both_combined": round(delta_both, 2),
            "explained_by_both_combined_pct": round(100 * delta_both / remainder, 1),
            "still_unexplained_after_both": round(remainder - delta_both, 2),
            "still_unexplained_after_both_pct": round(100 * (remainder - delta_both) / remainder, 1),
        },
    }
    with open(METRICS_DIR / "test_gap_remainder_diagnostic.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
