"""Evaluation-only economics analyses (no policy or model changes), all
reading from the already-saved reports/metrics/test_scored.parquet
rather than touching raw test rows again.

fixed_approval_rate_and_gap_analysis():
  1. Fixed approval-rate (70/80/90%) comparison on test, for 3 scores
     (LightGBM, scorecard, sub_grade map). XGBoost is not part of this
     table: it has no saved validation-set predictions, and the frozen
     xgboost model artifact is quarantined (see
     data/processed/models/quarantine/README_note.txt), so there is no
     way to re-score validation with it to set a cut-off. Cut-offs for
     the 3 scores shown are set on VALIDATION per score, then applied as
     a fixed value to test - test's own distribution is never used to
     pick a cut-off.
  2. Ex-ante/ex-post gap decomposition for LightGBM: how much of the gap is
     just the model's mean predicted PD (13.74%) being below the actual
     test default rate (14.88%), vs how much is left over. Plus a check of
     the recovery-timing hypothesis (newer loans have had less time to
     recover by the March 2019 data snapshot).
  3. Verification of the scorecard's ex-ante EV being inflated by a less
     dispersed PD distribution rather than a different approval set.

gap_remainder_diagnostic():
  How much of the LightGBM ex-ante/ex-post remainder gap (after removing
  the PD-level-drift portion) is explained by the train-derived
  interest-ratio and LGD assumptions not matching what actually happened
  on test's own Fully Paid / Charged Off loans. Reads one additional
  field (total_rec_int) from the cleaned table for the same already-
  scored test population, logged as an evaluation-only test access.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"

FIXED_APPROVAL_RATES_PCT = [70, 80, 90]


def score_validation_all(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)

    validation = pd.read_parquet(VALIDATION_PARQUET)
    for col in ["home_ownership", "purpose", "verification_status"]:
        validation[col] = validation[col].astype("category")

    from src.evaluation.score_models import (
        score_lightgbm,
        score_scorecard,
        v1_lightgbm_cfg,
    )

    out = validation[["id", "target"]].copy()
    out["pd_lightgbm"] = score_lightgbm(validation, v1_lightgbm_cfg(frozen))
    out["pd_scorecard"] = score_scorecard(validation, frozen["scorecard"])

    train_cleaned = con.execute(
        f"""
        SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
        """
    ).fetchdf()
    train_target = pd.read_parquet(TRAIN_PARQUET)[["id", "target"]]
    pd_map = train_target.merge(train_cleaned, on="id", how="left").groupby("sub_grade")["target"].mean().to_dict()

    val_cleaned = con.execute(
        f"""
        SELECT id, sub_grade, installment, loan_amnt FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual' AND EXTRACT(year FROM issue_d) = 2014
        """
    ).fetchdf()
    out = out.merge(val_cleaned, on="id", how="left")
    out["pd_sub_grade_mapped"] = out["sub_grade"].map(pd_map)
    return out, pd_map


def fixed_approval_rate_and_gap_analysis() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    with open(METRICS_DIR / "decision_policy.json") as f:
        policy = json.load(f)
    lgd = policy["lgd_used_in_ev_model"]
    prepayment_ratio = policy["prepayment_adjusted_interest"]["ratio"]

    test = pd.read_parquet(TEST_SCORED)
    print(f"Loaded {len(test)} rows from {TEST_SCORED} (already scored, no new test read)")

    # ================================================================
    # 1. Fixed approval-rate comparison, cut-offs set on VALIDATION
    # ================================================================
    print("\n=== Scoring validation (all 4) to set fixed-rate cut-offs ===")
    val, train_pd_map = score_validation_all(con)

    test["pd_sub_grade_mapped"] = test["sub_grade"].map(train_pd_map)
    test["scheduled_interest"] = test["installment"] * 36 - test["loan_amnt"]
    test["interest_income"] = test["scheduled_interest"] * prepayment_ratio
    test["realized_net_cash"] = test["total_pymnt"] - test["collection_recovery_fee"] - test["funded_amnt"]

    score_cols = {
        "lightgbm": "pd_lightgbm",
        "scorecard": "pd_scorecard", "sub_grade_map": "pd_sub_grade_mapped",
    }

    rows = []
    cutoffs_used = {}
    for name, col in score_cols.items():
        val_sorted = val.sort_values(col).reset_index(drop=True)
        cutoffs_used[name] = {}
        for rate_pct in FIXED_APPROVAL_RATES_PCT:
            n_approve = int(len(val_sorted) * rate_pct / 100)
            cutoff = float(val_sorted.iloc[:n_approve][col].max())
            cutoffs_used[name][rate_pct] = cutoff

            approved = test[test[col] <= cutoff]
            ev = float(
                ((1 - approved[col]) * approved["interest_income"]
                 - approved[col] * approved["loan_amnt"] * lgd).sum()
            )
            rows.append(
                {
                    "model": name,
                    "target_approval_rate_pct": rate_pct,
                    "pd_cutoff_from_validation": round(cutoff, 4),
                    "actual_approval_rate_on_test_pct": round(100 * len(approved) / len(test), 2),
                    "actual_default_rate_approved_pct": round(100 * approved["target"].mean(), 2),
                    "ex_ante_total_expected_net_value": round(ev, 2),
                    "ex_post_realized_net_cash_total": round(float(approved["realized_net_cash"].sum()), 2),
                }
            )
    fixed_table = pd.DataFrame(rows)
    fixed_table.to_csv(METRICS_DIR / "test_fixed_approval_rate_comparison.csv", index=False)
    print(fixed_table.to_string(index=False))

    n_sub_grade_levels = int(test["sub_grade"].nunique())
    with open(METRICS_DIR / "test_sub_grade_level_count.json", "w") as f:
        json.dump({"n_distinct_sub_grade_levels_test": n_sub_grade_levels}, f, indent=2)
    print(f"\nDistinct sub_grade levels in test: {n_sub_grade_levels}")

    # ================================================================
    # 2. Gap decomposition for LightGBM
    # ================================================================
    print("\n=== Gap decomposition: LightGBM ===")
    champion_full_cutoff = policy["chosen_pd_cutoff"]
    approved = test[test["pd_lightgbm"] <= champion_full_cutoff].copy()

    original_mean_pd = approved["pd_lightgbm"].mean()
    actual_rate = approved["target"].mean()
    scale = actual_rate / original_mean_pd
    approved["pd_scaled"] = (approved["pd_lightgbm"] * scale).clip(0, 1)

    ev_original = float(
        ((1 - approved["pd_lightgbm"]) * approved["interest_income"]
         - approved["pd_lightgbm"] * approved["loan_amnt"] * lgd).sum()
    )
    ev_scaled = float(
        ((1 - approved["pd_scaled"]) * approved["interest_income"]
         - approved["pd_scaled"] * approved["loan_amnt"] * lgd).sum()
    )
    ex_post = float(approved["realized_net_cash"].sum())

    gap_decomposition = {
        "approved_population": "LightGBM approvals at the fixed policy cutoff",
        "n_approved": len(approved),
        "mean_pd_original": round(float(original_mean_pd), 4),
        "mean_pd_scaled": round(float((approved["pd_scaled"]).mean()), 4),
        "actual_default_rate_approved": round(float(actual_rate), 4),
        "scale_factor": round(float(scale), 4),
        "ex_ante_ev_original": round(ev_original, 2),
        "ex_ante_ev_pd_scaled_to_actual": round(ev_scaled, 2),
        "ex_post_realized_cash": round(ex_post, 2),
        "original_gap": round(ev_original - ex_post, 2),
        "gap_explained_by_pd_level_drift": round(ev_original - ev_scaled, 2),
        "gap_explained_by_pd_level_drift_pct_of_original_gap": round(
            100 * (ev_original - ev_scaled) / (ev_original - ex_post), 2
        ),
        "remainder_gap": round(ev_scaled - ex_post, 2),
        "remainder_gap_pct_of_original_gap": round(100 * (ev_scaled - ex_post) / (ev_original - ex_post), 2),
    }
    print(json.dumps(gap_decomposition, indent=2))

    print("\n=== Recovery-timing check: train vs test charged-off loans ===")
    train_co = con.execute(
        f"""
        SELECT id, issue_d, last_pymnt_d, funded_amnt, recoveries,
               date_diff('month', issue_d, last_pymnt_d) AS months_to_last_pymnt
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
          AND loan_status = 'Charged Off'
        """
    ).fetchdf()
    test_co = test[test["target"] == 1].copy()
    test_co["months_to_last_pymnt"] = (
        (pd.to_datetime(test_co["last_pymnt_d"]).dt.year - pd.to_datetime(test_co["issue_d"]).dt.year) * 12
        + (pd.to_datetime(test_co["last_pymnt_d"]).dt.month - pd.to_datetime(test_co["issue_d"]).dt.month)
    )

    recovery_timing = {
        "train_charged_off": {
            "n": len(train_co),
            "median_months_to_last_pymnt": float(train_co["months_to_last_pymnt"].median()),
            "median_recoveries_over_funded_pct": round(
                float((train_co["recoveries"] / train_co["funded_amnt"]).median()) * 100, 2
            ),
            "mean_recoveries_over_funded_pct": round(
                float((train_co["recoveries"] / train_co["funded_amnt"]).mean()) * 100, 2
            ),
        },
        "test_charged_off": {
            "n": len(test_co),
            "median_months_to_last_pymnt": float(test_co["months_to_last_pymnt"].median()),
            "median_recoveries_over_funded_pct": round(
                float((test_co["recoveries"] / test_co["funded_amnt"]).median()) * 100, 2
            ),
            "mean_recoveries_over_funded_pct": round(
                float((test_co["recoveries"] / test_co["funded_amnt"]).mean()) * 100, 2
            ),
        },
        "note": "Data snapshot is March 2019. Test (2015-issued) charged-off loans have had less calendar time to accumulate post-charge-off recoveries than train (2012-2013-issued) charged-off loans by that snapshot date.",
    }
    print(json.dumps(recovery_timing, indent=2))

    with open(METRICS_DIR / "test_gap_decomposition_lightgbm.json", "w") as f:
        json.dump({"gap_decomposition": gap_decomposition, "recovery_timing_check": recovery_timing}, f, indent=2)

    # ================================================================
    # 3. Scorecard gap explanation check
    # ================================================================
    print("\n=== Scorecard vs LightGBM: approval-set overlap and PD dispersion ===")
    lgb_approved_ids = set(test[test["pd_lightgbm"] <= champion_full_cutoff]["id"])
    sc_cutoff = champion_full_cutoff  # same fixed PD cutoff policy applied to all models
    sc_approved_ids = set(test[test["pd_scorecard"] <= sc_cutoff]["id"])

    overlap = lgb_approved_ids & sc_approved_ids
    only_lgb = lgb_approved_ids - sc_approved_ids
    only_sc = sc_approved_ids - lgb_approved_ids

    common = test[test["id"].isin(overlap)]
    scorecard_check = {
        "lightgbm_approved_n": len(lgb_approved_ids),
        "scorecard_approved_n": len(sc_approved_ids),
        "approved_by_both_n": len(overlap),
        "approved_by_both_pct_of_lightgbm_set": round(100 * len(overlap) / len(lgb_approved_ids), 2),
        "approved_by_both_pct_of_scorecard_set": round(100 * len(overlap) / len(sc_approved_ids), 2),
        "only_lightgbm_approved_n": len(only_lgb),
        "only_scorecard_approved_n": len(only_sc),
        "pd_std_full_test": {
            "lightgbm": round(float(test["pd_lightgbm"].std()), 4),
            "scorecard": round(float(test["pd_scorecard"].std()), 4),
        },
        "on_commonly_approved_loans": {
            "mean_pd_lightgbm": round(float(common["pd_lightgbm"].mean()), 4),
            "mean_pd_scorecard": round(float(common["pd_scorecard"].mean()), 4),
            "std_pd_lightgbm": round(float(common["pd_lightgbm"].std()), 4),
            "std_pd_scorecard": round(float(common["pd_scorecard"].std()), 4),
        },
        "interpretation": (
            "Approval sets are nearly identical (see overlap percentages). The ex-ante EV difference comes "
            "from the scorecard assigning systematically lower/less dispersed PDs to the SAME commonly-approved "
            "loans (see mean/std PD on commonly-approved loans), which mechanically inflates its computed EV - "
            "not from the scorecard approving a different, better-risk set of loans."
        ),
    }
    print(json.dumps(scorecard_check, indent=2))
    with open(METRICS_DIR / "test_scorecard_gap_explanation.json", "w") as f:
        json.dump(scorecard_check, f, indent=2)

    print("\nDone.")


def compute_ev(df: pd.DataFrame, interest_ratio: float, lgd: float) -> float:
    scheduled_interest = df["installment"] * 36 - df["loan_amnt"]
    interest_income = scheduled_interest * interest_ratio
    return float(((1 - df["pd_scaled"]) * interest_income - df["pd_scaled"] * df["loan_amnt"] * lgd).sum())


def gap_remainder_diagnostic() -> None:
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
        "economics_gap.py (gap_remainder_diagnostic)", len(test),
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
    fixed_approval_rate_and_gap_analysis()
    print()
    gap_remainder_diagnostic()
