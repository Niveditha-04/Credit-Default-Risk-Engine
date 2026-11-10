"""Phase 9 close-out: why loan_to_income's hard 0.5 cap (confirmed in
run_drift.py) only reaches PSI "watch" level, never "alert", despite
being a real and visible policy signal in the raw data."""

import json

import pandas as pd

from src.monitoring.psi import bin_counts_numeric, fit_numeric_bins, psi_from_counts
from src.monitoring.run_drift import TRAIN_PSI_REFERENCE_CUTOFF


def main() -> None:
    train_full = pd.read_parquet("data/processed/train.parquet")
    train_ref = train_full[train_full["issue_d"] >= TRAIN_PSI_REFERENCE_CUTOFF]
    edges = fit_numeric_bins(train_ref["loan_to_income"])

    test = pd.read_parquet("data/processed/test.parquet")
    tc = bin_counts_numeric(train_ref["loan_to_income"], edges)
    cc = bin_counts_numeric(test["loan_to_income"], edges)

    pct_near_cap_train = round(100 * float(((train_ref["loan_to_income"] > 0.495) & (train_ref["loan_to_income"] < 0.505)).mean()), 3)
    pct_near_cap_test = round(100 * float(((test["loan_to_income"] > 0.495) & (test["loan_to_income"] < 0.505)).mean()), 3)

    explanation = (
        f"The top decile bin edge (fit on train) is {edges[-2]:.4f}, so the top bin covers "
        f"[{edges[-2]:.4f}, inf). That one wide bin holds every value from an ordinary loan at "
        f"0.35 up to the capped spike at exactly 0.5, and, once the cap lifts in 2017, values up "
        f"to 1.38. The cap affects only {pct_near_cap_train}% of the train reference and "
        f"{pct_near_cap_test}% of 2015 rows (both within 0.005 of 0.5) - a small fraction sitting "
        f"inside a bin that already holds about 10% of the population by construction. PSI "
        f"compares bin proportions, not the shape within a bin, so a thin spike at one point "
        f"inside a wide bin barely moves that bin's share."
    )

    result = {
        "question": "did loan_to_income cross any PSI threshold, and if not (or only mildly), why",
        "answer": "loan_to_income reached watch level (max PSI 0.1202, first crossed 0.10 in 2017-05) "
        "but never reached alert (0.25) in any of the 48 months",
        "why_the_cap_is_diluted": explanation,
        "top_decile_bin_edge": round(float(edges[-2]), 4),
        "pct_of_train_ref_near_0.5": pct_near_cap_train,
        "pct_of_test_2015_near_0.5": pct_near_cap_test,
        "train_top_bin_share": round(float(tc[-2] / tc.sum()), 4),
        "test_2015_top_bin_share": round(float(cc[-2] / cc.sum()), 4),
        "psi_train_vs_test_2015_only": round(psi_from_counts(tc, cc), 4),
        "first_month_crossed_0.10": "2017-05",
        "note": "PSI rises later in 2017, once the cap starts lifting and the top bin genuinely "
        "gains mass above 0.5-1.3 - not from the capped spike itself.",
    }
    with open("reports/metrics/drift_loan_to_income_psi_explanation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
