"""Post-hoc analysis: economics at matched funding volume. For each
model, funds exactly its own top N% of 2015 test loans by its own score
(lowest predicted PD first) - an evaluation device for comparing models
at the same volume, not a deployable policy (the deployed policy's
cutoff is fixed on validation; see decision_policy.json and
economics.py). Reads only from the already-saved
reports/metrics/test_scored.parquet, plus a train-year sub_grade default
rate map for the sub_grade model's own score. No model, feature, or
policy change.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"

SELECTION_RATES_PCT = list(range(10, 101, 10))
COST_OF_FUNDS_RATES_PCT = [0, 4, 8]
AVERAGE_BALANCE_FACTOR = 0.5  # approximates the average outstanding balance of an amortizing loan over its life

MODEL_COLS = {
    "lightgbm": "pd_lightgbm",
    "xgboost": "pd_xgboost",
    "scorecard": "pd_scorecard",
    "sub_grade_map": "pd_sub_grade_mapped",
}


def sub_grade_train_pd_map() -> dict:
    train = pd.read_parquet(TRAIN_PARQUET)
    con = duckdb.connect()
    cleaned_sub_grade = con.execute(f"SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')").fetchdf()
    merged = train[["id", "target"]].merge(cleaned_sub_grade, on="id", how="left")
    return merged.groupby("sub_grade")["target"].mean().to_dict()


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    log_access(
        "posthoc_matched_volume_economics.py",
        len(df),
        "Post-hoc: re-read saved test_scored.parquet to fund each model's own top N% of "
        "test loans by its own score, an evaluation device, not a policy. No model, "
        "feature, or policy change.",
    )

    pd_map = sub_grade_train_pd_map()
    df["pd_sub_grade_mapped"] = df["sub_grade"].map(pd_map)

    last_pymnt = pd.to_datetime(df["last_pymnt_d"])
    issue_d = pd.to_datetime(df["issue_d"])
    months = (last_pymnt.dt.year - issue_d.dt.year) * 12 + (last_pymnt.dt.month - issue_d.dt.month)
    n_missing_last_pymnt = int(months.isna().sum())
    df["months_to_last_pymnt"] = months.fillna(0).clip(lower=0)

    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]

    rows = []
    for name, col in MODEL_COLS.items():
        sorted_df = df.sort_values(col).reset_index(drop=True)
        for rate_pct in SELECTION_RATES_PCT:
            n_fund = int(len(sorted_df) * rate_pct / 100)
            funded = sorted_df.iloc[:n_fund]
            funded_amnt_total = float(funded["funded_amnt"].sum())
            realized_cash_total = float(funded["realized_net_cash"].sum())
            row = {
                "model": name,
                "selection_rate_pct": rate_pct,
                "n_funded": n_fund,
                "default_rate_pct": round(100 * float(funded["target"].mean()), 2),
                "realized_net_cash_total": round(realized_cash_total, 2),
                "realized_return_per_dollar_funded_pct": round(100 * realized_cash_total / funded_amnt_total, 4),
            }
            for cof_rate_pct in COST_OF_FUNDS_RATES_PCT:
                rate = cof_rate_pct / 100
                cost = rate * funded["funded_amnt"] * (funded["months_to_last_pymnt"] / 12) * AVERAGE_BALANCE_FACTOR
                adjusted_cash_total = float((funded["realized_net_cash"] - cost).sum())
                row[f"return_per_dollar_after_{cof_rate_pct}pct_cost_of_funds_pct"] = round(
                    100 * adjusted_cash_total / funded_amnt_total, 4
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(METRICS_DIR / "posthoc_matched_volume_economics.csv", index=False)
    print(result.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODEL_COLS:
        sub = result[result["model"] == name]
        ax.plot(sub["selection_rate_pct"], sub["realized_return_per_dollar_funded_pct"], marker="o", label=name)
    ax.set_xlabel("Selection rate (% of test loans funded, by the model's own score)")
    ax.set_ylabel("Realized return per dollar funded (%)")
    ax.set_title("Post-hoc: realized return per dollar vs. selection rate, 2015 test")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "posthoc_return_per_dollar_vs_selection_rate.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved figure to {fig_path}")

    highlight = result[result["selection_rate_pct"].isin([70, 80, 90])]
    summary = {
        "label": "post-hoc",
        "note": (
            "Average-balance approximation: cost of funds = rate x funded_amnt x "
            "(months from issue to last payment / 12) x 0.5, where 0.5 approximates the "
            "average outstanding balance of an amortizing loan over its observed life "
            "(starts near funded_amnt, ends near 0 if fully paid off). "
            f"months_to_last_pymnt is set to 0 for the {n_missing_last_pymnt} test loans with "
            "no recorded payment date."
        ),
        "selection_rates_reported_pct": SELECTION_RATES_PCT,
        "cost_of_funds_rates_pct": COST_OF_FUNDS_RATES_PCT,
        "at_70_80_90_pct_selection": highlight.to_dict("records"),
    }
    with open(METRICS_DIR / "posthoc_matched_volume_economics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
