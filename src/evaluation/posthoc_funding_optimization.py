"""Post-hoc: for each model and cost of funds (0/4/8% annual), finds the
funding fraction on 2015 test that maximizes TOTAL realized net cash
after cost of funds (a dollar amount, not a per-dollar rate) - funding
every loan is not necessarily best once financing cost is counted, since
a wider book also finances more of its own losers. Reads only the
already-saved reports/metrics/test_scored.parquet, a second, separate
read from the one-time evaluation scoring pass. No model, feature, or
policy change.

Also reports the annualized return on average capital at each optimum:
net cash after cost, divided by (funded amount x years outstanding x
0.5), where 0.5 approximates the average outstanding balance of an
amortizing loan over its observed life and years outstanding comes from
the realized months to last payment. This is the same cost-of-funds
average-balance approximation already used elsewhere
(posthoc_matched_volume_economics.py), applied in reverse to annualize
the return itself rather than just the cost - it is an approximation,
not a true duration-weighted yield.
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

SELECTION_RATES_PCT = list(range(5, 101, 5))
COST_OF_FUNDS_RATES_PCT = [0, 4, 8]
AVERAGE_BALANCE_FACTOR = 0.5

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
        "posthoc_funding_optimization.py",
        len(df),
        "Post-hoc: re-read saved test_scored.parquet to find, per model and cost of "
        "funds, the funding fraction that maximizes total realized net cash after cost, "
        "and the annualized return on average capital at that optimum. No model, "
        "feature, or policy change.",
    )

    df["pd_sub_grade_mapped"] = df["sub_grade"].map(sub_grade_train_pd_map())
    last_pymnt = pd.to_datetime(df["last_pymnt_d"])
    issue_d = pd.to_datetime(df["issue_d"])
    months = (last_pymnt.dt.year - issue_d.dt.year) * 12 + (last_pymnt.dt.month - issue_d.dt.month)
    df["months_to_last_pymnt"] = months.fillna(0).clip(lower=0)
    df["years_to_last_pymnt"] = df["months_to_last_pymnt"] / 12
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]

    curve_rows = []
    for name, col in MODEL_COLS.items():
        sorted_df = df.sort_values(col).reset_index(drop=True)
        for rate_pct in SELECTION_RATES_PCT:
            n_fund = int(len(sorted_df) * rate_pct / 100)
            funded = sorted_df.iloc[:n_fund]
            funded_amnt_total = float(funded["funded_amnt"].sum())
            avg_capital_total = float((funded["funded_amnt"] * funded["years_to_last_pymnt"] * AVERAGE_BALANCE_FACTOR).sum())
            for cof_rate_pct in COST_OF_FUNDS_RATES_PCT:
                rate = cof_rate_pct / 100
                cost = rate * funded["funded_amnt"] * funded["years_to_last_pymnt"] * AVERAGE_BALANCE_FACTOR
                net_cash_after_cost = float((funded["realized_net_cash"] - cost).sum())
                annualized_return_pct = (
                    round(100 * net_cash_after_cost / avg_capital_total, 4) if avg_capital_total > 0 else None
                )
                curve_rows.append({
                    "model": name,
                    "cost_of_funds_pct": cof_rate_pct,
                    "selection_rate_pct": rate_pct,
                    "n_funded": n_fund,
                    "funded_amnt_total": round(funded_amnt_total, 2),
                    "net_cash_after_cost_total": round(net_cash_after_cost, 2),
                    "annualized_return_on_avg_capital_pct": annualized_return_pct,
                })

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(METRICS_DIR / "posthoc_funding_optimization_curve.csv", index=False)

    optimal_rows = []
    for (name, cof), g in curve.groupby(["model", "cost_of_funds_pct"]):
        best = g.loc[g["net_cash_after_cost_total"].idxmax()]
        optimal_rows.append({
            "model": name,
            "cost_of_funds_pct": cof,
            "optimal_selection_rate_pct": int(best["selection_rate_pct"]),
            "net_cash_after_cost_total": best["net_cash_after_cost_total"],
            "annualized_return_on_avg_capital_pct_at_optimum": best["annualized_return_on_avg_capital_pct"],
        })
    optimal = pd.DataFrame(optimal_rows).sort_values(["cost_of_funds_pct", "model"])
    optimal.to_csv(METRICS_DIR / "posthoc_funding_optimization_optimal.csv", index=False)
    print(optimal.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    at_0pct = curve[curve["cost_of_funds_pct"] == 0]
    line_colors = {}
    for name in MODEL_COLS:
        sub = at_0pct[at_0pct["model"] == name]
        (line,) = ax.plot(sub["selection_rate_pct"], sub["net_cash_after_cost_total"] / 1e6, marker="o", markersize=3, label=name)
        line_colors[name] = line.get_color()
    for _, row in optimal[optimal["cost_of_funds_pct"] == 0].iterrows():
        at_opt = at_0pct[(at_0pct["model"] == row["model"]) & (at_0pct["selection_rate_pct"] == row["optimal_selection_rate_pct"])]
        ax.scatter(
            at_opt["selection_rate_pct"], at_opt["net_cash_after_cost_total"] / 1e6,
            s=80, zorder=5, color=line_colors[row["model"]], edgecolor="black",
        )
    ax.set_xlabel("Selection rate (% of test loans funded, by the model's own score)")
    ax.set_ylabel("Realized net cash after cost of funds ($M)")
    ax.set_title("Post-hoc: net cash vs. funding volume, 0% cost of funds (circled = optimum)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "posthoc_net_cash_vs_funding_volume.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")

    summary = {
        "label": "post-hoc",
        "note": (
            "Optimal selection rate maximizes TOTAL realized net cash after cost of "
            "funds (a dollar amount), not return per dollar funded - funding every "
            "loan is not always best once financing cost on the whole book is counted. "
            "Annualized return on average capital = net cash after cost / (funded "
            "amount x years to last payment x 0.5); this is an approximation (average "
            "balance 0.5, and years from last_pymnt_d rather than a true "
            "duration-weighted yield), stated as such."
        ),
        "optimal": optimal.to_dict("records"),
    }
    with open(METRICS_DIR / "posthoc_funding_optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
