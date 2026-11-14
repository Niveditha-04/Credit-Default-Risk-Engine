"""Post-hoc selection economics on test (2015) and validation (2014),
grouped by topic in one module instead of three separate scripts. Every
check here is a second, separate read of already-saved predictions
after the one-time evaluation that chose and froze the champion; none
of them changes a model, feature, or policy decision. Each section
logs its own access through src.evaluation.test_access_log (the one
shared loader for every check that re-reads
reports/metrics/test_scored.parquet).

Sections, run in order by main():
  1. Matched-volume economics: realized return per dollar funded, at
     each model's own top-N% selection, for LightGBM, XGBoost, the
     scorecard, and sub_grade's train-year PD map, with a paired
     bootstrap CI on LightGBM minus scorecard/sub_grade_map.
  2. Funding optimization: the funding fraction, per model and per cost
     of funds, that maximizes TOTAL realized net cash after cost.
  3. sub_grade + LightGBM at matched volume: sub_grade alone vs.
     sub_grade with LightGBM's champion features added (ablation c),
     on validation (2014) and test (2015).
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.test_access_log import load_test_scored, log_access
from src.models.gbm import feature_list

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

COST_OF_FUNDS_RATES_PCT = [0, 4, 8]
AVERAGE_BALANCE_FACTOR = 0.5  # approximates the average outstanding balance of an amortizing loan over its life

MODEL_COLS = {
    "lightgbm": "pd_lightgbm",
    "xgboost": "pd_xgboost",
    "scorecard": "pd_scorecard",
    "sub_grade_map": "pd_sub_grade_mapped",
}

N_BOOTSTRAP = 1000
SEED = 42


def sub_grade_train_pd_map() -> dict:
    """Shared across sections 1 and 2: the train-year sub_grade -> PD
    map read fresh from disk (train.parquet plus the cleaned table's
    sub_grade column)."""
    train = pd.read_parquet(TRAIN_PARQUET)
    con = duckdb.connect()
    cleaned_sub_grade = con.execute(f"SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')").fetchdf()
    merged = train[["id", "target"]].merge(cleaned_sub_grade, on="id", how="left")
    return merged.groupby("sub_grade")["target"].mean().to_dict()


# ---------------------------------------------------------------------
# 1. Matched-volume economics (was posthoc_matched_volume_economics.py)
# ---------------------------------------------------------------------

MATCHED_VOLUME_SELECTION_RATES_PCT = list(range(10, 101, 10))
MATCHED_VOLUME_BOOTSTRAP_RATES_PCT = [70, 80, 90]
MATCHED_VOLUME_BOOTSTRAP_MODELS = ["lightgbm", "scorecard", "sub_grade_map"]


def _return_per_dollar(score: np.ndarray, funded_amnt: np.ndarray, cash_minus_cost: np.ndarray, rate_pct: int) -> float:
    n_fund = int(len(score) * rate_pct / 100)
    idx = np.argsort(score)[:n_fund]
    funded_total = funded_amnt[idx].sum()
    if funded_total == 0:
        return float("nan")
    return 100 * float(cash_minus_cost[idx].sum() / funded_total)


def paired_bootstrap_matched_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstrap (1,000 resamples, seeded) on the difference in
    realized return per dollar funded between LightGBM and each of
    scorecard/sub_grade_map, at each of the 3 reported selection rates
    and 3 cost-of-funds rates. Paired: the same resampled rows underlie
    both models' return-per-dollar in a given resample, so the CI
    reflects the two models' realized outcomes on the SAME resampled
    loans, not two independently resampled populations."""
    rng = np.random.RandomState(SEED)
    n_rows = len(df)
    funded_amnt = df["funded_amnt"].values
    realized_net_cash = df["realized_net_cash"].values
    months_to_last_pymnt = df["months_to_last_pymnt"].values
    scores = {m: df[MODEL_COLS[m]].values for m in MATCHED_VOLUME_BOOTSTRAP_MODELS}

    # diffs[(rate_pct, cof_rate_pct, "scorecard")] -> list of per-resample diffs, etc.
    diffs: dict = {
        (rate_pct, cof_rate_pct, other): []
        for rate_pct in MATCHED_VOLUME_BOOTSTRAP_RATES_PCT
        for cof_rate_pct in COST_OF_FUNDS_RATES_PCT
        for other in ("scorecard", "sub_grade_map")
    }

    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n_rows, n_rows)
        funded_r = funded_amnt[idx]
        cash_r = realized_net_cash[idx]
        months_r = months_to_last_pymnt[idx]
        score_r = {m: scores[m][idx] for m in MATCHED_VOLUME_BOOTSTRAP_MODELS}
        for cof_rate_pct in COST_OF_FUNDS_RATES_PCT:
            cost_r = (cof_rate_pct / 100) * funded_r * (months_r / 12) * AVERAGE_BALANCE_FACTOR
            cash_minus_cost_r = cash_r - cost_r
            per_model = {
                m: {
                    rate_pct: _return_per_dollar(score_r[m], funded_r, cash_minus_cost_r, rate_pct)
                    for rate_pct in MATCHED_VOLUME_BOOTSTRAP_RATES_PCT
                }
                for m in MATCHED_VOLUME_BOOTSTRAP_MODELS
            }
            for rate_pct in MATCHED_VOLUME_BOOTSTRAP_RATES_PCT:
                for other in ("scorecard", "sub_grade_map"):
                    diffs[(rate_pct, cof_rate_pct, other)].append(
                        per_model["lightgbm"][rate_pct] - per_model[other][rate_pct]
                    )

    rows = []
    for rate_pct in MATCHED_VOLUME_BOOTSTRAP_RATES_PCT:
        for cof_rate_pct in COST_OF_FUNDS_RATES_PCT:
            for other in ("scorecard", "sub_grade_map"):
                d = np.array(diffs[(rate_pct, cof_rate_pct, other)])
                rows.append({
                    "selection_rate_pct": rate_pct,
                    "cost_of_funds_pct": cof_rate_pct,
                    "comparison": f"lightgbm_minus_{other}",
                    "mean_diff_pct": round(float(d.mean()), 4),
                    "ci_low": round(float(np.percentile(d, 2.5)), 4),
                    "ci_high": round(float(np.percentile(d, 97.5)), 4),
                    "significant": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                })
    return pd.DataFrame(rows)


def run_matched_volume_economics() -> None:
    df = load_test_scored(
        "posthoc_matched_volume_economics.py",
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
        for rate_pct in MATCHED_VOLUME_SELECTION_RATES_PCT:
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

    print("\n=== Paired bootstrap CI on LightGBM minus scorecard/sub_grade_map, 70/80/90%, 0/4/8% cost of funds ===")
    bootstrap_df = paired_bootstrap_matched_volume(df)
    bootstrap_df.to_csv(METRICS_DIR / "posthoc_matched_volume_bootstrap.csv", index=False)
    print(bootstrap_df.to_string(index=False))

    # Comparison table: return per dollar over the loan's life (not annualized) for
    # LightGBM, scorecard, sub_grade_map, their differences, and each difference's CI.
    comparison_rows = []
    pivot = result.pivot(index="selection_rate_pct", columns="model")
    for rate_pct in MATCHED_VOLUME_BOOTSTRAP_RATES_PCT:
        for cof_rate_pct in COST_OF_FUNDS_RATES_PCT:
            col = (
                "realized_return_per_dollar_funded_pct"
                if cof_rate_pct == 0
                else f"return_per_dollar_after_{cof_rate_pct}pct_cost_of_funds_pct"
            )
            row = {"selection_rate_pct": rate_pct, "cost_of_funds_pct": cof_rate_pct}
            for m in MATCHED_VOLUME_BOOTSTRAP_MODELS:
                row[f"{m}_return_per_dollar_pct"] = pivot.loc[rate_pct, (col, m)]
            for other in ("scorecard", "sub_grade_map"):
                b = bootstrap_df[
                    (bootstrap_df["selection_rate_pct"] == rate_pct)
                    & (bootstrap_df["cost_of_funds_pct"] == cof_rate_pct)
                    & (bootstrap_df["comparison"] == f"lightgbm_minus_{other}")
                ].iloc[0]
                row[f"lightgbm_minus_{other}_pct"] = round(
                    row["lightgbm_return_per_dollar_pct"] - row[f"{other}_return_per_dollar_pct"], 4
                )
                row[f"lightgbm_minus_{other}_ci_low"] = b["ci_low"]
                row[f"lightgbm_minus_{other}_ci_high"] = b["ci_high"]
                row[f"lightgbm_minus_{other}_significant"] = bool(b["significant"])
            comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(METRICS_DIR / "posthoc_matched_volume_comparison_table.csv", index=False)
    print("\n=== Comparison table (return per dollar over the loan's life, not annualized) ===")
    print(comparison_df.to_string(index=False))

    highlight = result[result["selection_rate_pct"].isin([70, 80, 90])]
    summary = {
        "label": "post-hoc",
        "note": (
            "Return per dollar funded is measured over the loan's observed life "
            "(issue to last payment on record), not annualized - a 36-month loan's "
            "realized return per dollar is not comparable to an annual rate without "
            "dividing by its holding period. Average-balance approximation: cost of "
            "funds = rate x funded_amnt x (months from issue to last payment / 12) x "
            "0.5, where 0.5 approximates the average outstanding balance of an "
            "amortizing loan over its observed life (starts near funded_amnt, ends "
            f"near 0 if fully paid off). months_to_last_pymnt is set to 0 for the "
            f"{n_missing_last_pymnt} test loans with no recorded payment date."
        ),
        "selection_rates_reported_pct": MATCHED_VOLUME_SELECTION_RATES_PCT,
        "cost_of_funds_rates_pct": COST_OF_FUNDS_RATES_PCT,
        "at_70_80_90_pct_selection": highlight.to_dict("records"),
        "bootstrap_comparison_table": "reports/metrics/posthoc_matched_volume_comparison_table.csv",
    }
    with open(METRICS_DIR / "posthoc_matched_volume_economics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 2. Funding optimization (was posthoc_funding_optimization.py)
# ---------------------------------------------------------------------

FUNDING_OPT_SELECTION_RATES_PCT = list(range(5, 101, 5))


def run_funding_optimization() -> None:
    """For each model and cost of funds (0/4/8% annual), finds the
    funding fraction on 2015 test that maximizes TOTAL realized net cash
    after cost of funds (a dollar amount, not a per-dollar rate) -
    funding every loan is not necessarily best once financing cost is
    counted, since a wider book also finances more of its own losers.

    Also reports the annualized return on average capital at each
    optimum: net cash after cost, divided by (funded amount x years
    outstanding x 0.5), where 0.5 approximates the average outstanding
    balance of an amortizing loan over its observed life and years
    outstanding comes from the realized months to last payment. This is
    the same cost-of-funds average-balance approximation already used
    in section 1, applied in reverse to annualize the return itself
    rather than just the cost - it is an approximation, not a true
    duration-weighted yield."""
    df = load_test_scored(
        "posthoc_funding_optimization.py",
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
        for rate_pct in FUNDING_OPT_SELECTION_RATES_PCT:
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


# ---------------------------------------------------------------------
# 3. sub_grade + LightGBM at matched volume (was
#    posthoc_subgrade_lightgbm_matched_volume.py)
# ---------------------------------------------------------------------

SUBGRADE_MATCHED_SELECTION_RATES_PCT = [70, 80, 90]
SUBGRADE_MATCHED_MAX_BOOST_ROUNDS = 3000
SUBGRADE_MATCHED_EARLY_STOPPING_ROUNDS = 50

FROZEN_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "seed": SEED,
    "verbosity": -1,
    "num_leaves": 15,
    "learning_rate": 0.08,
    "min_child_samples": 1500,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}


def join_sub_grade(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    sub_grade = con.execute(f"SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')").fetchdf()
    return df.merge(sub_grade, on="id", how="left")


def join_financials(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    financials = con.execute(
        f"SELECT id, funded_amnt, total_pymnt, collection_recovery_fee FROM read_parquet('{CLEANED_PARQUET}')"
    ).fetchdf()
    return df.merge(financials, on="id", how="left")


def sub_grade_train_pd_map_from_df(train_sg: pd.DataFrame) -> dict:
    """Section 3's own variant: the train-year PD map computed from an
    already-loaded, already sub_grade-joined train DataFrame, rather
    than reading train.parquet fresh from disk (sub_grade_train_pd_map,
    shared by sections 1 and 2, does the fresh-read version)."""
    return train_sg.groupby("sub_grade")["target"].mean().to_dict()


def matched_volume_table(df: pd.DataFrame, model_cols: dict, label: str) -> pd.DataFrame:
    df = df.copy()
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    rows = []
    for name, col in model_cols.items():
        sorted_df = df.sort_values(col).reset_index(drop=True)
        for rate_pct in SUBGRADE_MATCHED_SELECTION_RATES_PCT:
            n_fund = int(len(sorted_df) * rate_pct / 100)
            funded = sorted_df.iloc[:n_fund]
            funded_amnt_total = float(funded["funded_amnt"].sum())
            cash_total = float(funded["realized_net_cash"].sum())
            rows.append({
                "evaluation_year": label,
                "model": name,
                "selection_rate_pct": rate_pct,
                "n_funded": n_fund,
                "default_rate_pct": round(100 * float(funded["target"].mean()), 2),
                "realized_return_per_dollar_funded_pct": round(100 * cash_total / funded_amnt_total, 4),
            })
    return pd.DataFrame(rows)


def run_subgrade_lightgbm_matched_volume() -> None:
    """sub_grade alone vs. sub_grade + LightGBM (LightGBM with the
    champion's 69 features plus sub_grade as a feature - the same model
    as ablation c in src.models.posthoc_v2, refit here to get per-loan
    predictions) at matched funding volume, on validation (2014) and
    test (2015, post-hoc).

    Why not a clean, fully-resolved holdout year instead of reusing
    validation: 2016 loans are not fully resolved by this data's March
    2019 snapshot (see src/data/target.py,
    src.evaluation.posthoc_validation's replay-performance section), so
    realized_net_cash (built from total_pymnt, which keeps growing until
    a loan resolves) would understate cash for loans still being paid
    down - there is no year available in this data that is both unused
    by model selection and fully resolved. Validation (2014) is reused
    with that caveat stated plainly: it was the early-stopping target
    for this model's own tree count, so its economics here are not
    out-of-sample in the way test (2015) is."""
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    champion_feats = feature_list(list(train.columns))
    feats_plus_sub_grade = champion_feats + ["sub_grade"]

    train_sg = join_sub_grade(train.copy())
    validation_sg = join_sub_grade(validation.copy())
    test_sg = join_sub_grade(test.copy())

    cat_cols = CATEGORICAL_COLUMNS + ["sub_grade"]
    for col in cat_cols:
        train_sg[col] = train_sg[col].astype("category")
        validation_sg[col] = validation_sg[col].astype("category").cat.set_categories(train_sg[col].cat.categories)
        test_sg[col] = test_sg[col].astype("category").cat.set_categories(train_sg[col].cat.categories)

    log_access(
        "posthoc_subgrade_lightgbm_matched_volume.py",
        len(test_sg),
        "Post-hoc: refits LightGBM with champion features + sub_grade (ablation c), "
        "scores it on validation (2014) and test (2015, post-hoc), and compares matched "
        "-volume economics against sub_grade alone. No model, feature, or policy change.",
    )

    # Refit ablation c (same methodology as src.models.posthoc_v2's
    # ablations section: train 2012-2013, early-stopped against validation 2014).
    lgb_train = lgb.Dataset(
        train_sg[feats_plus_sub_grade], label=train_sg["target"].values, categorical_feature=cat_cols,
        free_raw_data=False, params={"feature_pre_filter": False},
    )
    lgb_val = lgb.Dataset(
        validation_sg[feats_plus_sub_grade], label=validation_sg["target"].values, reference=lgb_train, free_raw_data=False
    )
    booster = lgb.train(
        FROZEN_LGB_PARAMS, lgb_train, num_boost_round=SUBGRADE_MATCHED_MAX_BOOST_ROUNDS, valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(SUBGRADE_MATCHED_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    validation_sg["pd_sub_grade_plus_lightgbm"] = booster.predict(
        validation_sg[feats_plus_sub_grade], num_iteration=booster.best_iteration
    )
    test_sg["pd_sub_grade_plus_lightgbm"] = booster.predict(
        test_sg[feats_plus_sub_grade], num_iteration=booster.best_iteration
    )

    pd_map = sub_grade_train_pd_map_from_df(train_sg)
    validation_sg["pd_sub_grade_mapped"] = validation_sg["sub_grade"].map(pd_map)
    test_sg["pd_sub_grade_mapped"] = test_sg["sub_grade"].map(pd_map)

    validation_sg = join_financials(validation_sg)
    test_financials = pd.read_parquet(TEST_SCORED)[["id", "funded_amnt", "total_pymnt", "collection_recovery_fee"]]
    test_sg = test_sg.merge(test_financials, on="id", how="left")

    model_cols = {"sub_grade_map": "pd_sub_grade_mapped", "sub_grade_plus_lightgbm": "pd_sub_grade_plus_lightgbm"}
    validation_table = matched_volume_table(validation_sg, model_cols, "2014 (validation, in-sample for this model's tree count)")
    test_table = matched_volume_table(test_sg, model_cols, "2015 (test, post-hoc)")
    result = pd.concat([validation_table, test_table], ignore_index=True)
    result.to_csv(METRICS_DIR / "posthoc_subgrade_lightgbm_matched_volume.csv", index=False)
    print(result.to_string(index=False))

    summary = {
        "label": "post-hoc",
        "note": (
            "2016 is not used as a clean holdout year because 2016 loans are not fully "
            "resolved by this data's March 2019 snapshot - realized_net_cash (built from "
            "total_pymnt) would understate cash for loans still being paid down. There is "
            "no year in this data that is both unused by this model's own tree-count "
            "selection and fully resolved, so validation (2014) is reused with that "
            "caveat stated: it was this model's early-stopping target, so its economics "
            "here are in-sample for tuning, unlike test (2015)."
        ),
        "validation_auc_ablation_c_reference": 0.6891,
        "test_auc_ablation_c_reference": 0.6967,
        "by_year_and_rate": result.to_dict("records"),
    }
    with open(METRICS_DIR / "posthoc_subgrade_lightgbm_matched_volume_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    print("=== 1. Matched-volume economics ===")
    run_matched_volume_economics()
    print("\n=== 2. Funding optimization ===")
    run_funding_optimization()
    print("\n=== 3. sub_grade + LightGBM at matched volume ===")
    run_subgrade_lightgbm_matched_volume()


if __name__ == "__main__":
    main()
