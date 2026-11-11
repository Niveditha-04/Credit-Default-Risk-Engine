"""Full test (2015) evaluation. Reads ONLY from
reports/metrics/test_scored.parquet (the one-time scoring pass saved by
score_test.py) plus train.parquet (for the sub_grade-to-PD map). This
script does not open test.parquet again. The approval economics
(expected loss, expected net value, threshold table) are a separate
concern, computed once by src.evaluation.economics from the same
decision_policy.json and test_scored.parquet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
SEED = 42
N_BOOTSTRAP = 1000

MODEL_COLS = {"lightgbm": "pd_lightgbm", "xgboost": "pd_xgboost", "scorecard": "pd_scorecard"}

CENSUS_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast",
    "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South",
    "VA": "South", "DC": "South", "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West",
    "UT": "West", "WY": "West", "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def brier_skill_score(y_true, y_pred) -> float:
    brier_model = brier_score_loss(y_true, y_pred)
    brier_climatology = brier_score_loss(y_true, np.full_like(y_pred, y_true.mean()))
    return 1 - brier_model / brier_climatology


def core_metrics(y_true, y_pred) -> dict:
    auc = roc_auc_score(y_true, y_pred)
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y_true, y_pred), 4),
        "brier": round(brier_score_loss(y_true, y_pred), 5),
        "brier_skill_score": round(brier_skill_score(y_true, y_pred), 4),
        "mean_predicted_pd_pct": round(float(y_pred.mean()) * 100, 2),
        "actual_default_rate_pct": round(float(y_true.mean()) * 100, 2),
    }


def calibration_table(y_true, y_pred, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["decile"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    table = df.groupby("decile").agg(
        n=("actual", "size"), mean_predicted_pd=("pred", "mean"), actual_default_rate=("actual", "mean")
    )
    table["mean_predicted_pd_pct"] = (table["mean_predicted_pd"] * 100).round(2)
    table["actual_default_rate_pct"] = (table["actual_default_rate"] * 100).round(2)
    return table.drop(columns=["mean_predicted_pd", "actual_default_rate"])


def sub_grade_ordinal(sub_grade: pd.Series) -> pd.Series:
    letters = {l: i for i, l in enumerate("ABCDEFG")}
    return sub_grade.apply(lambda s: letters[s[0]] * 5 + int(s[1]))


def sub_grade_train_pd_map(train: pd.DataFrame, cleaned_sub_grade: pd.Series) -> dict:
    df = pd.DataFrame({"sub_grade": cleaned_sub_grade, "target": train["target"].values})
    return df.groupby("sub_grade")["target"].mean().to_dict()


def paired_bootstrap(y_true, score_a, score_b, metric_fn, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_rows = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, n_rows, n_rows)
        y_s, a_s, b_s = y_true[idx], score_a[idx], score_b[idx]
        if y_s.sum() == 0 or y_s.sum() == n_rows:
            continue
        diffs.append(metric_fn(y_s, a_s) - metric_fn(y_s, b_s))
    diffs = np.array(diffs)
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_high": round(float(np.percentile(diffs, 97.5)), 4),
        "n_resamples_used": len(diffs),
        "significant": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def single_model_bootstrap_ci(y_true, score, metric_fn, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_rows = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, n_rows, n_rows)
        y_s, s_s = y_true[idx], score[idx]
        if y_s.sum() == 0 or y_s.sum() == n_rows:
            continue
        vals.append(metric_fn(y_s, s_s))
    vals = np.array(vals)
    return {"ci_low": round(float(np.percentile(vals, 2.5)), 4), "ci_high": round(float(np.percentile(vals, 97.5)), 4)}


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    print(f"Loaded {len(df)} rows from {TEST_SCORED} (already scored, no new test read)")
    y = df["target"].values

    # ---------------------------------------------------------------
    # 3. Core test metrics per model + sub_grade benchmark setup
    # ---------------------------------------------------------------
    print("\n=== Core test metrics ===")
    results = {}
    calib_tables = {}
    for name, col in MODEL_COLS.items():
        results[name] = core_metrics(y, df[col].values)
        calib_tables[name] = calibration_table(y, df[col].values)
        print(f"{name}: {results[name]}")

    df["sub_grade_ordinal"] = sub_grade_ordinal(df["sub_grade"])

    # 5a: sub_grade rank AUC (ordinal score directly, no train fitting needed)
    sub_grade_auc = roc_auc_score(y, df["sub_grade_ordinal"].values)
    sub_grade_gini = 2 * sub_grade_auc - 1
    sub_grade_ks = ks_statistic(y, df["sub_grade_ordinal"].values)
    print(f"\nsub_grade (ordinal rank): AUC={sub_grade_auc:.4f}, Gini={sub_grade_gini:.4f}, KS={sub_grade_ks:.4f}")

    # 5b: sub_grade-to-PD map learned from TRAIN only
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    train = pd.read_parquet(TRAIN_PARQUET)
    train_cleaned = con.execute(
        """
        SELECT id, sub_grade FROM read_parquet('data/interim/accepted_2007_to_2018Q4_cleaned.parquet')
        WHERE term = 36 AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2013
        """
    ).fetchdf()
    train_with_sg = train[["id", "target"]].merge(train_cleaned, on="id", how="left")
    pd_map = train_with_sg.groupby("sub_grade")["target"].mean().to_dict()
    df["pd_sub_grade_mapped"] = df["sub_grade"].map(pd_map)
    sub_grade_calib = calibration_table(y, df["pd_sub_grade_mapped"].values)
    sub_grade_brier = brier_score_loss(y, df["pd_sub_grade_mapped"].values)
    sub_grade_bss = brier_skill_score(y, df["pd_sub_grade_mapped"].values)

    results["sub_grade"] = {
        "auc": round(sub_grade_auc, 4),
        "gini": round(sub_grade_gini, 4),
        "ks": round(sub_grade_ks, 4),
        "brier": round(sub_grade_brier, 5),
        "brier_skill_score": round(sub_grade_bss, 4),
        "mean_predicted_pd_pct": round(float(df["pd_sub_grade_mapped"].mean()) * 100, 2),
        "actual_default_rate_pct": round(float(y.mean()) * 100, 2),
        "note": "AUC/Gini/KS use sub_grade's ordinal rank directly (5a); Brier/calibration use the train-fit sub_grade-to-PD map (5b).",
    }

    with open(METRICS_DIR / "test_core_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    for name, table in calib_tables.items():
        table.to_csv(METRICS_DIR / f"test_calibration_{name}.csv")
    sub_grade_calib.to_csv(METRICS_DIR / "test_calibration_sub_grade.csv")

    # ---------------------------------------------------------------
    # 4. Bootstrap CIs
    # ---------------------------------------------------------------
    print("\n=== Bootstrap CIs (1000 resamples, seed=42) ===")
    ci_results = {}
    for name, col in MODEL_COLS.items():
        ci_results[name] = {
            "auc_ci": single_model_bootstrap_ci(y, df[col].values, roc_auc_score),
        }
    ci_results["sub_grade"] = {"auc_ci": single_model_bootstrap_ci(y, df["sub_grade_ordinal"].values, roc_auc_score)}

    paired = {
        "lightgbm_vs_scorecard": paired_bootstrap(y, df["pd_lightgbm"].values, df["pd_scorecard"].values, roc_auc_score),
        "lightgbm_vs_xgboost": paired_bootstrap(y, df["pd_lightgbm"].values, df["pd_xgboost"].values, roc_auc_score),
        "lightgbm_vs_sub_grade": paired_bootstrap(y, df["pd_lightgbm"].values, df["sub_grade_ordinal"].values, roc_auc_score),
    }
    print(json.dumps({"single_model_auc_ci": ci_results, "paired_auc_diff": paired}, indent=2))

    with open(METRICS_DIR / "test_bootstrap_ci.json", "w") as f:
        json.dump({"single_model_auc_ci": ci_results, "paired_auc_diff_ci": paired}, f, indent=2)

    # ---------------------------------------------------------------
    # 6. Stability: by issue quarter, by region
    # ---------------------------------------------------------------
    print("\n=== Stability: AUC by issue quarter (2015) ===")
    df["issue_quarter"] = pd.to_datetime(df["issue_d"]).dt.quarter
    quarter_rows = []
    for q in sorted(df["issue_quarter"].unique()):
        sub = df[df["issue_quarter"] == q]
        row = {"quarter": f"2015-Q{q}", "n": len(sub)}
        for name, col in MODEL_COLS.items():
            row[f"{name}_auc"] = round(roc_auc_score(sub["target"], sub[col]), 4)
        quarter_rows.append(row)
    quarter_df = pd.DataFrame(quarter_rows)
    quarter_df.to_csv(METRICS_DIR / "test_stability_by_quarter.csv", index=False)
    print(quarter_df.to_string(index=False))

    print("\n=== Stability (not fairness): AUC by Census region ===")
    df["region"] = df["addr_state"].map(CENSUS_REGION)
    region_rows = []
    for region in sorted(df["region"].dropna().unique()):
        sub = df[df["region"] == region]
        row = {"region": region, "n": len(sub)}
        for name, col in MODEL_COLS.items():
            row[f"{name}_auc"] = round(roc_auc_score(sub["target"], sub[col]), 4)
        region_rows.append(row)
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(METRICS_DIR / "test_stability_by_region.csv", index=False)
    print(region_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 7. Robustness: stricter label
    # ---------------------------------------------------------------
    print("\n=== Robustness: default-within-24-months label ===")
    months_to_last_pymnt = (
        (pd.to_datetime(df["last_pymnt_d"]).dt.year - pd.to_datetime(df["issue_d"]).dt.year) * 12
        + (pd.to_datetime(df["last_pymnt_d"]).dt.month - pd.to_datetime(df["issue_d"]).dt.month)
    )
    was_bad = df["target"] == 1
    fast_default = was_bad & (months_to_last_pymnt <= 24)
    slow_default_excluded = was_bad & ((months_to_last_pymnt > 24) | months_to_last_pymnt.isna())

    alt_df = df[~slow_default_excluded].copy()
    alt_df["target_alt"] = fast_default[~slow_default_excluded].astype(int)

    robustness_results = {
        "n_original_test": len(df),
        "n_excluded_slow_chargeoffs": int(slow_default_excluded.sum()),
        "n_alt_population": len(alt_df),
        "alt_default_rate_pct": round(100 * alt_df["target_alt"].mean(), 2),
        "metrics": {},
    }
    for name, col in MODEL_COLS.items():
        robustness_results["metrics"][name] = {
            "auc": round(roc_auc_score(alt_df["target_alt"], alt_df[col]), 4)
        }
    robustness_results["metrics"]["sub_grade"] = {
        "auc": round(roc_auc_score(alt_df["target_alt"], alt_df["sub_grade_ordinal"]), 4)
    }
    original_ranking = sorted(results.keys(), key=lambda k: results[k]["auc"], reverse=True)
    alt_ranking = sorted(robustness_results["metrics"].keys(), key=lambda k: robustness_results["metrics"][k]["auc"], reverse=True)
    robustness_results["original_ranking_by_auc"] = original_ranking
    robustness_results["alt_label_ranking_by_auc"] = alt_ranking
    robustness_results["ranking_holds"] = original_ranking == alt_ranking

    with open(METRICS_DIR / "test_robustness.json", "w") as f:
        json.dump(robustness_results, f, indent=2)
    print(json.dumps(robustness_results, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
