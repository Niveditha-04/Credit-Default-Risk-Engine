"""Post-hoc validation checks on test (2015) and the replay window,
grouped by topic in one module instead of five separate scripts.
Every check here is a second, separate read of already-saved
predictions or already-saved data after the one-time evaluation that
chose and froze the champion; none of them changes a model, feature,
or policy decision. Each section logs its own access through
src.evaluation.test_access_log (the one shared loader for every check
that re-reads reports/metrics/test_scored.parquet).

Sections, run in order by main():
  1. Calibration detail: reliability plot, Expected Calibration Error,
     and calibration by issue month and by FICO band.
  2. Monthly lift: paired monthly AUC lift of LightGBM over sub_grade
     and the scorecard, by test issue month.
  3. Segment table: LightGBM AUC and default rate by FICO band,
     purpose, loan size band, and issue quarter.
  4. Replay performance: a fixed 24-month-horizon label for the one
     replay window (2016-01 to 2016-10) that has enough seasoning by
     this data's March 2019 snapshot to support it, compared against
     the same label re-applied to test.
  5. Walk-forward: a second fold, trained on 2012-2014 and evaluated on
     test (2015), with the frozen champion's own hyperparameters and no
     tuning search.
  6. ROC, KS, and cumulative-gains curves on test (2015) for v1
     LightGBM, the scorecard, and sub_grade's ordinal rank.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.score_models import (
    load_frozen_config,
    score_lightgbm,
    v1_lightgbm_cfg,
)
from src.evaluation.test_access_log import load_test_scored, log_access
from src.evaluation.test_evaluation import ks_statistic, sub_grade_ordinal
from src.models.gbm import feature_list
from src.models.scorecard import (
    IV_MIN_THRESHOLD,
    candidate_features,
    reduce_by_correlation,
    reduce_by_vif,
)

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_PARQUET = "data/processed/test.parquet"
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
REPLAY_PARQUET = "data/processed/replay.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
FICO_BINS = [660, 680, 700, 720, 740, 760, 850]
SEED = 42

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
FROZEN_N_TREES = 273


def join_sub_grade(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    sub_grade = con.execute(f"SELECT id, sub_grade FROM read_parquet('{CLEANED_PARQUET}')").fetchdf()
    return df.merge(sub_grade, on="id", how="left")


# ---------------------------------------------------------------------
# 1. Calibration detail (was posthoc_calibration.py)
# ---------------------------------------------------------------------


def expected_calibration_error(calib: pd.DataFrame) -> float:
    n_total = calib["n"].sum()
    weighted_gap = (
        calib["n"] * (calib["mean_predicted_pd_pct"] - calib["actual_default_rate_pct"]).abs()
    ).sum()
    return float(weighted_gap / n_total / 100)  # back to a 0-1 ECE, not a percentage-point gap


def run_calibration() -> None:
    import matplotlib.pyplot as plt

    calib = pd.read_csv(METRICS_DIR / "test_calibration_lightgbm.csv")
    ece = expected_calibration_error(calib)
    print(f"Expected Calibration Error (LightGBM, test, 10 bins): {ece:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 35], [0, 35], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(calib["mean_predicted_pd_pct"], calib["actual_default_rate_pct"], marker="o", label="LightGBM, test (2015)")
    ax.set_xlabel("Mean predicted PD (%)")
    ax.set_ylabel("Actual default rate (%)")
    ax.set_title(f"Reliability plot, LightGBM, test 2015 (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "posthoc_reliability_plot.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    df = load_test_scored(
        "posthoc_calibration.py",
        "Post-hoc: re-read saved test_scored.parquet (plus test.parquet for fico_mean, "
        "not stored in test_scored.parquet) to compute calibration by issue month and by "
        "FICO band. No model, feature, or policy change.",
    )
    test_feats = pd.read_parquet(TEST_PARQUET, columns=["id", "fico_mean"])
    df = df.merge(test_feats, on="id", how="left")

    df["month"] = pd.to_datetime(df["issue_d"]).dt.to_period("M").astype(str)
    monthly_rows = []
    for month, g in df.groupby("month"):
        monthly_rows.append({
            "month": month, "n": len(g),
            "mean_predicted_pd_pct": round(100 * g["pd_lightgbm"].mean(), 2),
            "actual_default_rate_pct": round(100 * g["target"].mean(), 2),
        })
    monthly_calib = pd.DataFrame(monthly_rows).sort_values("month")
    monthly_calib.to_csv(METRICS_DIR / "posthoc_calibration_by_month.csv", index=False)
    print("\n=== Calibration by issue month ===")
    print(monthly_calib.to_string(index=False))

    df["fico_band"] = pd.cut(df["fico_mean"], bins=FICO_BINS)
    fico_rows = []
    for band, g in df.groupby("fico_band", observed=True):
        fico_rows.append({
            "fico_band": str(band), "n": len(g),
            "mean_predicted_pd_pct": round(100 * g["pd_lightgbm"].mean(), 2),
            "actual_default_rate_pct": round(100 * g["target"].mean(), 2),
        })
    fico_calib = pd.DataFrame(fico_rows)
    fico_calib.to_csv(METRICS_DIR / "posthoc_calibration_by_fico_band.csv", index=False)
    print("\n=== Calibration by FICO band ===")
    print(fico_calib.to_string(index=False))

    summary = {
        "label": "post-hoc (by-month and by-FICO-band breakdowns; ECE and the reliability "
        "plot use only the already-committed 10-bin calibration table)",
        "ece": round(ece, 4),
        "n_bins": len(calib),
        "by_month_max_abs_gap_pct": round(
            float((monthly_calib["mean_predicted_pd_pct"] - monthly_calib["actual_default_rate_pct"]).abs().max()), 2
        ),
        "by_fico_band_max_abs_gap_pct": round(
            float((fico_calib["mean_predicted_pd_pct"] - fico_calib["actual_default_rate_pct"]).abs().max()), 2
        ),
    }
    with open(METRICS_DIR / "posthoc_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 2. Monthly lift (was posthoc_monthly_lift.py)
# ---------------------------------------------------------------------


def run_monthly_lift() -> None:
    df = load_test_scored(
        "posthoc_monthly_lift.py",
        "Post-hoc: re-read saved test_scored.parquet to compute paired monthly AUC lift "
        "by issue month. No model, feature, or policy change.",
    )
    df["month"] = pd.to_datetime(df["issue_d"]).dt.to_period("M").astype(str)
    df["sub_grade_ordinal"] = sub_grade_ordinal(df["sub_grade"])

    rows = []
    for month, g in df.groupby("month"):
        y = g["target"].values
        auc_lgb = roc_auc_score(y, g["pd_lightgbm"].values)
        auc_sub = roc_auc_score(y, g["sub_grade_ordinal"].values)
        auc_sc = roc_auc_score(y, g["pd_scorecard"].values)
        rows.append({
            "month": month,
            "n": len(g),
            "auc_lightgbm": round(auc_lgb, 4),
            "auc_sub_grade": round(auc_sub, 4),
            "auc_scorecard": round(auc_sc, 4),
            "lift_vs_sub_grade": round(auc_lgb - auc_sub, 4),
            "lift_vs_scorecard": round(auc_lgb - auc_sc, 4),
        })
    monthly = pd.DataFrame(rows).sort_values("month")
    monthly.to_csv(METRICS_DIR / "posthoc_monthly_lift.csv", index=False)
    print(monthly.to_string(index=False))

    def summarize(col):
        vals = monthly[col]
        return {
            "mean": round(float(vals.mean()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "n_months_positive": int((vals > 0).sum()),
            "n_months_total": len(vals),
        }

    summary = {
        "label": "post-hoc",
        "lift_vs_sub_grade": summarize("lift_vs_sub_grade"),
        "lift_vs_scorecard": summarize("lift_vs_scorecard"),
    }
    with open(METRICS_DIR / "posthoc_monthly_lift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 3. Segment table (was posthoc_segments.py)
# ---------------------------------------------------------------------

LOAN_SIZE_BINS = [0, 5000, 10000, 15000, 20000, 25000, 35000]
N_TOP_PURPOSES = 6


def segment_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(group_col, observed=True):
        rows.append({
            "segment": str(value),
            "n": len(g),
            "default_rate_pct": round(100 * g["target"].mean(), 2),
            "auc": round(roc_auc_score(g["target"], g["pd_lightgbm"]), 4) if g["target"].nunique() == 2 else None,
        })
    return pd.DataFrame(rows)


def run_segments() -> None:
    df = load_test_scored(
        "posthoc_segments.py",
        "Post-hoc: re-read saved test_scored.parquet (plus test.parquet for fico_mean "
        "and purpose, not stored in test_scored.parquet) to compute a segment table "
        "(FICO band, purpose, loan size band, issue quarter). No model, feature, or "
        "policy change.",
    )
    test_feats = pd.read_parquet(TEST_PARQUET, columns=["id", "fico_mean", "purpose"])
    df = df.merge(test_feats, on="id", how="left")

    df["fico_band"] = pd.cut(df["fico_mean"], bins=FICO_BINS)
    df["loan_size_band"] = pd.cut(df["loan_amnt"], bins=LOAN_SIZE_BINS)
    df["quarter"] = pd.to_datetime(df["issue_d"]).dt.to_period("Q").astype(str)

    top_purposes = df["purpose"].value_counts().head(N_TOP_PURPOSES).index.tolist()
    df["purpose_grouped"] = df["purpose"].where(df["purpose"].isin(top_purposes), "all_other_purposes_combined")

    tables = {
        "fico_band": segment_table(df, "fico_band"),
        "purpose": segment_table(df, "purpose_grouped"),
        "loan_size_band": segment_table(df, "loan_size_band"),
        "issue_quarter": segment_table(df, "quarter"),
    }

    all_rows = []
    for dimension, table in tables.items():
        table.insert(0, "dimension", dimension)
        all_rows.append(table)
        print(f"\n=== Segment: {dimension} ===")
        print(table.to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(METRICS_DIR / "posthoc_segment_table.csv", index=False)

    summary = {
        "label": "post-hoc",
        "top_purposes_by_volume": top_purposes,
        "fico_band_auc_range": [
            float(tables["fico_band"]["auc"].min()), float(tables["fico_band"]["auc"].max())
        ],
        "purpose_auc_range": [
            float(tables["purpose"]["auc"].min()), float(tables["purpose"]["auc"].max())
        ],
        "loan_size_band_auc_range": [
            float(tables["loan_size_band"]["auc"].min()), float(tables["loan_size_band"]["auc"].max())
        ],
        "issue_quarter_auc_range": [
            float(tables["issue_quarter"]["auc"].min()), float(tables["issue_quarter"]["auc"].max())
        ],
    }
    with open(METRICS_DIR / "posthoc_segment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 4. Replay performance (was posthoc_replay_performance.py)
# ---------------------------------------------------------------------

REPLAY_WINDOW_START = "2016-01-01"
REPLAY_WINDOW_END = "2016-10-31"  # last vintage with >= 24 + 5 months observable by the March 2019 snapshot
STILL_OPEN_STATUSES = ["Current", "Late (16-30 days)", "Late (31-120 days)", "In Grace Period"]


def label_24_month(loan_status: pd.Series, issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.DataFrame:
    months = (
        (pd.to_datetime(last_pymnt_d).dt.year - pd.to_datetime(issue_d).dt.year) * 12
        + (pd.to_datetime(last_pymnt_d).dt.month - pd.to_datetime(issue_d).dt.month)
    )
    was_bad = loan_status.isin(["Charged Off", "Default"])
    fast_default = was_bad & (months <= 24)
    cannot_label = was_bad & ((months > 24) | months.isna())
    label = fast_default.astype(float)
    label[cannot_label] = float("nan")
    return pd.DataFrame({"label_24m": label, "excluded": cannot_label})


def run_replay_performance() -> None:
    con = duckdb.connect()
    replay_window = con.execute(
        f"""
        SELECT id, issue_d, loan_status, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND issue_d >= '{REPLAY_WINDOW_START}' AND issue_d <= '{REPLAY_WINDOW_END}'
        """
    ).fetchdf()

    log_access(
        "posthoc_replay_performance.py",
        len(replay_window),
        "Post-hoc, exploratory: reads loan_status/last_pymnt_d for replay loans issued "
        "2016-01 to 2016-10 (every vintage in this window has >= 29 months of seasoning "
        "by the March 2019 snapshot, a 5-month buffer past the 24-month label horizon) "
        "to build a 24-month-horizon label for a partial, informal performance check. "
        "Not the project's target, not cached for reuse, no model, feature, or policy change.",
    )

    replay_window["quarter"] = pd.to_datetime(replay_window["issue_d"]).dt.to_period("Q").astype(str)
    still_open_rows = []
    for quarter, g in replay_window.groupby("quarter"):
        row = {"quarter": quarter, "n": len(g)}
        for status in STILL_OPEN_STATUSES:
            row[status] = int((g["loan_status"] == status).sum())
        still_open_rows.append(row)
    still_open_df = pd.DataFrame(still_open_rows)
    still_open_df.to_csv(METRICS_DIR / "posthoc_replay_still_open_by_vintage.csv", index=False)
    print("=== Loans by status at the March 2019 snapshot, by issue quarter (censoring check) ===")
    print(still_open_df.to_string(index=False))

    labelled = label_24_month(replay_window["loan_status"], replay_window["issue_d"], replay_window["last_pymnt_d"])
    replay_window = pd.concat([replay_window, labelled], axis=1)
    n_excluded = int(replay_window["excluded"].sum())

    replay_feats = pd.read_parquet(REPLAY_PARQUET)
    merged = replay_window.merge(replay_feats, on="id", how="inner", suffixes=("", "_feat"))
    merged = merged[~merged["excluded"]].copy()

    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")

    # This check is specifically about v1 (the champion at the time it was
    # written, same scope as the walk-forward check's fold 1 above):
    # v1_lightgbm_cfg, never lightgbm_chosen, which now describes v2.
    frozen = load_frozen_config()
    cfg = v1_lightgbm_cfg(frozen)
    merged["pd_lightgbm"] = score_lightgbm(merged, cfg)
    merged["quarter"] = pd.to_datetime(merged["issue_d"]).dt.to_period("Q").astype(str)

    replay_rows = []
    for quarter, g in merged.groupby("quarter"):
        replay_rows.append({
            "period": quarter, "source": "replay (post-hoc, 24-month label)",
            "n": len(g), "n_positive": int(g["label_24m"].sum()),
            "label_24m_rate_pct": round(100 * g["label_24m"].mean(), 2),
            "auc": round(roc_auc_score(g["label_24m"], g["pd_lightgbm"]), 4),
        })

    # Same 24-month label, applied to test (2015), by quarter, using the
    # existing one-time test scoring pass's saved predictions.
    test_scored = pd.read_parquet(METRICS_DIR / "test_scored.parquet")
    con2 = duckdb.connect()
    test_status = con2.execute(
        f"SELECT id, loan_status FROM read_parquet('{CLEANED_PARQUET}') "
        f"WHERE EXTRACT(year FROM issue_d) = 2015"
    ).fetchdf()
    test_merged = test_scored.merge(test_status, on="id", how="left")
    test_labelled = label_24_month(test_merged["loan_status"], test_merged["issue_d"], test_merged["last_pymnt_d"])
    test_merged = pd.concat([test_merged, test_labelled], axis=1)
    test_merged = test_merged[~test_merged["excluded"]].copy()
    test_merged["quarter"] = pd.to_datetime(test_merged["issue_d"]).dt.to_period("Q").astype(str)

    test_rows = []
    for quarter, g in test_merged.groupby("quarter"):
        test_rows.append({
            "period": quarter, "source": "test 2015 (same 24-month label, re-applied)",
            "n": len(g), "n_positive": int(g["label_24m"].sum()),
            "label_24m_rate_pct": round(100 * g["label_24m"].mean(), 2),
            "auc": round(roc_auc_score(g["label_24m"], g["pd_lightgbm"]), 4),
        })

    result = pd.DataFrame(test_rows + replay_rows)
    result.to_csv(METRICS_DIR / "posthoc_replay_24m_performance.csv", index=False)
    print(result.to_string(index=False))

    summary = {
        "label": "post-hoc, exploratory",
        "note": (
            "last_pymnt_d is an approximation of when a loan stopped performing, not an "
            "exact default date (LendingClub does not record one). A loan whose "
            "charge-off happened more than 24 months after issue, or whose last_pymnt_d "
            "is missing, cannot be placed inside or outside the 24-month window and is "
            "excluded from this check entirely (not counted as a 0 or a 1)."
        ),
        "replay_window": [REPLAY_WINDOW_START, REPLAY_WINDOW_END],
        "n_excluded_cannot_be_labelled": n_excluded,
        "n_scored": len(merged),
        "by_quarter": result.to_dict("records"),
        "replay_auc_range": [
            round(float(result[result["source"].str.startswith("replay")]["auc"].min()), 4),
            round(float(result[result["source"].str.startswith("replay")]["auc"].max()), 4),
        ],
        "test_2015_auc_range_same_label": [
            round(float(result[result["source"].str.startswith("test")]["auc"].min()), 4),
            round(float(result[result["source"].str.startswith("test")]["auc"].max()), 4),
        ],
    }
    with open(METRICS_DIR / "posthoc_replay_24m_performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 5. Walk-forward (was posthoc_walkforward.py)
# ---------------------------------------------------------------------


def fit_scorecard_pipeline(train: pd.DataFrame, evaluate: pd.DataFrame):
    """Replicates scorecard.py's main(): WOE binning -> IV filter ->
    correlation reduction -> VIF reduction -> sign-correction logistic
    regression, on whatever train data is passed in."""
    candidates = candidate_features()
    cat_in_candidates = [c for c in CATEGORICAL_COLUMNS if c in candidates]
    bp = BinningProcess(variable_names=candidates, categorical_variables=cat_in_candidates)
    bp.fit(train[candidates], train["target"].values)

    iv_table = bp.summary()
    iv_map = dict(zip(iv_table["name"], iv_table["iv"]))
    above_min = [c for c in candidates if iv_map.get(c, 0) >= IV_MIN_THRESHOLD]

    woe_train = pd.DataFrame(
        bp.transform(train[candidates], metric="woe"), columns=candidates, index=train.index
    )[above_min]
    after_corr, _ = reduce_by_correlation(woe_train, iv_map)
    final_features, _ = reduce_by_vif(woe_train[after_corr])

    y_train = train["target"].values
    model_features = list(final_features)
    while True:
        clf = LogisticRegression(penalty=None, max_iter=1000)
        clf.fit(woe_train[model_features], y_train)
        coefs = pd.Series(clf.coef_[0], index=model_features)
        positive = coefs[coefs > 0]
        if positive.empty:
            break
        worst = positive.abs().idxmin()
        model_features.remove(worst)
    final_features = model_features

    woe_eval = pd.DataFrame(
        bp.transform(evaluate[candidates], metric="woe"), columns=candidates, index=evaluate.index
    )[final_features]
    eval_pred = clf.predict_proba(woe_eval)[:, 1]
    eval_auc = float(roc_auc_score(evaluate["target"].values, eval_pred))
    return eval_auc, final_features


def fit_frozen_lightgbm_walkforward(train: pd.DataFrame, feats: list[str]):
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feats]
    dataset = lgb.Dataset(train[feats], label=train["target"].values, categorical_feature=cat_cols, free_raw_data=False)
    return lgb.train(FROZEN_LGB_PARAMS, dataset, num_boost_round=FROZEN_N_TREES)


def run_walkforward() -> None:
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    champion_feats = feature_list(list(train.columns))
    results = []

    # ---- Fold 1: train 2012-2013 -> evaluate 2014 (already fit, reuse) ----
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    # v1's own train/validation metrics moved to lightgbm_v1_history when
    # v2 was promoted (frozen_model_config.json's lightgbm_chosen now
    # describes v2); fold 1 is specifically v1's already-fit split, so it
    # reads from history, same number (0.6834) as before the promotion.
    lgb_fold1_auc = frozen["lightgbm_v1_history"]["train_validation_metrics"]["validation_auc"]
    with open(METRICS_DIR / "scorecard_validation_metrics.json") as f:
        sc_metrics = json.load(f)
    scorecard_fold1_auc = sc_metrics["validation"]["auc"]

    validation_sg = join_sub_grade(validation.copy())
    validation_sg["sub_grade_ordinal"] = sub_grade_ordinal(validation_sg["sub_grade"])
    sub_grade_fold1_auc = float(roc_auc_score(validation_sg["target"].values, validation_sg["sub_grade_ordinal"].values))

    results.append({
        "fold": 1, "train_years": "2012-2013", "evaluate_years": "2014", "evaluate_label": "primary (validation)",
        "lightgbm_auc": round(lgb_fold1_auc, 4), "scorecard_auc": round(scorecard_fold1_auc, 4),
        "sub_grade_auc": round(sub_grade_fold1_auc, 4),
        "lift_vs_scorecard": round(lgb_fold1_auc - scorecard_fold1_auc, 4),
        "lift_vs_sub_grade": round(lgb_fold1_auc - sub_grade_fold1_auc, 4),
    })

    # ---- Fold 2: train 2012-2014 -> evaluate 2015 (post-hoc) ----
    fold2_train = pd.concat([train, validation], ignore_index=True)
    for col in CATEGORICAL_COLUMNS:
        fold2_train[col] = fold2_train[col].astype("category")
    test_c = test.copy()
    for col in CATEGORICAL_COLUMNS:
        test_c[col] = test_c[col].astype("category").cat.set_categories(fold2_train[col].cat.categories)

    log_access(
        "posthoc_walkforward.py",
        len(test_c),
        "Post-hoc: reads test.parquet to evaluate a walk-forward fold (train 2012-2014, "
        "frozen LightGBM hyperparameters and tree count, no tuning; scorecard refit "
        "through its own pipeline) against test (2015). No change to the frozen champion.",
    )

    booster = fit_frozen_lightgbm_walkforward(fold2_train, champion_feats)
    lgb_pred = booster.predict(test_c[champion_feats], num_iteration=booster.best_iteration)
    lgb_fold2_auc = float(roc_auc_score(test_c["target"].values, lgb_pred))

    scorecard_fold2_auc, scorecard_fold2_feats = fit_scorecard_pipeline(fold2_train, test_c)

    test_sg = join_sub_grade(test_c.copy())
    test_sg["sub_grade_ordinal"] = sub_grade_ordinal(test_sg["sub_grade"])
    sub_grade_fold2_auc = float(roc_auc_score(test_sg["target"].values, test_sg["sub_grade_ordinal"].values))

    results.append({
        "fold": 2, "train_years": "2012-2014", "evaluate_years": "2015", "evaluate_label": "post-hoc (test)",
        "lightgbm_auc": round(lgb_fold2_auc, 4), "scorecard_auc": round(scorecard_fold2_auc, 4),
        "sub_grade_auc": round(sub_grade_fold2_auc, 4),
        "lift_vs_scorecard": round(lgb_fold2_auc - scorecard_fold2_auc, 4),
        "lift_vs_sub_grade": round(lgb_fold2_auc - sub_grade_fold2_auc, 4),
    })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    summary = {
        "label": "post-hoc (fold 2 only; fold 1 reuses the champion's own committed validation numbers)",
        "note": (
            "LightGBM reuses the frozen champion's exact hyperparameters AND exact tree "
            "count (273) in fold 2 - there is no non-test holdout left within 2012-2014 "
            "to early-stop against without using test for a design decision, so the "
            "champion's already-chosen tree count is reused directly instead of "
            "re-running early stopping. The scorecard reruns its full pipeline (WOE "
            "binning, IV filter, correlation reduction, VIF reduction, sign-correction "
            "logistic regression) on each fold's own train data; fold 2's final feature "
            "set is allowed to differ from the frozen scorecard's 17 features."
        ),
        "fold_2_scorecard_features": scorecard_fold2_feats,
        "folds": results,
        "lift_vs_scorecard_mean": round(float(np.mean([r["lift_vs_scorecard"] for r in results])), 4),
        "lift_vs_scorecard_range": [
            round(min(r["lift_vs_scorecard"] for r in results), 4),
            round(max(r["lift_vs_scorecard"] for r in results), 4),
        ],
        "lift_vs_sub_grade_mean": round(float(np.mean([r["lift_vs_sub_grade"] for r in results])), 4),
        "lift_vs_sub_grade_range": [
            round(min(r["lift_vs_sub_grade"] for r in results), 4),
            round(max(r["lift_vs_sub_grade"] for r in results), 4),
        ],
    }
    results_df.to_csv(METRICS_DIR / "posthoc_walkforward.csv", index=False)
    with open(METRICS_DIR / "posthoc_walkforward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# 6. ROC, KS, and cumulative-gains curves on test (2015)
# ---------------------------------------------------------------------


def gains_curve(y_true: np.ndarray, score: np.ndarray) -> tuple:
    """% of loans declined (highest-score-first) on the x-axis vs. % of
    actual defaults that declining would have caught, on the y-axis.
    Declining the highest-score N% first is the same ranking direction
    used throughout this project's decision-quality checks."""
    order = np.argsort(-score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    total_defaults = y_sorted.sum()
    cum_defaults = np.cumsum(y_sorted)
    pct_declined = np.arange(1, n + 1) / n * 100
    pct_defaults_caught = cum_defaults / total_defaults * 100
    return pct_declined, pct_defaults_caught


def run_roc_ks_gains() -> None:
    import matplotlib.pyplot as plt

    df = load_test_scored(
        "posthoc_validation.py:run_roc_ks_gains",
        "Post-hoc: re-read saved test_scored.parquet to draw ROC, KS, and cumulative "
        "-gains curves for v1 LightGBM, the scorecard, and sub_grade's ordinal rank on "
        "test (2015). No model, feature, or policy change; every AUC drawn here already "
        "matches reports/metrics/test_core_metrics.json.",
    )
    y = df["target"].values
    df["sub_grade_ordinal"] = sub_grade_ordinal(df["sub_grade"])

    models = {
        "LightGBM (v1, champion)": df["pd_lightgbm"].values,
        "Scorecard": df["pd_scorecard"].values,
        "sub_grade (ordinal rank)": df["sub_grade_ordinal"].values,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: ROC curves, all three models.
    ax = axes[0]
    for name, score in models.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC, test (2015)")
    ax.legend(fontsize=8)

    # Panel 2: KS curve, v1 LightGBM only, max gap marked.
    ax = axes[1]
    fpr, tpr, thresh = roc_curve(y, df["pd_lightgbm"].values)
    ks_gap = tpr - fpr
    max_idx = int(np.argmax(ks_gap))
    ks_value = float(ks_gap[max_idx])
    ax.plot(thresh, tpr, label="TPR (cumulative % defaults)")
    ax.plot(thresh, fpr, label="FPR (cumulative % non-defaults)")
    ax.plot(
        [thresh[max_idx], thresh[max_idx]], [fpr[max_idx], tpr[max_idx]],
        color="black", linestyle=":", label=f"KS={ks_value:.4f}",
    )
    ax.set_xlim(0, max(float(np.nanmax(thresh[np.isfinite(thresh)])), 1.0))
    ax.set_xlabel("Predicted PD threshold")
    ax.set_ylabel("Cumulative rate")
    ax.set_title("KS curve, LightGBM (v1), test (2015)")
    ax.legend(fontsize=8)

    assert round(ks_value, 4) == round(ks_statistic(y, df["pd_lightgbm"].values), 4)

    # Panel 3: cumulative gains, all three models.
    ax = axes[2]
    for name, score in models.items():
        pct_declined, pct_caught = gains_curve(y, score)
        ax.plot(pct_declined, pct_caught, label=name)
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("% of loans declined (highest score first)")
    ax.set_ylabel("% of actual defaults caught")
    ax.set_title("Cumulative gains, test (2015)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = FIG_DIR / "test_roc_ks_gains.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


def main() -> None:
    print("=== 1. Calibration detail ===")
    run_calibration()
    print("\n=== 2. Monthly lift ===")
    run_monthly_lift()
    print("\n=== 3. Segment table ===")
    run_segments()
    print("\n=== 4. Replay performance ===")
    run_replay_performance()
    print("\n=== 5. Walk-forward ===")
    run_walkforward()
    print("\n=== 6. ROC, KS, and cumulative-gains curves ===")
    run_roc_ks_gains()


if __name__ == "__main__":
    main()
