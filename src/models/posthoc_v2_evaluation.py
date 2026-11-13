"""Evaluates v1 (the frozen champion, unchanged) and v2 (defined in
docs/v2_decision_rule.md, committed before this script was run) against
the pre-registered adoption rule in that same file. v2 is fit fresh, in
memory, for this evaluation only - it is never written to
data/processed/models/ and never registered in MLflow as a champion.

Population: 2016 vintages, issue_d <= 2016-10-31 (every loan here has
>= 29 months of seasoning by this data's March 2019 snapshot), term 36,
individual application - the same restricted, censoring-safe window
src/evaluation/posthoc_replay_performance.py uses. Label: the same fixed
24-month-horizon default label used throughout this project's post-hoc
replay checks.

This is explicitly post-hoc and exploratory. It never changes the frozen
v1 champion, its MLflow alias, or the deployed policy - "adoption" here
means only recording the rule's pass/fail outcome.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list

METRICS_DIR = Path("reports/metrics")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
REPLAY_PARQUET = "data/processed/replay.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

EVAL_WINDOW_START = "2016-01-01"
EVAL_WINDOW_END = "2016-10-31"
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
MONOTONE_DIRECTIONS = {
    "fico_mean": -1,
    "dti": 1,
    "loan_to_income": 1,
    "revol_util": 1,
    "bc_util": 1,
    "inq_last_6mths": 1,
    "acc_open_past_24mths": 1,
    "annual_inc": -1,
}

# Adoption rule, reports/metrics/v2_decision_rule.md / docs/v2_decision_rule.md.
AUC_SLACK = 0.002
ECE_SLACK = 0.005


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


def expected_calibration_error(y: pd.Series, pred, n_bins: int = 10) -> float:
    df = pd.DataFrame({"pred": pred, "y": y})
    df["bin"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    table = df.groupby("bin").agg(n=("y", "size"), mean_pred=("pred", "mean"), mean_actual=("y", "mean"))
    weighted_gap = (table["n"] * (table["mean_pred"] - table["mean_actual"]).abs()).sum()
    return float(weighted_gap / table["n"].sum())


def build_eval_population() -> pd.DataFrame:
    con = duckdb.connect()
    window = con.execute(
        f"""
        SELECT id, issue_d, loan_status, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND issue_d >= '{EVAL_WINDOW_START}' AND issue_d <= '{EVAL_WINDOW_END}'
        """
    ).fetchdf()
    labelled = label_24_month(window["loan_status"], window["issue_d"], window["last_pymnt_d"])
    window = pd.concat([window, labelled], axis=1)
    window = window[~window["excluded"]].copy()

    feats = pd.read_parquet(REPLAY_PARQUET)
    merged = window.merge(feats, on="id", how="inner", suffixes=("", "_feat"))
    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")
    return merged


def fit_v2(champion_feats: list[str]) -> lgb.Booster:
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    fold_train = pd.concat([train, validation], ignore_index=True)
    for col in CATEGORICAL_COLUMNS:
        fold_train[col] = fold_train[col].astype("category")

    monotone_constraints = [MONOTONE_DIRECTIONS.get(f, 0) for f in champion_feats]
    dataset = lgb.Dataset(
        fold_train[champion_feats], label=fold_train["target"].values,
        categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False,
    )
    params = dict(FROZEN_LGB_PARAMS)
    params["monotone_constraints"] = monotone_constraints
    return lgb.train(params, dataset, num_boost_round=FROZEN_N_TREES)


def main() -> None:
    population = build_eval_population()
    log_access(
        "posthoc_v2_evaluation.py",
        len(population),
        "Post-hoc, exploratory: scores v1 (frozen champion, unchanged) and a freshly-fit, "
        "in-memory-only v2 on 2016 vintages (issue <= 2016-10-31, 24-month label) to "
        "evaluate the pre-registered adoption rule in docs/v2_decision_rule.md, committed "
        "before this script was run. No change to the frozen v1 champion.",
    )

    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    population["pd_v1"] = score_lightgbm(population, cfg)

    v2_booster = fit_v2(champion_feats)
    population["pd_v2"] = v2_booster.predict(population[champion_feats], num_iteration=v2_booster.best_iteration)

    y = population["label_24m"]
    auc_v1 = round(float(roc_auc_score(y, population["pd_v1"])), 4)
    auc_v2 = round(float(roc_auc_score(y, population["pd_v2"])), 4)
    ece_v1 = round(expected_calibration_error(y, population["pd_v1"]), 4)
    ece_v2 = round(expected_calibration_error(y, population["pd_v2"]), 4)

    auc_condition = auc_v2 >= auc_v1 - AUC_SLACK
    ece_condition = ece_v2 <= ece_v1 + ECE_SLACK
    adopt = bool(auc_condition and ece_condition)

    result = {
        "label": "post-hoc, exploratory",
        "rule_file": "docs/v2_decision_rule.md",
        "evaluation_population": {
            "window": [EVAL_WINDOW_START, EVAL_WINDOW_END],
            "n_scored": len(population),
            "label_24m_rate_pct": round(100 * float(y.mean()), 2),
        },
        "v1": {"auc": auc_v1, "ece": ece_v1},
        "v2": {"auc": auc_v2, "ece": ece_v2},
        "rule": {
            "auc_condition": f"AUC(v2) >= AUC(v1) - {AUC_SLACK}",
            "auc_condition_value": f"{auc_v2} >= {round(auc_v1 - AUC_SLACK, 4)}",
            "auc_condition_met": auc_condition,
            "ece_condition": f"ECE(v2) <= ECE(v1) + {ECE_SLACK}",
            "ece_condition_value": f"{ece_v2} <= {round(ece_v1 + ECE_SLACK, 4)}",
            "ece_condition_met": ece_condition,
        },
        "adopt_v2": adopt,
        "note": (
            "Adoption here means recording this decision and the numbers it was made on. "
            "It does not change the frozen v1 champion's artifacts, MLflow alias, or "
            "served model - promoting an actual v2 champion is a separate, later action."
        ),
    }
    with open(METRICS_DIR / "posthoc_v2_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
