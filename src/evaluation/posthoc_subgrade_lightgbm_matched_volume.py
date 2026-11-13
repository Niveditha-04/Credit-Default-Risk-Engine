"""Post-hoc: sub_grade alone vs. sub_grade + LightGBM (LightGBM with the
champion's 69 features plus sub_grade as a feature - the same model as
ablation c in src/models/posthoc_ablations.py, refit here to get
per-loan predictions) at matched funding volume, on validation (2014)
and test (2015, post-hoc).

Why not a clean, fully-resolved holdout year instead of reusing
validation: 2016 loans are not fully resolved by this data's March 2019
snapshot (see src/data/target.py, src/evaluation/posthoc_replay_performance.py),
so realized_net_cash (built from total_pymnt, which keeps growing until
a loan resolves) would understate cash for loans still being paid down -
there is no year available in this data that is both unused by model
selection and fully resolved. Validation (2014) is reused with that
caveat stated plainly: it was the early-stopping target for this
model's own tree count, so its economics here are not out-of-sample in
the way test (2015) is.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list

METRICS_DIR = Path("reports/metrics")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

SELECTION_RATES_PCT = [70, 80, 90]
SEED = 42
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

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


def sub_grade_train_pd_map(train_sg: pd.DataFrame) -> dict:
    return train_sg.groupby("sub_grade")["target"].mean().to_dict()


def matched_volume_table(df: pd.DataFrame, model_cols: dict, label: str) -> pd.DataFrame:
    df = df.copy()
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    rows = []
    for name, col in model_cols.items():
        sorted_df = df.sort_values(col).reset_index(drop=True)
        for rate_pct in SELECTION_RATES_PCT:
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


def main() -> None:
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

    # Refit ablation c (same methodology as src/models/posthoc_ablations.py:
    # train 2012-2013, early-stopped against validation 2014).
    lgb_train = lgb.Dataset(
        train_sg[feats_plus_sub_grade], label=train_sg["target"].values, categorical_feature=cat_cols,
        free_raw_data=False, params={"feature_pre_filter": False},
    )
    lgb_val = lgb.Dataset(
        validation_sg[feats_plus_sub_grade], label=validation_sg["target"].values, reference=lgb_train, free_raw_data=False
    )
    booster = lgb.train(
        FROZEN_LGB_PARAMS, lgb_train, num_boost_round=MAX_BOOST_ROUNDS, valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    validation_sg["pd_sub_grade_plus_lightgbm"] = booster.predict(
        validation_sg[feats_plus_sub_grade], num_iteration=booster.best_iteration
    )
    test_sg["pd_sub_grade_plus_lightgbm"] = booster.predict(
        test_sg[feats_plus_sub_grade], num_iteration=booster.best_iteration
    )

    pd_map = sub_grade_train_pd_map(train_sg)
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


if __name__ == "__main__":
    main()
