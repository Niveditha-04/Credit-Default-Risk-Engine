"""Post-hoc walk-forward check with the frozen champion's own
hyperparameters (no tuning search).

Fold 1 is exactly the champion's own train (2012-2013) -> validation
(2014) split, already fit and committed - this fold reuses those
numbers rather than refitting anything, plus a fresh (cheap, no
fitting) sub_grade ordinal-rank AUC on validation.

Fold 2 trains fresh on 2012-2014 (train + validation concatenated) and
evaluates on test (2015) - a genuinely new, separate, post-hoc read of
test, used only to check whether the lift over sub_grade/scorecard
holds a year later with a year more training data, never to pick or
tune a model. The frozen champion is not touched by this file.

LightGBM reuses the champion's exact hyperparameters AND exact tree
count (273) for fold 2: there is no legitimate non-test holdout left
within 2012-2014 to early-stop against without leaking a design
decision from test, so the already-chosen tree count is reused directly
instead of re-running early stopping.

The scorecard goes through the SAME pipeline as scorecard.py's main()
on each fold's own train data: WOE binning, IV filter, correlation
reduction, VIF reduction, sign-correction logistic regression. Fold 2's
final feature set is allowed to differ from fold 1's (the frozen
scorecard's 17 features) - whatever that pipeline selects on 2012-2014
data.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation.test_access_log import log_access
from src.evaluation.test_evaluation import sub_grade_ordinal
from src.models.gbm import feature_list
from src.models.scorecard import (
    IV_MIN_THRESHOLD,
    candidate_features,
    reduce_by_correlation,
    reduce_by_vif,
)

METRICS_DIR = Path("reports/metrics")
TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
TEST_PARQUET = "data/processed/test.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
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


def fit_frozen_lightgbm(train: pd.DataFrame, feats: list[str]):
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feats]
    dataset = lgb.Dataset(train[feats], label=train["target"].values, categorical_feature=cat_cols, free_raw_data=False)
    return lgb.train(FROZEN_LGB_PARAMS, dataset, num_boost_round=FROZEN_N_TREES)


def main() -> None:
    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    champion_feats = feature_list(list(train.columns))
    results = []

    # ---- Fold 1: train 2012-2013 -> evaluate 2014 (already fit, reuse) ----
    with open(METRICS_DIR / "frozen_model_config.json") as f:
        frozen = json.load(f)
    lgb_fold1_auc = frozen["lightgbm_chosen"]["train_validation_metrics"]["validation_auc"]
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

    booster = fit_frozen_lightgbm(fold2_train, champion_feats)
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


if __name__ == "__main__":
    main()
