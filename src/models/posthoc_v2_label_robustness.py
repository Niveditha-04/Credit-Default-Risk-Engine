"""Post-hoc, exploratory robustness check on the pre-registered v2
decision (docs/v2_decision_rule.md, reports/metrics/posthoc_v2_evaluation.json):
reruns v1 vs v2 on the same 2016-vintage population (issue <= 2016-10-31)
with a SECOND, "corrected" 24-month label, to check whether the recorded
decision (adopt v2) depends on the exact handling of charge-offs the
original label excluded.

The label definition used here was chosen AFTER the pre-registered
evaluation already ran and reported its result - this script does not
change that result (reports/metrics/posthoc_v2_evaluation.json is left
exactly as it was written). It is reported here, side by side, purely
as a robustness check.

Original label (docs/v2_decision_rule.md, already evaluated):
    1 = charged off within 24 months of issue
    0 = not charged off, observed past 24 months
    excluded = charged off more than 24 months after issue, or with no
        recorded last_pymnt_d at all

Corrected label (this script, new):
    1 = charged off within 24 months of issue, OR charged off with no
        recorded last_pymnt_d at all (no payment ever made reads as an
        immediate, fast failure, not an unknown-timing one)
    0 = charged off more than 24 months after issue (it survived the
        24-month window before eventually failing), OR not charged off
    No exclusions: every loan in the population gets a label.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access
from src.models.gbm import feature_list
from src.models.posthoc_v2_evaluation import (
    CATEGORICAL_COLUMNS,
    CLEANED_PARQUET,
    EVAL_WINDOW_END,
    EVAL_WINDOW_START,
    REPLAY_PARQUET,
    TRAIN_PARQUET,
    expected_calibration_error,
    fit_v2,
)

METRICS_DIR = Path("reports/metrics")


def label_24_month_corrected(loan_status: pd.Series, issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.Series:
    months = (
        (pd.to_datetime(last_pymnt_d).dt.year - pd.to_datetime(issue_d).dt.year) * 12
        + (pd.to_datetime(last_pymnt_d).dt.month - pd.to_datetime(issue_d).dt.month)
    )
    was_bad = loan_status.isin(["Charged Off", "Default"])
    fast_default = was_bad & (months <= 24)
    no_payment_ever = was_bad & months.isna()
    return (fast_default | no_payment_ever).astype(float)


def label_24_month_original(loan_status: pd.Series, issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.DataFrame:
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


def build_population_both_labels() -> pd.DataFrame:
    con = duckdb.connect()
    window = con.execute(
        f"""
        SELECT id, issue_d, loan_status, last_pymnt_d
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE term = 36 AND application_type = 'Individual'
          AND issue_d >= '{EVAL_WINDOW_START}' AND issue_d <= '{EVAL_WINDOW_END}'
        """
    ).fetchdf()

    original = label_24_month_original(window["loan_status"], window["issue_d"], window["last_pymnt_d"])
    window["label_24m_original"] = original["label_24m"]
    window["excluded_original"] = original["excluded"]
    window["label_24m_corrected"] = label_24_month_corrected(window["loan_status"], window["issue_d"], window["last_pymnt_d"])

    feats = pd.read_parquet(REPLAY_PARQUET)
    merged = window.merge(feats, on="id", how="inner", suffixes=("", "_feat"))
    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")
    return merged


def main() -> None:
    population = build_population_both_labels()
    log_access(
        "posthoc_v2_label_robustness.py",
        len(population),
        "Post-hoc, exploratory robustness check (does not change the already-recorded "
        "pre-registered v2 decision): rescores v1 and a freshly-refit v2 on the same "
        "2016-vintage population with a second, corrected 24-month label, chosen after "
        "the pre-registered evaluation already ran. No change to the frozen v1 champion.",
    )

    train = pd.read_parquet(TRAIN_PARQUET)
    champion_feats = feature_list(list(train.columns))

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    population["pd_v1"] = score_lightgbm(population, cfg)

    v2_booster = fit_v2(champion_feats)
    population["pd_v2"] = v2_booster.predict(population[champion_feats], num_iteration=v2_booster.best_iteration)

    results = {}
    for label_name, label_col, excluded_col in [
        ("original", "label_24m_original", "excluded_original"),
        ("corrected", "label_24m_corrected", None),
    ]:
        if excluded_col is not None:
            scored = population[~population[excluded_col]]
        else:
            scored = population
        y = scored[label_col]
        auc_v1 = round(float(roc_auc_score(y, scored["pd_v1"])), 4)
        auc_v2 = round(float(roc_auc_score(y, scored["pd_v2"])), 4)
        ece_v1 = round(expected_calibration_error(y, scored["pd_v1"]), 4)
        ece_v2 = round(expected_calibration_error(y, scored["pd_v2"]), 4)
        auc_condition = auc_v2 >= auc_v1 - 0.002
        ece_condition = ece_v2 <= ece_v1 + 0.005
        results[label_name] = {
            "n_scored": len(scored),
            "label_rate_pct": round(100 * float(y.mean()), 2),
            "v1": {"auc": auc_v1, "ece": ece_v1},
            "v2": {"auc": auc_v2, "ece": ece_v2},
            "auc_condition_met": auc_condition,
            "ece_condition_met": ece_condition,
            "adopt_v2_under_this_label": bool(auc_condition and ece_condition),
        }

    summary = {
        "label": "post-hoc, exploratory robustness check - does not change the pre-registered decision",
        "pre_registered_result_untouched": "reports/metrics/posthoc_v2_evaluation.json",
        "note": (
            "The corrected label was chosen after the pre-registered run already reported "
            "its result. It reclassifies two cases the original label excluded: a "
            "charge-off more than 24 months after issue now counts as 0 (survived the "
            "24-month window), and a charge-off with no recorded last_pymnt_d now counts "
            "as 1 (no payment ever made reads as an immediate failure)."
        ),
        "results_by_label": results,
        "decision_changes": results["original"]["adopt_v2_under_this_label"] != results["corrected"]["adopt_v2_under_this_label"],
    }
    with open(METRICS_DIR / "posthoc_v2_label_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
