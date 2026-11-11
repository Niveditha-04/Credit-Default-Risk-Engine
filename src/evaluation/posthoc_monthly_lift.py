"""Post-hoc analysis: paired monthly AUC lift of LightGBM over sub_grade
and over the scorecard, by test (2015) issue month. Reads only from the
already-saved reports/metrics/test_scored.parquet - a second, separate
read from the one-time evaluation scoring pass, used only to check
whether the champion's pooled-year AUC advantage holds month by month.
No model, feature, or policy change.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.test_access_log import log_access
from src.evaluation.test_evaluation import sub_grade_ordinal

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    log_access(
        "posthoc_monthly_lift.py",
        len(df),
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


if __name__ == "__main__":
    main()
