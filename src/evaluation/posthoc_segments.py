"""Post-hoc segment table on test (2015): LightGBM AUC and default rate
by FICO band, purpose (top 6 by volume, the rest combined), loan size
band, and issue quarter. Reads only the already-saved
reports/metrics/test_scored.parquet, plus data/processed/test.parquet
for fico_mean and purpose (not stored in test_scored.parquet). No
model, feature, or policy change.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
TEST_PARQUET = "data/processed/test.parquet"

FICO_BINS = [660, 680, 700, 720, 740, 760, 850]
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


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    log_access(
        "posthoc_segments.py",
        len(df),
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


if __name__ == "__main__":
    main()
