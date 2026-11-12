"""Post-hoc partial performance check on replay (2016-2018). Replay
carries no official target (see src/data/target.py) because most of it
has not had time to fully resolve by this data's March 2019 snapshot.
This script builds one narrow exception: loans issued January 2016
through March 2017 have all had at least 24 months to season by March
2019, so a FIXED 24-month-horizon default label (identical in spirit to
the stricter robustness check already applied to test) is derivable for
them without waiting for full resolution. This is a post-hoc,
exploratory label used only to compare AUC by issue quarter against the
same label applied to test (2015); it is not the project's target, it
is not cached anywhere for reuse, and it never feeds into the frozen
champion, any feature, or the deployed policy.

24-month label: for a loan issued at month 0,
  1 = charged off (loan_status Charged Off or Default) with
      last_pymnt_d within 24 months of issue_d
  0 = everything else (Fully Paid, or still Current/Late/In Grace
      Period after surviving past the 24-month mark without charging
      off) - every loan in this window has been observed at least 24
      months by the March 2019 snapshot, so "0" here means "did not
      charge off within the window", not "fully resolved good".
  excluded (cannot be labelled) = charged off, but either more than 24
      months after issue or with no recorded last_pymnt_d at all - a
      loan that failed slowly is neither a clean fast-default-1 nor a
      clean survivor-0, and last_pymnt_d is only an approximation of
      when a loan actually stopped performing, not a precise default
      date (LendingClub does not record one).
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.score_models import load_frozen_config, score_lightgbm
from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
REPLAY_PARQUET = "data/processed/replay.parquet"
CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
REPLAY_WINDOW_START = "2016-01-01"
REPLAY_WINDOW_END = "2017-03-31"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]


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


def main() -> None:
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
        "2016-01 to 2017-03 (the only replay window with >= 24 months of seasoning by "
        "the March 2019 snapshot) to build a 24-month-horizon label for a partial, "
        "informal performance check. Not the project's target, not cached for reuse, "
        "no model, feature, or policy change.",
    )

    labelled = label_24_month(replay_window["loan_status"], replay_window["issue_d"], replay_window["last_pymnt_d"])
    replay_window = pd.concat([replay_window, labelled], axis=1)
    n_excluded = int(replay_window["excluded"].sum())

    replay_feats = pd.read_parquet(REPLAY_PARQUET)
    merged = replay_window.merge(replay_feats, on="id", how="inner", suffixes=("", "_feat"))
    merged = merged[~merged["excluded"]].copy()

    for col in CATEGORICAL_COLUMNS:
        merged[col] = merged[col].astype("category")

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
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
    test_scored = pd.read_parquet(TEST_SCORED)
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


if __name__ == "__main__":
    main()
