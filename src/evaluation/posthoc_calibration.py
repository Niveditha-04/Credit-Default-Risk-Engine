"""Post-hoc calibration detail for the champion on test (2015): a
reliability plot and Expected Calibration Error (ECE) from the already
-committed 10-bin calibration table, plus two further breakdowns not
computed elsewhere - calibration by issue month and by FICO band. The
by-month and by-FICO-band breakdowns read reports/metrics/test_scored.parquet
again (a second, separate read from the one-time evaluation scoring
pass) and data/processed/test.parquet (for fico_mean, not stored in
test_scored.parquet) - both post-hoc, no model, feature, or policy
change. The reliability plot and ECE use only the already-committed
reports/metrics/test_calibration_lightgbm.csv, no new test read.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
TEST_PARQUET = "data/processed/test.parquet"

FICO_BINS = [660, 680, 700, 720, 740, 760, 850]


def expected_calibration_error(calib: pd.DataFrame) -> float:
    n_total = calib["n"].sum()
    weighted_gap = (
        calib["n"] * (calib["mean_predicted_pd_pct"] - calib["actual_default_rate_pct"]).abs()
    ).sum()
    return float(weighted_gap / n_total / 100)  # back to a 0-1 ECE, not a percentage-point gap


def main() -> None:
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

    df = pd.read_parquet(TEST_SCORED)
    log_access(
        "posthoc_calibration.py",
        len(df),
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


if __name__ == "__main__":
    main()
