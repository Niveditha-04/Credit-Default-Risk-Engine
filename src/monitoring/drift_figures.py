"""Phase 9 close-out: two small PNG summaries, since the Evidently HTML
reports (31MB) are no longer committed. HISTORICAL REPLAY, not live
monitoring - see run_drift.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")


def plot_top_feature_psi() -> None:
    monthly = pd.read_csv(METRICS_DIR / "drift_data_monthly_psi.csv")
    crossings = pd.read_csv(METRICS_DIR / "drift_data_first_crossing.csv")
    top_features = crossings.sort_values("max_psi_2015_2018", ascending=False).head(6)["feature"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    for feat in top_features:
        series = monthly[monthly["feature"] == feat].sort_values("month")
        ax.plot(series["month"], series["psi"], marker="o", markersize=3, label=feat)

    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="watch (0.10)")
    ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="alert (0.25)")
    ax.set_title("Monthly PSI, the 6 features that reached alert level (2015-2018 historical replay)")
    ax.set_xlabel("month")
    ax.set_ylabel("PSI vs train reference")
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    out = FIG_DIR / "drift_top_feature_psi_2015_2018.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def plot_2015_predicted_vs_actual() -> None:
    perf = pd.read_csv(METRICS_DIR / "drift_performance_2015_monthly.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(perf["month"], perf["mean_predicted_pd_pct"], marker="o", label="mean predicted PD (%)")
    ax.plot(perf["month"], perf["actual_default_rate_pct"], marker="s", label="actual default rate (%)")
    ax.set_title("2015 only: mean predicted PD vs actual default rate, by month")
    ax.set_xlabel("month")
    ax.set_ylabel("percent")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = FIG_DIR / "drift_2015_predicted_vs_actual.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_feature_psi()
    plot_2015_predicted_vs_actual()


if __name__ == "__main__":
    main()
