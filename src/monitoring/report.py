"""Drift reporting: alert scan and PNG summaries. HISTORICAL REPLAY, not
live monitoring - see run_drift.py, which these read from.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.report_paths import metrics_path

FIG_DIR = Path("reports/figures")
EXTRA_DIR = Path("reports/extra")
WATCH_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25


def drift_alerts() -> None:
    """Reads the already-computed monthly PSI and prints every
    feature/month crossing the watch or alert threshold. Describes what
    already happened in 2015-01 through 2018-12, not a live alert feed."""
    data_drift = pd.read_csv(metrics_path("drift_data_monthly_psi.csv"))
    prediction_drift = pd.read_csv(metrics_path("drift_prediction_monthly.csv"))

    print("=== HISTORICAL REPLAY alert scan: 2015-01 to 2018-12 (not live monitoring) ===\n")

    print(f"--- Data drift: feature/month crossings (watch >= {WATCH_THRESHOLD}, alert >= {ALERT_THRESHOLD}) ---")
    crossings = data_drift[data_drift["psi"] >= WATCH_THRESHOLD].sort_values(["feature", "month"])
    if crossings.empty:
        print("No feature crossed the watch threshold in any month.")
    for _, row in crossings.iterrows():
        level = "ALERT" if row["psi"] >= ALERT_THRESHOLD else "WATCH"
        print(f"  [{level}] {row['month']}  {row['feature']:<35s} PSI={row['psi']:.4f}  (n={row['n']})")

    print(f"\n--- Prediction drift: month crossings (watch >= {WATCH_THRESHOLD}, alert >= {ALERT_THRESHOLD}) ---")
    pred_crossings = prediction_drift[prediction_drift["psi_vs_train"] >= WATCH_THRESHOLD]
    if pred_crossings.empty:
        print("No month crossed the watch threshold for the predicted-PD distribution.")
    for _, row in pred_crossings.iterrows():
        level = "ALERT" if row["psi_vs_train"] >= ALERT_THRESHOLD else "WATCH"
        print(f"  [{level}] {row['month']}  predicted PD distribution  PSI={row['psi_vs_train']:.4f}  "
              f"(mean predicted PD={row['mean_predicted_pd_pct']}%, n={row['n']})")

    n_feature_months_watch = (data_drift["psi"] >= WATCH_THRESHOLD).sum()
    n_feature_months_alert = (data_drift["psi"] >= ALERT_THRESHOLD).sum()
    n_features_ever_watch = crossings["feature"].nunique() if not crossings.empty else 0
    n_features_ever_alert = crossings[crossings["psi"] >= ALERT_THRESHOLD]["feature"].nunique() if not crossings.empty else 0

    print("\n--- Summary ---")
    print(f"Feature-months at watch or above: {n_feature_months_watch} / {len(data_drift)}")
    print(f"Feature-months at alert: {n_feature_months_alert} / {len(data_drift)}")
    print(f"Distinct features that ever reached watch: {n_features_ever_watch} / {data_drift['feature'].nunique()}")
    print(f"Distinct features that ever reached alert: {n_features_ever_alert} / {data_drift['feature'].nunique()}")


def plot_top_feature_psi() -> None:
    monthly = pd.read_csv(metrics_path("drift_data_monthly_psi.csv"))
    crossings = pd.read_csv(metrics_path("drift_data_first_crossing.csv"))
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
    perf = pd.read_csv(metrics_path("drift_performance_2015_monthly.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(perf["month"], perf["mean_predicted_pd_pct"], marker="o", label="mean predicted PD (%)")
    ax.plot(perf["month"], perf["actual_default_rate_pct"], marker="s", label="actual default rate (%)")
    ax.set_title("2015 only: mean predicted PD vs actual default rate, by month")
    ax.set_xlabel("month")
    ax.set_ylabel("percent")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    out = EXTRA_DIR / "drift_2015_predicted_vs_actual.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def drift_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_feature_psi()
    plot_2015_predicted_vs_actual()


def combined_psi_table() -> pd.DataFrame:
    """One row per feature: PSI against two different references -
    validation (2014) vs train, the closest-in-time reference available
    without touching test or replay, and the max PSI reached against
    train across the full 2015-2018 historical replay - next to that
    feature's SHAP rank, so a reader can see at a glance whether the
    features that drift most are the ones the model leans on most."""
    baseline = pd.read_csv(metrics_path("drift_baseline_psi_validation_vs_train.csv"))
    crossings = pd.read_csv(metrics_path("drift_data_first_crossing.csv"))
    shap_rank = pd.read_csv(metrics_path("shap_global_importance.csv"), index_col=0)["mean_abs_shap_rank"]

    table = baseline.merge(crossings, on="feature", how="outer")
    table["shap_rank"] = table["feature"].map(shap_rank)
    table = table.rename(columns={"band": "band_validation_vs_train"})
    table["band_max_2015_2018"] = pd.cut(
        table["max_psi_2015_2018"], bins=[-0.01, 0.10, 0.25, 10], labels=["stable", "watch", "alert"]
    )
    table = table.sort_values("shap_rank")
    table.to_csv(metrics_path("drift_combined_psi_table.csv"), index=False)
    return table


if __name__ == "__main__":
    drift_alerts()
    print()
    drift_figures()
    print()
    print(combined_psi_table().to_string(index=False))
