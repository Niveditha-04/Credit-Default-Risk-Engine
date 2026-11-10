"""Phase 9 alert script: reads the already-computed monthly PSI (from
run_drift.py) and prints every feature/month crossing the watch (0.10)
or alert (0.25) threshold. HISTORICAL REPLAY - this describes what
already happened in 2015-01 through 2018-12, not a live alert feed.
"""

from pathlib import Path

import pandas as pd

METRICS_DIR = Path("reports/metrics")
WATCH_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25


def main() -> None:
    data_drift = pd.read_csv(METRICS_DIR / "drift_data_monthly_psi.csv")
    prediction_drift = pd.read_csv(METRICS_DIR / "drift_prediction_monthly.csv")

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


if __name__ == "__main__":
    main()
