"""Drift reporting: alert scan, PNG summaries, and the
loan_to_income PSI explanation. HISTORICAL REPLAY, not live monitoring -
see run_drift.py, which all three of these read from.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.monitoring.psi import bin_counts_numeric, fit_numeric_bins, psi_from_counts
from src.monitoring.run_drift import TRAIN_PSI_REFERENCE_CUTOFF

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
WATCH_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25


def drift_alerts() -> None:
    """Reads the already-computed monthly PSI and prints every
    feature/month crossing the watch or alert threshold. Describes what
    already happened in 2015-01 through 2018-12, not a live alert feed."""
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


def drift_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_feature_psi()
    plot_2015_predicted_vs_actual()


def explain_loan_to_income_psi() -> None:
    """Why loan_to_income's hard 0.5 cap (confirmed in run_drift.py) only
    reaches PSI "watch" level, never "alert", despite being a real and
    visible policy signal in the raw data."""
    train_full = pd.read_parquet("data/processed/train.parquet")
    train_ref = train_full[train_full["issue_d"] >= TRAIN_PSI_REFERENCE_CUTOFF]
    edges = fit_numeric_bins(train_ref["loan_to_income"])

    test = pd.read_parquet("data/processed/test.parquet")
    tc = bin_counts_numeric(train_ref["loan_to_income"], edges)
    cc = bin_counts_numeric(test["loan_to_income"], edges)

    pct_near_cap_train = round(100 * float(((train_ref["loan_to_income"] > 0.495) & (train_ref["loan_to_income"] < 0.505)).mean()), 3)
    pct_near_cap_test = round(100 * float(((test["loan_to_income"] > 0.495) & (test["loan_to_income"] < 0.505)).mean()), 3)

    explanation = (
        f"The top decile bin edge (fit on train) is {edges[-2]:.4f}, so the top bin covers "
        f"[{edges[-2]:.4f}, inf). That one wide bin holds every value from an ordinary loan at "
        f"0.35 up to the capped spike at exactly 0.5, and, once the cap lifts in 2017, values up "
        f"to 1.38. The cap affects only {pct_near_cap_train}% of the train reference and "
        f"{pct_near_cap_test}% of 2015 rows (both within 0.005 of 0.5) - a small fraction sitting "
        f"inside a bin that already holds about 10% of the population by construction. PSI "
        f"compares bin proportions, not the shape within a bin, so a thin spike at one point "
        f"inside a wide bin barely moves that bin's share."
    )

    result = {
        "question": "did loan_to_income cross any PSI threshold, and if not (or only mildly), why",
        "answer": "loan_to_income reached watch level (max PSI 0.1202, first crossed 0.10 in 2017-05) "
        "but never reached alert (0.25) in any of the 48 months",
        "why_the_cap_is_diluted": explanation,
        "top_decile_bin_edge": round(float(edges[-2]), 4),
        "pct_of_train_ref_near_0.5": pct_near_cap_train,
        "pct_of_test_2015_near_0.5": pct_near_cap_test,
        "train_top_bin_share": round(float(tc[-2] / tc.sum()), 4),
        "test_2015_top_bin_share": round(float(cc[-2] / cc.sum()), 4),
        "psi_train_vs_test_2015_only": round(psi_from_counts(tc, cc), 4),
        "first_month_crossed_0.10": "2017-05",
        "note": "PSI rises later in 2017, once the cap starts lifting and the top bin genuinely "
        "gains mass above 0.5-1.3 - not from the capped spike itself.",
    }
    with open(METRICS_DIR / "drift_loan_to_income_psi_explanation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    drift_alerts()
    print()
    drift_figures()
    print()
    explain_loan_to_income_psi()
