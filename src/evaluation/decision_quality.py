"""Post-hoc, descriptive decision-quality view of the existing approval
cutoffs. Nothing here selects or changes a cutoff: the 70/80/90% and
chosen-policy cutoffs are the ones already set on validation
(reports/metrics/decision_policy.json) and already scored once on test
(reports/metrics/test_scored.parquet). This only describes what each
cutoff actually does to the champion's (LightGBM) book on test: how
many defaults it catches by declining, how many good loans are declined
for every default avoided, and the realized dollars on each side of
that trade.

"Approved" = pd_lightgbm <= the cutoff (chosen policy) or the lowest-PD
X% of test loans by rank (70/80/90%, same rank convention already used
in src/evaluation/economics.py's fixed-approval-rate views). "Declined"
is everyone else. Every dollar figure is ex-post realized (total_pymnt
minus fees minus funded_amnt), since test is already fully resolved -
this is "what this cutoff would have done to the historical book,"
not a forecast.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.test_access_log import log_access

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")
TEST_SCORED = METRICS_DIR / "test_scored.parquet"
FIXED_APPROVAL_RATES_PCT = [70, 80, 90]
SWEEP_APPROVAL_RATES_PCT = np.arange(1, 100, 1)


def add_realized_net_cash(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["realized_net_cash"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    return df


def decision_quality_row(df: pd.DataFrame, approved_mask: pd.Series, label: str, pd_cutoff: float) -> dict:
    """Pure function (no file I/O): decision-quality metrics for one
    approve/decline split of df. df must already have a 0/1 `target`
    column and a `realized_net_cash` column."""
    n = len(df)
    declined_mask = ~approved_mask
    n_approved = int(approved_mask.sum())
    n_declined = int(declined_mask.sum())
    assert n_approved + n_declined == n

    n_defaults_total = int(df["target"].sum())
    n_defaults_declined = int(df.loc[declined_mask, "target"].sum())
    n_defaults_approved = int(df.loc[approved_mask, "target"].sum())
    assert n_defaults_declined + n_defaults_approved == n_defaults_total

    n_good_declined = n_declined - n_defaults_declined

    capture_rate_pct = 100 * n_defaults_declined / n_defaults_total if n_defaults_total else float("nan")
    miss_rate_pct = 100 * n_defaults_approved / n_defaults_total if n_defaults_total else float("nan")

    precision_of_declines_pct = 100 * n_defaults_declined / n_declined if n_declined else float("nan")
    good_declined_per_default_avoided = (
        n_good_declined / n_defaults_declined if n_defaults_declined else float("nan")
    )

    loss_avoided_on_declined_defaults = -float(
        df.loc[declined_mask & (df["target"] == 1), "realized_net_cash"].sum()
    )
    interest_forgone_on_declined_good_loans = float(
        df.loc[declined_mask & (df["target"] == 0), "realized_net_cash"].sum()
    )
    realized_net_cash_approved_book = float(df.loc[approved_mask, "realized_net_cash"].sum())

    return {
        "policy": label,
        "pd_cutoff": round(float(pd_cutoff), 4),
        "n_total": n,
        "n_approved": n_approved,
        "n_declined": n_declined,
        "approval_rate_pct": round(100 * n_approved / n, 2),
        "default_rate_approved_pct": round(100 * df.loc[approved_mask, "target"].mean(), 2) if n_approved else float("nan"),
        "n_defaults_total": n_defaults_total,
        "n_defaults_captured_by_decline": n_defaults_declined,
        "n_defaults_missed_still_approved": n_defaults_approved,
        "default_capture_rate_pct": round(capture_rate_pct, 2),
        "default_miss_rate_pct": round(miss_rate_pct, 2),
        "precision_of_declines_pct": round(precision_of_declines_pct, 2),
        "good_loans_declined_per_default_avoided": round(good_declined_per_default_avoided, 3),
        "loss_dollars_avoided_on_declined_defaults": round(loss_avoided_on_declined_defaults, 2),
        "interest_income_forgone_on_declined_good_loans": round(interest_forgone_on_declined_good_loans, 2),
        "realized_net_cash_approved_book": round(realized_net_cash_approved_book, 2),
    }


def rank_based_approval(df: pd.DataFrame, rate_pct: float, pd_col: str = "pd_lightgbm") -> tuple[pd.Series, float]:
    """Approve the lowest-PD rate_pct% of loans by rank. Returns the
    approved mask (aligned to df's index) and the PD value at the
    approval boundary."""
    sorted_df = df.sort_values(pd_col, kind="mergesort")
    n_approve = int(len(df) * rate_pct / 100)
    approved_ids = set(sorted_df.index[:n_approve])
    mask = df.index.to_series().isin(approved_ids)
    cutoff = float(sorted_df[pd_col].iloc[n_approve - 1]) if n_approve else float("-inf")
    return mask, cutoff


def main() -> None:
    df = pd.read_parquet(TEST_SCORED)
    log_access(
        "decision_quality.py",
        len(df),
        "Post-hoc, descriptive only: re-read saved test_scored.parquet to describe what the "
        "existing 70/80/90% and chosen-policy cutoffs (set on validation) actually do to the "
        "champion's (v1) book on test - default capture, decline precision, and realized "
        "dollars on each side. No cutoff is selected or changed here.",
    )
    df = add_realized_net_cash(df)

    with open(METRICS_DIR / "decision_policy.json") as f:
        policy = json.load(f)
    chosen_cutoff = policy["chosen_pd_cutoff"]

    rows = []
    for rate_pct in FIXED_APPROVAL_RATES_PCT:
        mask, cutoff = rank_based_approval(df, rate_pct)
        rows.append(decision_quality_row(df, mask, f"{rate_pct}% approval rate (fixed)", cutoff))

    chosen_mask = df["pd_lightgbm"] <= chosen_cutoff
    rows.append(decision_quality_row(df, chosen_mask, "chosen policy (EV-maximizing, set on validation)", chosen_cutoff))

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "test_decision_quality.csv", index=False)
    print(out.to_string(index=False))

    # Sweep for the figure's curve - same rank-based convention, many more
    # points than the four reported cutoffs, purely descriptive.
    sweep_rows = []
    for rate_pct in SWEEP_APPROVAL_RATES_PCT:
        mask, cutoff = rank_based_approval(df, rate_pct)
        sweep_rows.append(decision_quality_row(df, mask, f"{rate_pct}% (sweep)", cutoff))
    sweep = pd.DataFrame(sweep_rows)

    chosen_row = out.iloc[-1]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(sweep["approval_rate_pct"], sweep["default_capture_rate_pct"], color="tab:blue", label="% of defaults caught by declining")
    ax1.set_xlabel("Approval rate (%)")
    ax1.set_ylabel("% of defaults caught by declining", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(sweep["approval_rate_pct"], sweep["good_loans_declined_per_default_avoided"], color="tab:orange", label="Good loans declined per default avoided")
    ax2.set_ylabel("Good loans declined per default avoided", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    for _, r in out.iloc[:-1].iterrows():
        ax1.axvline(r["approval_rate_pct"], color="gray", linestyle=":", linewidth=1)
    ax1.scatter([chosen_row["approval_rate_pct"]], [chosen_row["default_capture_rate_pct"]], color="black", zorder=5, label="Chosen policy")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.set_title("Decision-quality trade-off, champion (v1), test (2015)")
    fig.tight_layout()
    fig_path = FIG_DIR / "decision_tradeoff_curve.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")


if __name__ == "__main__":
    main()
