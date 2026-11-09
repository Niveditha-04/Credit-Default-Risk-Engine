"""Baseline scorecard: WOE binning (train only) -> IV-based feature
selection -> correlation/VIF reduction -> logistic regression -> points
scaling -> evaluation on validation (2014). Test (2015) is never touched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.features.allow_list import feature_columns

TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
METRICS_DIR = Path("reports/metrics")

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]

ENGINEERED_NUMERIC = [
    "fico_mean",
    "credit_history_months",
    "log_annual_inc",
    "loan_to_income",
    "revol_bal_to_income",
]

# The 4 never_flag__* columns kept as candidates: their source mths_since_*
# column is 100% missing in Jan 2012 (rollout era, verified in Phase 2), so
# the source's own "missing" bin mixes two different populations -
# borrowers who genuinely never had the event, and borrowers whose status
# is simply unknown because the field didn't exist yet. The flag separates
# these (0 / 1 / NULL-for-unknown); the raw source cannot. See
# reports/metrics/missing_bin_diagnostic.csv for the evidence.
ROLLOUT_AFFECTED_SOURCES = [
    "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
]
ENGINEERED_FLAGS_KEPT = [f"never_flag__{c}" for c in ROLLOUT_AFFECTED_SOURCES]

# The other 2 never_flag__* columns are dropped: their source column was
# already non-null in January 2012 (no rollout), so flag=1 is an exact,
# redundant restatement of "source IS NULL" - WOE-binning the raw source
# already gives the missing bin its own WOE, which is strictly more
# information (plus real numeric gradation among non-missing values) than
# the binary flag alone.
NON_ROLLOUT_SOURCES = ["mths_since_last_delinq", "mths_since_last_record"]
ENGINEERED_FLAGS_DROPPED = [f"never_flag__{c}" for c in NON_ROLLOUT_SOURCES]

IV_MIN_THRESHOLD = 0.02  # convention: below this, a feature is "useless" by IV
IV_SUSPICIOUS_THRESHOLD = 0.5  # convention: above this, investigate for leakage
CORR_THRESHOLD = 0.7
VIF_THRESHOLD = 5.0

# Points scaling conventions (stated, not derived)
PDO = 20
BASE_POINTS = 600
BASE_ODDS = 50  # 50:1 good:bad at BASE_POINTS


def candidate_features() -> list[str]:
    raw = [c for c in feature_columns() if c not in ROLLOUT_AFFECTED_SOURCES]
    return raw + ENGINEERED_NUMERIC + ENGINEERED_FLAGS_KEPT


def missing_bin_diagnostic(train: pd.DataFrame) -> pd.DataFrame:
    """Task: for the rollout-era columns and mths_since_last_delinq, show
    the WOE and default rate of the missing bin, on the RAW source column
    (diagnostic only - these raw columns are not used as model features for
    the 4 rollout-affected ones)."""
    rows = []
    for src in ROLLOUT_AFFECTED_SOURCES + NON_ROLLOUT_SOURCES + ["mths_since_last_record"]:
        if src not in train.columns:
            continue
        bp = BinningProcess(variable_names=[src])
        bp.fit(train[[src]], train["target"].values)
        table = bp.get_binned_variable(src).binning_table.build()
        missing_row = table[table["Bin"] == "Missing"].iloc[0]
        is_rollout = src in ROLLOUT_AFFECTED_SOURCES
        rows.append(
            {
                "source_column": src,
                "is_rollout_affected": is_rollout,
                "missing_bin_count": int(missing_row["Count"]),
                "missing_bin_pct_of_train": round(missing_row["Count"] / len(train) * 100, 2),
                "missing_bin_default_rate_pct": round(missing_row["Event rate"] * 100, 2),
                "missing_bin_woe": round(missing_row["WoE"], 4),
                "train_overall_default_rate_pct": round(train["target"].mean() * 100, 2),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset="source_column")
    df.to_csv(METRICS_DIR / "missing_bin_diagnostic.csv", index=False)
    return df


def compute_vif(woe_df: pd.DataFrame) -> pd.Series:
    """VIF via R^2 of each feature regressed on the rest (no statsmodels
    dependency, just sklearn linear regression)."""
    from sklearn.linear_model import LinearRegression

    vifs = {}
    cols = list(woe_df.columns)
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        lr = LinearRegression().fit(woe_df[others], woe_df[col])
        r2 = lr.score(woe_df[others], woe_df[col])
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs)


def reduce_by_correlation(woe_df: pd.DataFrame, iv_map: dict) -> tuple[list[str], list[dict]]:
    """Iteratively drop the lower-IV member of any |r| > CORR_THRESHOLD pair."""
    remaining = list(woe_df.columns)
    log = []
    while True:
        corr = woe_df[remaining].corr().abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0)
        max_corr = corr.max() if len(remaining) > 1 else 0
        if max_corr <= CORR_THRESHOLD:
            break
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        col_a, col_b = remaining[i], remaining[j]
        drop = col_a if iv_map[col_a] < iv_map[col_b] else col_b
        keep = col_b if drop == col_a else col_a
        log.append(
            {
                "dropped": drop,
                "kept": keep,
                "correlation": round(max_corr, 3),
                "dropped_iv": round(iv_map[drop], 4),
                "kept_iv": round(iv_map[keep], 4),
            }
        )
        remaining.remove(drop)
    return remaining, log


def reduce_by_vif(woe_df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    remaining = list(woe_df.columns)
    log = []
    while len(remaining) > 1:
        vifs = compute_vif(woe_df[remaining])
        worst = vifs.idxmax()
        if vifs[worst] <= VIF_THRESHOLD:
            break
        log.append({"dropped": worst, "vif": round(float(vifs[worst]), 2)})
        remaining.remove(worst)
    return remaining, log


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PARQUET)
    validation = pd.read_parquet(VALIDATION_PARQUET)

    print("=== Step 1: missing-bin diagnostic (rollout-era + last_delinq) ===")
    diag = missing_bin_diagnostic(train)
    print(diag.to_string(index=False))

    candidates = candidate_features()
    print(f"\n=== Step 2: binning {len(candidates)} candidate features on train only ===")

    bp = BinningProcess(
        variable_names=candidates,
        categorical_variables=[c for c in CATEGORICAL_COLUMNS if c in candidates],
    )
    bp.fit(train[candidates], train["target"].values)

    iv_table = bp.summary().sort_values("iv", ascending=False)
    iv_table.to_csv(METRICS_DIR / "iv_table.csv", index=False)
    print(iv_table[["name", "dtype", "n_bins", "iv"]].to_string(index=False))

    print("\n=== Step 3: investigate IV > 0.5 ===")
    suspicious = iv_table[iv_table["iv"] > IV_SUSPICIOUS_THRESHOLD]
    suspicious_report = []
    for _, row in suspicious.iterrows():
        name = row["name"]
        table = bp.get_binned_variable(name).binning_table.build()
        suspicious_report.append({"feature": name, "iv": row["iv"], "bins": table.to_dict("records")})
        print(f"  {name}: IV={row['iv']:.4f}")
        print(table.to_string(index=False))
    with open(METRICS_DIR / "iv_suspicious_investigation.json", "w") as f:
        json.dump(suspicious_report, f, indent=2, default=str)

    iv_map = dict(zip(iv_table["name"], iv_table["iv"]))
    above_min = [c for c in candidates if iv_map.get(c, 0) >= IV_MIN_THRESHOLD]
    print(f"\n=== Step 4: IV filter (>= {IV_MIN_THRESHOLD}): {len(above_min)} / {len(candidates)} kept ===")

    print("\n=== Step 5: correlation reduction (WOE space, |r| > 0.7) ===")
    woe_train = pd.DataFrame(
        bp.transform(train[candidates], metric="woe"), columns=candidates, index=train.index
    )[above_min]
    after_corr, corr_log = reduce_by_correlation(woe_train, iv_map)
    for entry in corr_log:
        print(f"  dropped {entry['dropped']} (IV={entry['dropped_iv']}) vs kept {entry['kept']} "
              f"(IV={entry['kept_iv']}), r={entry['correlation']}")
    print(f"  {len(after_corr)} / {len(above_min)} kept")

    print("\n=== Step 6: VIF reduction (threshold > 5.0) ===")
    final_features, vif_log = reduce_by_vif(woe_train[after_corr])
    for entry in vif_log:
        print(f"  dropped {entry['dropped']} (VIF={entry['vif']})")
    final_vifs = compute_vif(woe_train[final_features])
    print(f"  {len(final_features)} / {len(after_corr)} kept. Final VIFs:")
    print(final_vifs.round(2).to_string())

    features_after_vif = list(final_features)

    print(f"\n=== Step 7: fit logistic regression on {len(final_features)} WOE features ===")
    y_train = train["target"].values
    sign_drop_log = []
    model_features = list(final_features)
    while True:
        clf = LogisticRegression(penalty=None, max_iter=1000)
        clf.fit(woe_train[model_features], y_train)
        coefs = pd.Series(clf.coef_[0], index=model_features)
        # Under this WOE convention (higher WOE = lower event rate, verified
        # on mths_since_last_delinq above), every coefficient should be
        # negative: a more protective WOE should lower predicted P(default).
        positive = coefs[coefs > 0]
        if positive.empty:
            break
        # Drop the smallest-magnitude positive coefficient first: a sign
        # flip with a tiny coefficient is the classic symptom of
        # multicollinearity among the *retained* WOE features (VIF <= 5
        # does not guarantee zero residual correlation), not a real
        # business-sense violation worth keeping.
        worst = positive.abs().idxmin()
        sign_drop_log.append(
            {"dropped": worst, "coefficient": round(float(coefs[worst]), 4), "reason": "positive coefficient (sign mismatch)"}
        )
        print(f"  sign mismatch: dropping {worst} (coef={coefs[worst]:.4f}), refitting")
        model_features.remove(worst)

    final_features = model_features
    coefs = coefs.sort_values()
    coef_signs = set(np.sign(coefs.values))
    print(coefs.round(4).to_string())
    print(f"  intercept: {clf.intercept_[0]:.4f}")
    print(f"  all coefficients same sign: {len(coef_signs) == 1}")
    print(f"  final model has {len(final_features)} features after sign correction")

    selection_log = {
        "n_candidates": len(candidates),
        "n_after_iv_filter": len(above_min),
        "n_after_correlation_filter": len(after_corr),
        "n_after_vif_filter": len(features_after_vif),
        "n_final_after_sign_correction": len(final_features),
        "iv_threshold": IV_MIN_THRESHOLD,
        "correlation_threshold": CORR_THRESHOLD,
        "vif_threshold": VIF_THRESHOLD,
        "dropped_by_iv": sorted(set(candidates) - set(above_min)),
        "dropped_by_correlation": corr_log,
        "dropped_by_vif": vif_log,
        "dropped_by_sign_correction": sign_drop_log,
        "final_features": final_features,
    }
    with open(METRICS_DIR / "feature_selection_log.json", "w") as f:
        json.dump(selection_log, f, indent=2, default=str)

    print("\n=== Step 8: points scaling (600 points at odds 50:1, PDO=20) ===")
    factor = PDO / np.log(2)
    offset = BASE_POINTS - factor * np.log(BASE_ODDS)
    scorecard_points = {}
    for feat in final_features:
        ob = bp.get_binned_variable(feat)
        table = ob.binning_table.build()
        coef = coefs[feat]
        n = len(final_features)
        bin_points = []
        for idx, row in table.iterrows():
            if idx == "Totals":
                continue
            woe = row["WoE"] if isinstance(row["WoE"], (int, float)) else 0.0
            pts = -(coef * woe + clf.intercept_[0] / n) * factor + offset / n
            bin_label = row["Bin"] if isinstance(row["Bin"], str) else list(row["Bin"])
            bin_points.append({"bin": bin_label, "woe": woe, "points": round(pts, 1)})
        scorecard_points[feat] = bin_points
    with open(METRICS_DIR / "scorecard_points.json", "w") as f:
        json.dump(
            {"factor": factor, "offset": offset, "pdo": PDO, "base_points": BASE_POINTS,
             "base_odds": BASE_ODDS, "points_by_feature": scorecard_points},
            f, indent=2, default=str,
        )
    print(f"  factor={factor:.3f}, offset={offset:.3f}")

    print("\n=== Step 9: evaluate on train and validation (2014). Test (2015) is NOT touched. ===")
    woe_validation = pd.DataFrame(
        bp.transform(validation[candidates], metric="woe"), columns=candidates, index=validation.index
    )[final_features]
    y_validation = validation["target"].values
    val_pred = clf.predict_proba(woe_validation)[:, 1]
    train_pred = clf.predict_proba(woe_train[final_features])[:, 1]

    def split_metrics(y_true, y_pred, split_name, n_rows, issue_years):
        auc = roc_auc_score(y_true, y_pred)
        return {
            "split": split_name,
            "issue_years": issue_years,
            "n_rows": n_rows,
            "auc": round(auc, 4),
            "gini": round(2 * auc - 1, 4),
            "ks": round(ks_statistic(y_true, y_pred), 4),
            "mean_predicted_pd_pct": round(100 * y_pred.mean(), 2),
            "actual_default_rate_pct": round(100 * y_true.mean(), 2),
        }

    train_metrics = split_metrics(y_train, train_pred, "train", len(train), "2012-2013")
    val_metrics = split_metrics(y_validation, val_pred, "validation", len(validation), "2014")
    val_metrics["test_touched"] = False
    train_metrics["auc_minus_validation_auc"] = round(train_metrics["auc"] - val_metrics["auc"], 4)

    eval_metrics = {"train": train_metrics, "validation": val_metrics}
    with open(METRICS_DIR / "scorecard_validation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(json.dumps(eval_metrics, indent=2))

    print("\n=== Step 9b: calibration (10-bin, validation) ===")
    calib_df = pd.DataFrame({"pred": val_pred, "actual": y_validation})
    calib_df["decile"] = pd.qcut(calib_df["pred"], 10, labels=False, duplicates="drop")
    calib_table = calib_df.groupby("decile").agg(
        n=("actual", "size"),
        mean_predicted_pd=("pred", "mean"),
        actual_default_rate=("actual", "mean"),
    )
    calib_table["mean_predicted_pd_pct"] = (calib_table["mean_predicted_pd"] * 100).round(2)
    calib_table["actual_default_rate_pct"] = (calib_table["actual_default_rate"] * 100).round(2)
    calib_table = calib_table.drop(columns=["mean_predicted_pd", "actual_default_rate"])
    calib_table.to_csv(METRICS_DIR / "scorecard_calibration_validation.csv")
    print(calib_table.to_string())

    print("\n=== Step 9c: score distribution and default rate by score band (validation) ===")
    # Score = Offset - Factor * logit, where logit = intercept + sum(coef*WOE)
    # is exactly clf.decision_function(). This matches the per-bin points
    # formula in Step 8: summing a row's bin points across all features
    # reproduces this same total score.
    logit_validation = clf.decision_function(woe_validation)
    score_validation = offset - factor * logit_validation
    score_df = pd.DataFrame({"score": score_validation, "actual": y_validation})
    score_df["band"] = pd.qcut(score_df["score"], 10, labels=False, duplicates="drop")
    score_band_table = score_df.groupby("band").agg(
        n=("actual", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        default_rate=("actual", "mean"),
    )
    score_band_table["default_rate_pct"] = (score_band_table["default_rate"] * 100).round(2)
    score_band_table = score_band_table.drop(columns=["default_rate"]).round(1)
    score_band_table.to_csv(METRICS_DIR / "scorecard_score_bands_validation.csv")
    print(score_band_table.to_string())

    score_summary = {
        "score_min": round(float(score_validation.min()), 1),
        "score_max": round(float(score_validation.max()), 1),
        "score_median": round(float(np.median(score_validation)), 1),
        "base_points": BASE_POINTS,
        "base_odds": BASE_ODDS,
        "pdo": PDO,
        "factor": round(factor, 3),
        "offset": round(offset, 3),
    }
    with open(METRICS_DIR / "scorecard_score_summary.json", "w") as f:
        json.dump(score_summary, f, indent=2)
    print(json.dumps(score_summary, indent=2))

    coef_table = coefs.reset_index()
    coef_table.columns = ["feature", "coefficient"]
    coef_table["iv"] = coef_table["feature"].map(iv_map)
    coef_table.to_csv(METRICS_DIR / "logistic_regression_coefficients.csv", index=False)

    import joblib

    Path("data/processed/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(bp, "data/processed/models/binning_process.joblib")
    joblib.dump(clf, "data/processed/models/scorecard_logreg.joblib")
    with open("data/processed/models/final_features.json", "w") as f:
        json.dump(final_features, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
