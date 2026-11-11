"""Core scoring logic for the API: raw request -> engineered features ->
calibrated PD -> risk band -> illustrative reason codes.

Uses the exact same engineer_features() and score_lightgbm() functions the
training/evaluation pipeline uses (src/features/feature_engineering.py,
src/evaluation/score_models.py) - no copied logic.

Reason codes come from booster.predict(pred_contrib=True), LightGBM's own
built-in per-feature contribution output (exact TreeSHAP, computed in C++
inside the library already being used for prediction) - not the separate
shap package. Checked directly against shap.TreeExplainer on a 2,000-row
sample: max abs difference 0.0, identical to float64 precision. Using the
booster's own output instead of a second library removes shap AND its
matplotlib dependency from the serving image entirely.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api.reason_codes import describe
from src.evaluation.score_models import (
    CATEGORICAL_COLUMNS,
    load_frozen_config,
    score_lightgbm,
)
from src.features.feature_engineering import engineer_features

METRICS_DIR = Path("reports/metrics")
API_DIR = Path("src/api")

CATEGORICAL_INPUT_FIELDS = ["home_ownership", "purpose", "verification_status"]


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _warning(field: str, reason: str, value) -> dict:
    return {"field": field, "reason": reason, "value": str(value)}


class ScoringEngine:
    """Loaded once at API startup; holds the frozen model, calibrator,
    risk bands, and training-range reference, all read from files
    produced by the training and freeze scripts."""

    def __init__(self) -> None:
        self.frozen = load_frozen_config()
        self.cfg = self.frozen["lightgbm_chosen"]
        self.features: list[str] = self.cfg["features"]
        self.model_name = "lightgbm_chosen"

        self.booster = joblib.load(self.cfg["model_path"])
        # The model version is self-describing and verifiable without a
        # live MLflow server: the git commit the frozen config was cut
        # from, plus the SHA-256 of the exact booster file baked into this
        # image. Anyone can independently recompute the second half from
        # the artifact and confirm it's the one actually running.
        joblib_sha256 = _sha256_of(self.cfg["model_path"])
        self.model_version = (
            f"lightgbm_chosen@git:{self.frozen['git_hash_at_freeze'][:12]}"
            f"+sha256:{joblib_sha256[:12]}"
        )

        with open(API_DIR / "risk_bands.json") as f:
            self.risk_bands = json.load(f)["bands"]

        with open(API_DIR / "feature_ranges.json") as f:
            self.feature_ranges: dict = json.load(f)

        shap_importance = pd.read_csv(METRICS_DIR / "shap_global_importance.csv", index_col=0)
        self.top10_shap_features = set(shap_importance.sort_values("mean_abs_shap_rank").head(10).index)

    def build_raw_row(self, payload: dict) -> pd.DataFrame:
        """Turn one validated request payload into a single-row DataFrame
        with every column engineer_features() and the model need.
        issue_d is not an applicant-supplied field: for a live application
        there is no issue date yet, so "today" is used, matching how the
        engineered features (credit_history_months, the two date-gated
        never_flag__* columns) are meant to be read for a fresh application."""
        row = dict(payload)
        row["issue_d"] = pd.Timestamp(row["issue_d"]) if row.get("issue_d") else pd.Timestamp(date.today())
        row["earliest_cr_line"] = pd.Timestamp(row["earliest_cr_line"])
        df = pd.DataFrame([row])
        # A single-row frame with a None value gets object dtype for that
        # column (there's no other value to infer float64 from), which
        # LightGBM rejects outright. Force every non-categorical,
        # non-datetime column to numeric so a None becomes a proper NaN.
        skip = set(CATEGORICAL_INPUT_FIELDS) | {"issue_d", "earliest_cr_line"}
        for col in df.columns:
            if col not in skip:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def assemble_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Combined raw+engineered row, columns ordered as the model
        expects. Not yet category-cast: score_lightgbm() (used for the
        actual PD) does that cast itself, and the SHAP explainer needs its
        own cast applied separately in top_drivers()."""
        engineered = engineer_features(raw_df)
        combined = pd.concat([raw_df, engineered], axis=1)
        return combined[self.features].copy()

    def check_warnings(self, payload: dict) -> list[dict]:
        warnings = []
        for field, value in payload.items():
            if value is None or field not in self.feature_ranges:
                continue
            ref = self.feature_ranges[field]
            if ref["type"] == "categorical":
                if value not in ref["known_values"]:
                    warnings.append(_warning(field, "unseen_category_not_in_training_data", value))
                continue
            is_date = ref["type"] == "date"
            v, lo, hi = (pd.Timestamp(value), pd.Timestamp(ref["min"]), pd.Timestamp(ref["max"])) if is_date \
                else (value, ref["min"], ref["max"])
            if v < lo or v > hi:
                warnings.append(_warning(field, f"outside_training_range_({ref['min']}_to_{ref['max']})", value))
        return warnings

    def missing_fields_for(self, payload: dict) -> list[str]:
        """Optional fields the request left out. A missing field that is
        also one of the model's top-10 global SHAP features (self.top10_shap_features)
        gets an extra warning in check_warnings_for_missing, since the
        served score then runs on less information than the evaluated
        model saw on most of its training rows."""
        return sorted(field for field, value in payload.items() if value is None)

    def check_warnings_for_missing(self, missing_fields: list[str]) -> list[dict]:
        return [
            _warning(field, "missing_top10_shap_feature_lowers_fidelity", "missing")
            for field in missing_fields
            if field in self.top10_shap_features
        ]

    def risk_band_for(self, pd_value: float) -> str:
        for band in self.risk_bands:
            lo, hi = band["pd_min"], band["pd_max"]
            if band["pd_max_inclusive"]:
                if lo <= pd_value <= hi:
                    return band["band"]
            else:
                if lo <= pd_value < hi:
                    return band["band"]
        # PD outside every band's range (e.g. below the lowest edge or an
        # edge-case float above the top edge): clamp to the nearest band.
        return self.risk_bands[0]["band"] if pd_value < self.risk_bands[0]["pd_min"] else self.risk_bands[-1]["band"]

    def top_drivers(self, feat_row: pd.DataFrame, n: int = 3) -> list[dict]:
        feat_row = feat_row.copy()
        for col in CATEGORICAL_COLUMNS:
            feat_row[col] = feat_row[col].astype("category")
        # pred_contrib=True returns one column per feature (same order as
        # self.features) plus a final bias/base-value column - that last
        # column is the expected-value term SHAP calls base_value, not a
        # feature, so it's sliced off here.
        contrib = self.booster.predict(
            feat_row, pred_contrib=True, num_iteration=self.booster.best_iteration
        )[0, :-1]
        order = np.argsort(-contrib)  # most risk-increasing first
        drivers = []
        for idx in order[:n]:
            feat = self.features[idx]
            drivers.append({
                "feature": feat,
                "plain_english": describe(feat),
                "shap_value_log_odds": round(float(contrib[idx]), 6),
            })
        return drivers

    def score(self, payload: dict, include_reason_codes: bool = True) -> dict:
        raw_df = self.build_raw_row(payload)
        feat_row = self.assemble_features(raw_df)

        pd_value = float(score_lightgbm(feat_row, self.cfg)[0])
        # SHAP (top_drivers) is the most expensive part of a request by far
        # - the ?include_reason_codes=false switch exists so the latency
        # measurement can report the cost with and without it.
        drivers = self.top_drivers(feat_row) if include_reason_codes else []
        band = self.risk_band_for(pd_value)
        missing_fields = self.missing_fields_for(payload)
        warnings = self.check_warnings(payload) + self.check_warnings_for_missing(missing_fields)

        return {
            "calibrated_pd": pd_value,
            "risk_band": band,
            "top_3_risk_increasing_drivers_illustrative_only": drivers,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "missing_fields": missing_fields,
            "warnings": warnings,
        }
