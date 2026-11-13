"""Every serving asset the API loads at startup (feature ranges, risk
bands, the SHAP ranking behind missing-field warnings, the required-field
list) stores the SHA-256 of the model it was built from. This test reads
the served model's hash from reports/metrics/frozen_model_config.json
and fails if any serving asset was built from a different model - the
one way a promotion (a new champion, a new calibrator) could silently
leave a stale asset behind.
"""

import json
from pathlib import Path

FROZEN_CONFIG = Path("reports/metrics/frozen_model_config.json")
FEATURE_RANGES = Path("src/api/feature_ranges.json")
RISK_BANDS = Path("src/api/risk_bands.json")
FIELD_REQUIREMENTS = Path("reports/metrics/api_field_requirements.json")
SHAP_SERVED_SUMMARY = Path("reports/metrics/shap_global_importance_served_summary.json")


def _served_model_hash() -> str:
    with open(FROZEN_CONFIG) as f:
        frozen = json.load(f)
    return frozen["lightgbm_chosen"]["model_sha256"]


def test_feature_ranges_built_from_served_model():
    with open(FEATURE_RANGES) as f:
        ranges = json.load(f)
    assert ranges["_metadata"]["source_model_sha256"] == _served_model_hash()


def test_risk_bands_built_from_served_model():
    with open(RISK_BANDS) as f:
        bands = json.load(f)
    assert bands["source_model_sha256"] == _served_model_hash()


def test_field_requirements_built_from_served_model():
    with open(FIELD_REQUIREMENTS) as f:
        reqs = json.load(f)
    assert reqs["source_model_sha256"] == _served_model_hash()


def test_shap_ranking_built_from_served_model():
    with open(SHAP_SERVED_SUMMARY) as f:
        summary = json.load(f)
    assert summary["source_model_sha256"] == _served_model_hash()
