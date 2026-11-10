"""Direct unit tests for ScoringEngine methods, against a tiny dummy
model (same pattern as test_api.py) - covers branches the end-to-end API
tests don't exercise directly (risk band edge clamping, numeric range
warnings, cross-field validation)."""

import json

import pytest

from tests.test_api import build_dummy_model

with open("reports/metrics/frozen_model_config.json") as f:
    REAL_FEATURES = json.load(f)["lightgbm_chosen"]["features"]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    dummy_frozen = build_dummy_model(tmp_path)
    import src.api.scoring as scoring_mod
    import src.evaluation.score_models as score_models_mod

    def fake_load_frozen_config(path: str = "reports/metrics/frozen_model_config.json"):
        return dummy_frozen

    monkeypatch.setattr(score_models_mod, "load_frozen_config", fake_load_frozen_config)
    monkeypatch.setattr(scoring_mod, "load_frozen_config", fake_load_frozen_config)
    return scoring_mod.ScoringEngine()


def test_risk_band_for_clamps_below_lowest_edge(engine):
    lowest = engine.risk_bands[0]
    band = engine.risk_band_for(lowest["pd_min"] - 0.5)
    assert band == lowest["band"]


def test_risk_band_for_clamps_above_highest_edge(engine):
    highest = engine.risk_bands[-1]
    band = engine.risk_band_for(highest["pd_max"] + 0.5)
    assert band == highest["band"]


def test_risk_band_for_middle_band_uses_exclusive_upper_bound(engine):
    mid_band = engine.risk_bands[len(engine.risk_bands) // 2]
    band = engine.risk_band_for(mid_band["pd_min"])
    assert band == mid_band["band"]


def test_risk_band_for_exact_top_edge_is_inclusive(engine):
    highest = engine.risk_bands[-1]
    band = engine.risk_band_for(highest["pd_max"])
    assert band == highest["band"]


def test_numeric_value_above_training_range_warns(engine):
    field, ref = next((f, r) for f, r in engine.feature_ranges.items() if r["type"] == "numeric")
    payload = {field: ref["max"] + 1_000_000.0}
    warnings = engine.check_warnings(payload)
    assert any(w["field"] == field for w in warnings)


def test_numeric_value_inside_training_range_does_not_warn(engine):
    field, ref = next((f, r) for f, r in engine.feature_ranges.items() if r["type"] == "numeric")
    midpoint = (ref["min"] + ref["max"]) / 2
    warnings = engine.check_warnings({field: midpoint})
    assert not any(w["field"] == field for w in warnings)


def test_fico_range_high_below_low_is_rejected():
    from src.api.schemas import ScoreRequest

    payload = {
        "loan_amnt": 10000.0, "annual_inc": 50000.0, "dti": 20.0,
        "fico_range_low": 750.0, "fico_range_high": 700.0,
        "earliest_cr_line": "2010-01-01",
        "home_ownership": "RENT", "purpose": "other",
        "verification_status": "Verified",
    }
    with pytest.raises(ValueError, match="fico_range_high must be >= fico_range_low"):
        ScoreRequest(**payload)
