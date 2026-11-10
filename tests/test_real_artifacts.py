"""Tests against the REAL frozen champion and REAL processed data -
gitignored, present on the developer's machine, never in CI. Skipped
automatically if the files aren't there (so this file doesn't crash CI
even if someone forgets to pass -m "not requires_data"), and marked
requires_data either way so the marker-based skip is explicit too."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

VALIDATION_PARQUET = Path("data/processed/validation.parquet")
FROZEN_CONFIG = Path("reports/metrics/frozen_model_config.json")

pytestmark = pytest.mark.requires_data

requires_real_artifacts = pytest.mark.skipif(
    not (VALIDATION_PARQUET.exists() and FROZEN_CONFIG.exists()),
    reason="real data/model artifacts not present (expected outside a developer checkout)",
)


@requires_real_artifacts
def test_frozen_champion_scores_a_real_sample_reproducibly():
    from src.evaluation.score_models import load_frozen_config, score_lightgbm

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]
    sample = pd.read_parquet(VALIDATION_PARQUET).head(50)

    first = score_lightgbm(sample, cfg)
    second = score_lightgbm(sample, cfg)
    assert (first == second).all(), "scoring the same rows twice must give bit-identical results"
    assert ((first >= 0) & (first <= 1)).all()


@requires_real_artifacts
def test_frozen_champion_feature_list_matches_saved_features_json():
    with open(FROZEN_CONFIG) as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]
    with open(Path(cfg["model_path"]).parent / "features.json") as f:
        saved_features = json.load(f)
    assert saved_features == cfg["features"]


@requires_real_artifacts
def test_frozen_champion_booster_loads_and_has_expected_feature_count():
    with open(FROZEN_CONFIG) as f:
        frozen = json.load(f)
    cfg = frozen["lightgbm_chosen"]
    booster = joblib.load(cfg["model_path"])
    assert booster.num_feature() == len(cfg["features"])
