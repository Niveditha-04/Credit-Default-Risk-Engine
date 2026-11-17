"""API tests against a TINY DUMMY MODEL trained on synthetic data, so
most of these run in CI without the real (gitignored) champion
artifacts. The real champion is checked by the requires_data parity
test at the bottom of this file (same tolerance, same sampling the
real champion is otherwise checked with, just run locally where the
real artifacts exist) - the dummy-model tests check the API's
request/response contract, validation behavior, and ScoringEngine
internals, not model quality.

src/api/main.py builds its ScoringEngine() singleton at import time, so
load_frozen_config must be patched BEFORE src.api.main is imported -
every dummy-model test below goes through the `client` fixture, which
does the patch-then-import in the right order.
"""

import importlib
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

with open("config/model_config.json") as f:
    REAL_FEATURES = json.load(f)["lightgbm_chosen"]["features"]

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
N_SYNTHETIC_ROWS = 200


def synthetic_training_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {}
    for feat in REAL_FEATURES:
        if feat == "home_ownership":
            data[feat] = rng.choice(["RENT", "MORTGAGE", "OWN"], N_SYNTHETIC_ROWS)
        elif feat == "purpose":
            data[feat] = rng.choice(["debt_consolidation", "credit_card", "other"], N_SYNTHETIC_ROWS)
        elif feat == "verification_status":
            data[feat] = rng.choice(["Verified", "Not Verified"], N_SYNTHETIC_ROWS)
        else:
            data[feat] = rng.uniform(0, 100, N_SYNTHETIC_ROWS)
    df = pd.DataFrame(data)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    df["target"] = rng.randint(0, 2, N_SYNTHETIC_ROWS)
    return df


def build_dummy_model(tmp_path):
    """Trains a tiny LightGBM booster + Platt calibrator on synthetic
    data, saves both, and returns a frozen-config-shaped dict pointing
    at them."""
    df = synthetic_training_frame()
    dataset = lgb.Dataset(df[REAL_FEATURES], label=df["target"], categorical_feature=CATEGORICAL_COLUMNS)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 3, "min_child_samples": 5, "verbosity": -1, "deterministic": True, "seed": 0},
        dataset,
        num_boost_round=5,
    )
    raw_preds = booster.predict(df[REAL_FEATURES], raw_score=True)
    calibrator = LogisticRegression()
    calibrator.fit(raw_preds.reshape(-1, 1), df["target"])

    model_path = tmp_path / "dummy_model.joblib"
    calibrator_path = tmp_path / "dummy_calibrator.joblib"
    joblib.dump(booster, model_path)
    joblib.dump(calibrator, calibrator_path)

    return {
        "lightgbm_chosen": {
            "features": REAL_FEATURES,
            "model_path": str(model_path),
            "calibrator": {"path": str(calibrator_path)},
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    dummy_frozen = build_dummy_model(tmp_path)

    # Patch load_frozen_config everywhere it's already been imported by
    # name (score_models and scoring both do `from ... import
    # load_frozen_config`, which copies a reference - patching only the
    # origin module would miss these copies).
    import src.evaluation.score_models as score_models_mod

    def fake_load_frozen_config(path: str = "config/model_config.json"):
        return dummy_frozen

    monkeypatch.setattr(score_models_mod, "load_frozen_config", fake_load_frozen_config)

    for modname in ["src.api.scoring", "src.api.main"]:
        if modname in sys.modules:
            del sys.modules[modname]

    import src.api.scoring as scoring_mod
    monkeypatch.setattr(scoring_mod, "load_frozen_config", fake_load_frozen_config)

    main_mod = importlib.import_module("src.api.main")
    return TestClient(main_mod.app)


VALID_PAYLOAD = {
    "loan_amnt": 15000.0, "annual_inc": 65000.0, "dti": 18.5,
    "fico_range_low": 680.0, "fico_range_high": 684.0,
    "earliest_cr_line": "2005-03-01",
    "home_ownership": "RENT", "purpose": "debt_consolidation",
    "verification_status": "Verified", "emp_length": 5.0,
    # Required fields (< 1% missing in train, src/api/field_requirements.py).
    "acc_now_delinq": 0.0, "chargeoff_within_12_mths": 0.0,
    "collections_12_mths_ex_med": 0.0, "delinq_2yrs": 0.0, "delinq_amnt": 0.0,
    "inq_last_6mths": 1.0, "open_acc": 10.0, "pub_rec": 0.0,
    "pub_rec_bankruptcies": 0.0, "revol_bal": 8000.0, "revol_util": 45.0,
    "tax_liens": 0.0, "total_acc": 20.0,
}
REQUIRED_FIELDS = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "earliest_cr_line", "home_ownership", "purpose", "verification_status",
    "acc_now_delinq", "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "delinq_2yrs", "delinq_amnt", "inq_last_6mths", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "tax_liens", "total_acc",
]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_valid_score_returns_pd_in_range(client):
    r = client.post("/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["calibrated_pd"] <= 1.0
    assert body["risk_band"] in {"Very Low", "Low", "Medium", "High", "Very High"}
    assert len(body["top_3_risk_increasing_drivers_illustrative_only"]) == 3
    assert "request_id" in body


def test_negative_income_rejected_with_422(client):
    bad = dict(VALID_PAYLOAD, annual_inc=-500.0)
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_fico_out_of_range_rejected_with_422(client):
    bad = dict(VALID_PAYLOAD, fico_range_low=200.0)
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_negative_dti_rejected_with_422(client):
    bad = dict(VALID_PAYLOAD, dti=-1.0)
    r = client.post("/score", json=bad)
    assert r.status_code == 422


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_rejected_with_422(client, field):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
    r = client.post("/score", json=bad)
    assert r.status_code == 422
    assert any(e["field"] == field for e in r.json()["errors"])


def test_unseen_category_warns_but_does_not_crash(client):
    unseen = dict(VALID_PAYLOAD, purpose="totally_unseen_purpose_xyz")
    r = client.post("/score", json=unseen)
    assert r.status_code == 200
    warnings = r.json()["warnings"]
    assert any(w["field"] == "purpose" and "unseen" in w["reason"] for w in warnings)


def test_dti_above_training_cap_warns_but_does_not_crash(client):
    """DTI has no upper bound in the schema (real values go as high as
    49.96 in 2016, reports/metrics/data/eligibility_caps_by_year.csv), so a
    high DTI is accepted (200) but flagged as outside the training range
    (src/api/feature_ranges.json's max, currently 39.99)."""
    high_dti = dict(VALID_PAYLOAD, dti=45.0)
    r = client.post("/score", json=high_dti)
    assert r.status_code == 200
    warnings = r.json()["warnings"]
    assert any(w["field"] == "dti" and "outside_training_range" in w["reason"] for w in warnings)


def test_missing_optional_fields_do_not_crash(client):
    minimal = dict(VALID_PAYLOAD)
    r = client.post("/score", json=minimal)
    assert r.status_code == 200


def test_missing_fields_listed_in_response(client):
    minimal = dict(VALID_PAYLOAD)
    r = client.post("/score", json=minimal)
    body = r.json()
    assert "acc_open_past_24mths" in body["missing_fields"]
    assert "loan_amnt" not in body["missing_fields"]


def test_missing_top10_shap_feature_warns(client):
    minimal = dict(VALID_PAYLOAD)
    r = client.post("/score", json=minimal)
    warnings = r.json()["warnings"]
    assert any(
        w["field"] == "mths_since_recent_inq" and "missing_top10_shap" in w["reason"] for w in warnings
    )


# ---------------------------------------------------------------------
# Direct unit tests for ScoringEngine methods (same dummy-model pattern
# as above) - covers branches the end-to-end API tests above don't
# exercise directly (risk band edge clamping, numeric range warnings,
# cross-field validation).
# ---------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path, monkeypatch):
    dummy_frozen = build_dummy_model(tmp_path)
    import src.api.scoring as scoring_mod
    import src.evaluation.score_models as score_models_mod

    def fake_load_frozen_config(path: str = "config/model_config.json"):
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


# ---------------------------------------------------------------------
# Every serving asset the API loads at startup (feature ranges, risk
# bands, the SHAP ranking behind missing-field warnings, the
# required-field list) stores the SHA-256 of the model it was built
# from. These read the served model's hash from
# config/model_config.json and fail if any serving
# asset was built from a different model - the one way a promotion (a
# new champion, a new calibrator) could silently leave a stale asset
# behind. All of these files are small and committed, so this runs in
# CI without real training data.
# ---------------------------------------------------------------------

FEATURE_RANGES = Path("src/api/feature_ranges.json")
RISK_BANDS = Path("src/api/risk_bands.json")
FIELD_REQUIREMENTS = Path("reports/metrics/models/api_field_requirements.json")
SHAP_SERVED_SUMMARY = Path("reports/metrics/explain/shap_global_importance_served_summary.json")


def _served_model_hash() -> str:
    with open("config/model_config.json") as f:
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


# ---------------------------------------------------------------------
# Parity check against the REAL champion: 1,000 seeded test-set rows,
# sent ONE AT A TIME through the live FastAPI app (TestClient - no
# network, but the full request/response/validation path), must return
# the exact same PD as the frozen champion's batch scoring
# (src.evaluation.score_models), within 1e-9. Includes rows with
# missing optional fields and every purpose category in the sample by
# construction, not by luck. Needs the real (gitignored) test data and
# model artifacts, so it is marked requires_data and skipped in CI.
# ---------------------------------------------------------------------

PARITY_TEST_PARQUET = "data/processed/test.parquet"
PARITY_CLEANED_PARQUET = "data/interim/accepted_2007_to_2018Q4_cleaned.parquet"
PARITY_SEED = 42
PARITY_N_SAMPLE = 1000
PARITY_TOLERANCE = 1e-9

requires_real_data = pytest.mark.skipif(
    not Path(PARITY_TEST_PARQUET).exists(),
    reason="real test.parquet not present (expected outside a developer checkout)",
)


def _build_parity_sample() -> pd.DataFrame:
    import duckdb

    test = pd.read_parquet(PARITY_TEST_PARQUET)
    con = duckdb.connect()
    cleaned = con.execute(
        f"SELECT id, fico_range_low, fico_range_high, earliest_cr_line "
        f"FROM read_parquet('{PARITY_CLEANED_PARQUET}') WHERE id IN "
        f"(SELECT id FROM read_parquet('{PARITY_TEST_PARQUET}'))"
    ).fetchdf()
    merged = test.merge(cleaned, on="id", how="inner")
    assert len(merged) == len(test), f"join dropped rows: {len(test)} -> {len(merged)}"

    # One row per purpose category first, so every category is guaranteed
    # to appear even though the categories are not evenly sized. (A manual
    # loop, not groupby().apply() - pandas can silently drop the grouping
    # column from the result when the applied function returns a frame
    # shaped just like its input, which it does here.)
    one_per_purpose = pd.concat(
        [merged[merged["purpose"] == p].sample(n=1, random_state=PARITY_SEED) for p in sorted(merged["purpose"].unique())]
    )

    # A block of rows that have at least one missing optional field.
    optional_cols = [c for c in merged.columns if c not in (
        "id", "issue_d", "split", "target", "loan_amnt", "annual_inc", "dti",
        "fico_range_low", "fico_range_high", "earliest_cr_line",
        "home_ownership", "purpose", "verification_status",
    )]
    has_missing = merged[optional_cols].isna().any(axis=1)
    missing_rows = merged[has_missing].sample(
        n=min(100, int(has_missing.sum())), random_state=PARITY_SEED
    )

    forced = pd.concat([one_per_purpose, missing_rows]).drop_duplicates(subset="id")
    remaining_needed = PARITY_N_SAMPLE - len(forced)
    pool = merged[~merged["id"].isin(forced["id"])]
    filler = pool.sample(n=remaining_needed, random_state=PARITY_SEED)

    sample = pd.concat([forced, filler]).drop_duplicates(subset="id").reset_index(drop=True)
    assert len(sample) == PARITY_N_SAMPLE, f"expected {PARITY_N_SAMPLE} rows, got {len(sample)}"
    return sample


def _parity_row_to_payload(row: pd.Series, schema_fields: list[str]) -> dict:
    payload = {}
    for f in schema_fields:
        v = row[f]
        if v is None or pd.isna(v):
            payload[f] = None
        elif f in ("earliest_cr_line", "issue_d"):
            payload[f] = str(pd.Timestamp(v).date())
        elif f in ("home_ownership", "purpose", "verification_status"):
            payload[f] = str(v)
        else:
            payload[f] = float(v)
    return payload


@pytest.mark.requires_data
@requires_real_data
def test_api_parity_with_real_champion_batch_scoring():
    from src.api.schemas import ScoreRequest
    from src.evaluation.score_models import load_frozen_config, score_lightgbm

    # A dummy-model test earlier in this process may have left
    # src.api.main's ScoringEngine() singleton built against the dummy
    # config in sys.modules - force a fresh import against the real,
    # unpatched load_frozen_config so this test scores with the real
    # champion, not whatever the last dummy-model test built.
    for modname in ["src.api.scoring", "src.api.main"]:
        if modname in sys.modules:
            del sys.modules[modname]
    real_app = importlib.import_module("src.api.main").app

    frozen = load_frozen_config()
    cfg = frozen["lightgbm_chosen"]

    sample = _build_parity_sample()
    batch_pd = score_lightgbm(sample, cfg)

    client = TestClient(real_app)
    schema_fields = list(ScoreRequest.model_fields.keys())

    diffs = []
    for i in range(len(sample)):
        row = sample.iloc[i]
        payload = _parity_row_to_payload(row, schema_fields)

        resp = client.post("/score", json=payload)
        assert resp.status_code == 200, f"row {i} (id={row['id']}) failed: {resp.status_code} {resp.text}"
        api_pd = resp.json()["calibrated_pd"]
        diffs.append(abs(api_pd - batch_pd[i]))

    diffs = np.array(diffs)
    assert (diffs <= PARITY_TOLERANCE).all(), (
        f"{int((diffs > PARITY_TOLERANCE).sum())} of {len(sample)} rows exceeded "
        f"tolerance {PARITY_TOLERANCE} (max diff {diffs.max()})"
    )
