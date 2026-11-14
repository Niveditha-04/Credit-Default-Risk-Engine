"""API tests against a TINY DUMMY MODEL trained on synthetic data, so
these run in CI without the real (gitignored) champion artifacts. The
real champion is checked separately by src/api/parity_check.py (a
verification script, not a pytest test, run locally where the real
artifacts exist) - these tests check the API's request/response
contract and validation behavior, not model quality.

src/api/main.py builds its ScoringEngine() singleton at import time, so
load_frozen_config must be patched BEFORE src.api.main is imported -
every test below goes through the `client` fixture, which does the
patch-then-import in the right order.
"""

import importlib
import json
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

with open("reports/metrics/frozen_model_config.json") as f:
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
        "git_hash_at_freeze": "0000testdummy0000",
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

    def fake_load_frozen_config(path: str = "reports/metrics/frozen_model_config.json"):
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
    49.96 in 2016, reports/metrics/eligibility_caps_by_year.csv), so a
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
