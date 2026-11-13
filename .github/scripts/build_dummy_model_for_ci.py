"""CI only: builds a tiny, structurally-valid dummy LightGBM booster and
Platt calibrator at the exact paths the Dockerfile COPYs from
(data/processed/models/lightgbm/), so `docker build` can be verified in
CI without the real, gitignored, frozen champion artifacts. Same pattern
tests/test_api.py already uses to test the API against a dummy model,
on the real committed frozen feature list (reports/metrics/
frozen_model_config.json) so the Dockerfile's COPY step and the served
app's startup both exercise the real feature contract. Never run outside
CI - this never touches the real frozen artifacts or any committed file.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
N_SYNTHETIC_ROWS = 200
MODEL_DIR = Path("data/processed/models/lightgbm")


def main() -> None:
    with open("reports/metrics/frozen_model_config.json") as f:
        feats = json.load(f)["lightgbm_chosen"]["features"]

    rng = np.random.RandomState(0)
    data = {}
    for feat in feats:
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
    target = rng.randint(0, 2, N_SYNTHETIC_ROWS)

    dataset = lgb.Dataset(df[feats], label=target, categorical_feature=CATEGORICAL_COLUMNS)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 3, "min_child_samples": 5, "verbosity": -1, "deterministic": True, "seed": 0},
        dataset,
        num_boost_round=5,
    )
    raw_preds = booster.predict(df[feats], raw_score=True)
    calibrator = LogisticRegression()
    calibrator.fit(raw_preds.reshape(-1, 1), target)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, MODEL_DIR / "model_chosen.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")
    print(f"Wrote dummy {MODEL_DIR / 'model_chosen.joblib'} and {MODEL_DIR / 'calibrator.joblib'} for CI docker build")


if __name__ == "__main__":
    main()
