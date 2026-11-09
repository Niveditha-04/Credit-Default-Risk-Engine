.PHONY: setup data features train evaluate serve test drift verify mlflow-ui

VENV = .venv/bin

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -r requirements.txt

# Phase 1-2: raw file -> cleaned, typed table with target/split assigned.
data:
	$(VENV)/python src/data/convert_to_parquet.py
	$(VENV)/python -m src.data.clean
	$(VENV)/python -m src.data.target

# Phase 2: allow-list, rollout diagnostics, final train/validation/test/replay tables.
features: data
	$(VENV)/python -m src.features.missing_by_year
	$(VENV)/python -m src.features.field_rollout
	$(VENV)/python -m src.features.allow_list
	$(VENV)/python -m src.features.build_features

# Phase 3-4: scorecard + LightGBM/XGBoost, both logged to MLflow by default.
train: features
	$(VENV)/python -m src.models.scorecard
	$(VENV)/python -m src.models.gbm

evaluate:
	$(VENV)/python -m src.evaluation.evaluate

serve:
	$(VENV)/uvicorn src.api.main:app --reload

test:
	$(VENV)/pytest -q

drift:
	$(VENV)/python -m src.monitoring.run_drift

# Load models:/credit_default_lightgbm_champion@champion and assert it
# matches the frozen model exactly on validation.
verify:
	$(VENV)/python -m src.models.verify_champion_alias

# Port 5001, not MLflow's default 5000 - macOS AirPlay Receiver listens on
# 5000 and silently intercepts it.
mlflow-ui:
	$(VENV)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
