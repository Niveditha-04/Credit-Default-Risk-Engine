.PHONY: setup data features train promote calibrate freeze evaluate serve test drift verify mlflow-ui reproduce

VENV = .venv/bin
RAW_CSV_GZ = data/raw/accepted_2007_to_2018Q4.csv.gz
RAW_PARQUET = data/interim/accepted_2007_to_2018Q4_raw.parquet
RAW_PROFILE = reports/metrics/raw_data_profile.json

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -r requirements-dev.txt

# Raw file -> cleaned, typed table with target/split assigned. Skips the
# CSV.gz -> parquet conversion when the raw file's hash already matches
# the one recorded the last time it ran, and the parquet is still there.
data:
	@if [ -f $(RAW_PROFILE) ] && [ -f $(RAW_PARQUET) ] && [ -f $(RAW_CSV_GZ) ] && \
	   $(VENV)/python -c "import json, hashlib, sys; \
	     h = hashlib.sha256(); f = open('$(RAW_CSV_GZ)', 'rb'); \
	     [h.update(chunk) for chunk in iter(lambda: f.read(8 * 1024 * 1024), b'')]; \
	     profile = json.load(open('$(RAW_PROFILE)')); \
	     sys.exit(0 if h.hexdigest() == profile['raw_file_sha256'] else 1)" 2>/dev/null; then \
		echo "$(RAW_CSV_GZ) hash matches $(RAW_PROFILE) and $(RAW_PARQUET) already exists - skipping convert_to_parquet.py"; \
	else \
		$(VENV)/python -m src.data.convert_to_parquet; \
	fi
	$(VENV)/python -m src.data.clean
	$(VENV)/python -m src.data.target

# Allow-list, rollout diagnostics, final train/validation/test/replay tables.
features: data
	$(VENV)/python -m src.features.missing_by_year
	$(VENV)/python -m src.features.field_rollout
	$(VENV)/python -m src.features.allow_list
	$(VENV)/python -m src.features.build_features

# Scorecard + LightGBM/XGBoost tuning search, each logged to MLflow.
# Writes a candidate under data/processed/models/candidates/<run_id>/
# without touching the frozen champion - see promote.
train: features
	$(VENV)/python -m src.models.scorecard
	$(VENV)/python -m src.models.gbm

# Reruns src.models.gbm with --promote: trains again (seeded, but not
# guaranteed bit-for-bit identical to a previous run - see
# reports/metrics/training_order_sensitivity.json) and this time copies
# the result over the frozen data/processed/models/ artifacts and
# registers it in MLflow with the "champion" alias.
promote:
	$(VENV)/python -m src.models.gbm --promote

# Out-of-sample Platt calibration for the two comparison models
# (XGBoost, scorecard), fit the same way as the LightGBM champion's.
calibrate:
	$(VENV)/python -m src.models.calibrate_comparison_models

# Writes reports/metrics/frozen_model_config.json: the one file the test
# evaluation loads every model, feature list, and calibrator from.
freeze:
	$(VENV)/python -m src.models.freeze_model

# Sets the approval policy on validation/train, scores test once,
# then computes metrics and economics from that one saved scoring pass.
evaluate:
	$(VENV)/python -m src.evaluation.decision_policy
	$(VENV)/python -m src.evaluation.score_test
	$(VENV)/python -m src.evaluation.test_evaluation
	$(VENV)/python -m src.evaluation.economics

serve:
	$(VENV)/python -m uvicorn src.api.main:app --reload

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
	$(VENV)/python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Runs the full pipeline end to end, in order.
reproduce: data features train promote calibrate freeze evaluate drift
