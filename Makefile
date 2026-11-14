.PHONY: setup data features train promote-v1 promote calibrate freeze evaluate serve test drift verify mlflow-ui reproduce

VENV = .venv/bin
RAW_CSV_GZ = data/raw/accepted_2007_to_2018Q4.csv.gz
RAW_PARQUET = data/interim/accepted_2007_to_2018Q4_raw.parquet
RAW_PROFILE = reports/metrics/raw_data_profile.json
# Passed through to promote_v2.py. Empty by default: `make promote` only
# refits v2 and writes it to data/processed/models/candidates/, leaving
# the served model, frozen_model_config.json, and MLflow untouched.
# `make promote PROMOTE_V2_FLAG=--promote` deploys it for real.
PROMOTE_V2_FLAG :=

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

# Scorecard only. LightGBM/XGBoost train as part of promote, which runs
# the tuning search and the frozen-artifact copy in one invocation, so
# there is no separate, discarded gbm.py candidate run here.
train: features
	$(VENV)/python -m src.models.scorecard

# v1's recipe: runs src.models.gbm with --promote (the LightGBM/XGBoost
# tuning search, seeded but not guaranteed bit-for-bit identical to a
# previous run - see reports/metrics/training_order_sensitivity.json),
# then copies the result over the frozen data/processed/models/
# artifacts and registers it in MLflow with the "champion" alias. Kept
# as its own target, separate from `promote` (v2, below), since v2 is
# refit from this recipe's own output and every v1-pinned analysis in
# this project needs it to exist.
promote-v1: train
	$(VENV)/python -m src.models.gbm --promote

# Reproduces v2 (refit on 2012-2014, v1's frozen hyperparameters and
# tree count, 8 monotone constraints, Platt calibrator fit on 2015).
# Writes to data/processed/models/candidates/v2_promotion/ only, unless
# PROMOTE_V2_FLAG=--promote is passed, which deploys the refit over the
# frozen data/processed/models/lightgbm/ path (backing up v1's
# artifacts first), updates frozen_model_config.json, and registers v2
# in MLflow as champion. Needs freeze to already have run (reads
# frozen_model_config.json's allow-list to validate v2's own features).
# Never reads or writes reports/metrics/test_scored.parquet, the locked
# one-time test evaluation file - only test.parquet, for calibration.
promote: freeze
	$(VENV)/python -m src.models.promote_v2 $(PROMOTE_V2_FLAG)

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

# Runs the full pipeline end to end, in order: v1's recipe first (train
# is not listed directly - promote-v1 already depends on it), ending
# with v2 reproduced as a candidate only (PROMOTE_V2_FLAG is empty here
# on purpose: deploying v2 over the served model is a deliberate,
# separate action, `make promote PROMOTE_V2_FLAG=--promote`, never
# implied by reproduce).
reproduce: data features promote-v1 calibrate freeze evaluate drift promote
