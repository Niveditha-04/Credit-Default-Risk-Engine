FROM python:3.11-slim

# libgomp1: LightGBM's runtime OpenMP dependency on Linux (see CLAUDE.md
# notes carried forward - macOS needs libomp via brew instead, Linux/Docker
# needs libgomp1 via apt).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code the API actually imports.
COPY src/api/ src/api/
COPY src/evaluation/score_models.py src/evaluation/score_models.py
COPY src/evaluation/__init__.py src/evaluation/__init__.py
COPY src/features/feature_engineering.py src/features/feature_engineering.py
COPY src/features/__init__.py src/features/__init__.py
COPY src/models/math_utils.py src/models/math_utils.py
COPY src/models/__init__.py src/models/__init__.py
COPY src/__init__.py src/__init__.py

# Frozen champion artifacts, baked in at build time.
COPY reports/metrics/frozen_model_config.json reports/metrics/frozen_model_config.json
COPY data/processed/models/lightgbm/model_chosen.joblib data/processed/models/lightgbm/model_chosen.joblib
COPY data/processed/models/lightgbm/calibrator.joblib data/processed/models/lightgbm/calibrator.joblib

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
