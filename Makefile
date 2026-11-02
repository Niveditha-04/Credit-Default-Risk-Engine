.PHONY: setup data train evaluate serve test drift

VENV = .venv/bin

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -r requirements.txt

data:
	$(VENV)/python src/data/convert_to_parquet.py

train:
	$(VENV)/python -m src.models.train

evaluate:
	$(VENV)/python -m src.evaluation.evaluate

serve:
	$(VENV)/uvicorn src.api.main:app --reload

test:
	$(VENV)/pytest -q

drift:
	$(VENV)/python -m src.monitoring.run_drift
