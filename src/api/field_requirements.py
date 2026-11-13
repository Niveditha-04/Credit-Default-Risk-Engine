"""Train+validation (2012-2014) missing rate for every raw field the API
accepts as input (src.api.schemas.ScoreRequest) - the evidence behind
which fields are required (< 1% missing) versus optional (>= 1%
missing, several legitimately missing-not-at-random: an event that
never happened). 2012-2014 because that is the served model's training
population (v2 is refit on train+validation). Engineered features
(fico_mean, loan_to_income, credit_history_months, revol_bal_to_income,
never_flag__*) are never accepted as API input, so they are not in
this table.
"""

import json
from pathlib import Path

import pandas as pd

from src.api.schemas import ScoreRequest
from src.evaluation.score_models import load_frozen_config

TRAIN_PARQUET = "data/processed/train.parquet"
VALIDATION_PARQUET = "data/processed/validation.parquet"
OUT_CSV = Path("reports/metrics/api_field_missing_rates.csv")
OUT_JSON = Path("reports/metrics/api_field_requirements.json")
REQUIRED_THRESHOLD_PCT = 1.0

# Fields in the schema that are not raw model features scored for
# missingness here: issue_d is not a model input at all (computed
# server-side or defaulted to today), and fico_range_low/fico_range_high/
# earliest_cr_line feed the engineered fico_mean/credit_history_months
# rather than being scored on their own missing rate.
NON_FEATURE_FIELDS = {"issue_d", "fico_range_low", "fico_range_high", "earliest_cr_line"}


def main() -> None:
    frozen = load_frozen_config()
    train = pd.concat(
        [pd.read_parquet(TRAIN_PARQUET), pd.read_parquet(VALIDATION_PARQUET)], ignore_index=True
    )
    fields = [f for f in ScoreRequest.model_fields if f not in NON_FEATURE_FIELDS]

    rows = []
    for field in fields:
        pct_missing = round(100 * float(train[field].isna().mean()), 3) if field in train.columns else None
        is_required = ScoreRequest.model_fields[field].is_required()
        rows.append({
            "field": field,
            "pct_missing_in_train": pct_missing,
            "required_in_api": is_required,
            "meets_required_threshold": pct_missing is not None and pct_missing < REQUIRED_THRESHOLD_PCT,
        })

    df = pd.DataFrame(rows).sort_values("pct_missing_in_train")
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))

    mismatches = df[df["meets_required_threshold"] != df["required_in_api"]]
    summary = {
        "computed_on": "train + validation (2012-2014), the served model's training population",
        "source_model_sha256": frozen["lightgbm_chosen"]["model_sha256"],
        "required_threshold_pct": REQUIRED_THRESHOLD_PCT,
        "required_fields": sorted(df[df["required_in_api"]]["field"]),
        "optional_fields": sorted(df[~df["required_in_api"]]["field"]),
        "n_required": int(df["required_in_api"].sum()),
        "n_optional": int((~df["required_in_api"]).sum()),
        "schema_matches_threshold_rule": bool(mismatches.empty),
        "mismatches": mismatches.to_dict("records"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    assert mismatches.empty, (
        f"schemas.py required/optional split does not match the < {REQUIRED_THRESHOLD_PCT}% train-missing "
        f"rule for: {mismatches['field'].tolist()}"
    )


if __name__ == "__main__":
    main()
