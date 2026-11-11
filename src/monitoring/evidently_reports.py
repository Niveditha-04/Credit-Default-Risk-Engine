"""Evidently HTML drift reports, train (reference) vs each of
2015/2016/2017/2018 (current). HISTORICAL REPLAY, not live monitoring -
see src/monitoring/run_drift.py for that distinction and for the custom,
train-deciles PSI this project treats as the source of truth. These
Evidently reports are a complementary, visual cross-check using
Evidently's own drift detection (PSI for numeric/categorical columns),
not a replacement for the numbers in reports/metrics/drift_*.

Uses the same train reference restriction as run_drift.py: train rows
with issue_d >= 2012-09-01, because 37 of the 69 features were still
mid-rollout before that (see reports/metrics/field_rollout.csv) and
using unrestricted train would show the same rollout artifact here too.
"""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.evaluation.score_models import load_frozen_config
from src.monitoring.run_drift import TRAIN_PSI_REFERENCE_CUTOFF

DRIFT_DIR = Path("reports/drift")
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]


def build_dataset(df: pd.DataFrame, features: list[str]) -> Dataset:
    numerical = [c for c in features if c not in CATEGORICAL_COLUMNS]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=CATEGORICAL_COLUMNS)
    return Dataset.from_pandas(df[features], data_definition=definition)


def main() -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_config()
    features = frozen["lightgbm_chosen"]["features"]

    train_full = pd.read_parquet("data/processed/train.parquet")
    train_ref = train_full[train_full["issue_d"] >= TRAIN_PSI_REFERENCE_CUTOFF].reset_index(drop=True)
    reference_dataset = build_dataset(train_ref, features)

    test = pd.read_parquet("data/processed/test.parquet")
    replay = pd.read_parquet("data/processed/replay.parquet")
    replay["year"] = pd.to_datetime(replay["issue_d"]).dt.year

    years = {
        2015: test,
        2016: replay[replay["year"] == 2016],
        2017: replay[replay["year"] == 2017],
        2018: replay[replay["year"] == 2018],
    }

    file_sizes = {}
    for year, df in years.items():
        current_dataset = build_dataset(df, features)
        report = Report([DataDriftPreset(method="psi", cat_method="psi")])
        snapshot = report.run(current_dataset, reference_dataset)
        out_path = DRIFT_DIR / f"drift_train_vs_{year}.html"
        snapshot.save_html(str(out_path))
        size_kb = round(out_path.stat().st_size / 1024, 1)
        file_sizes[str(year)] = {"path": str(out_path), "size_kb": size_kb}
        print(f"{year}: {len(df)} rows -> {out_path} ({size_kb} KB)")

    import json

    with open("reports/metrics/drift_evidently_file_sizes.json", "w") as f:
        json.dump(file_sizes, f, indent=2)
    print(json.dumps(file_sizes, indent=2))


if __name__ == "__main__":
    main()
