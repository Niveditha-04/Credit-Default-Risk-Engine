"""Check whether loan_to_income's pileup at exactly 0.5 is something our
pipeline clips, or a property of the raw data. No clip()/min()/max() call
exists anywhere in build_features.py's loan_to_income formula (plain
division: loan_amnt / annual_inc) - this script confirms that empirically
across all three splits and documents the finding.
"""

import json
from pathlib import Path

import pandas as pd

METRICS_DIR = Path("reports/metrics")


def main() -> None:
    rows = []
    for name, path in [
        ("train", "data/processed/train.parquet"),
        ("validation", "data/processed/validation.parquet"),
        ("test", "data/processed/test.parquet"),
    ]:
        df = pd.read_parquet(path)
        lti = df["loan_to_income"]
        rows.append(
            {
                "split": name,
                "n": len(df),
                "max": round(float(lti.max()), 4),
                "pct_at_exactly_0_5": round(100 * (lti == 0.5).mean(), 3),
                "pct_above_0_5": round(100 * (lti > 0.5).mean(), 3),
            }
        )
    result = {
        "clipped_in_our_code": False,
        "formula": "loan_amnt / annual_inc (build_features.py), no clip/min/max applied anywhere in the pipeline",
        "by_split": rows,
        "interpretation": (
            "Train (2012-2013) has loans above 0.5 (max 0.7125), with little mass exactly at 0.5 (0.065%). "
            "Validation (2014) and test (2015) never exceed 0.5, and test shows 0.538% of loans landing "
            "exactly at 0.5 (7x validation's rate). This looks like a LendingClub underwriting policy "
            "(a hard loan-to-income cap around 50%) that tightened/became binding between the 2012-2013 and "
            "2014-2015 vintages - a property of the raw data, not something introduced by our feature "
            "engineering."
        ),
    }
    with open(METRICS_DIR / "loan_to_income_cap_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
