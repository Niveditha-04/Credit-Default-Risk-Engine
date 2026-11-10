"""Target definition and population filters for the modelling table.

Target (lifetime default):
    1 = Charged Off, Default
    0 = Fully Paid
    excluded = Current, In Grace Period, Late (16-30 days), Late (31-120 days),
               both "Does not meet the credit policy" statuses

Population: term = 36 months, application_type = Individual, issue year in
[2012, 2015] for train/validation/test. The replay population (2016-2018)
uses the same term/application_type filter but keeps every loan_status,
since no label is used there.
"""

from pathlib import Path

import duckdb

CLEANED_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_cleaned.parquet")

DEFAULT_STATUSES = ("Charged Off", "Default")
PAID_STATUSES = ("Fully Paid",)
EXCLUDED_STATUSES = (
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
)

SPLIT_YEARS = {
    "train": (2012, 2013),
    "validation": (2014, 2014),
    "test": (2015, 2015),
    "replay": (2016, 2018),
}


def target_case_sql() -> str:
    default_list = ", ".join(f"'{s}'" for s in DEFAULT_STATUSES)
    paid_list = ", ".join(f"'{s}'" for s in PAID_STATUSES)
    return f"""
        CASE
            WHEN loan_status IN ({default_list}) THEN 1
            WHEN loan_status IN ({paid_list}) THEN 0
            ELSE NULL
        END
    """


def split_case_sql() -> str:
    parts = []
    for split, (y0, y1) in SPLIT_YEARS.items():
        parts.append(
            f"WHEN EXTRACT(year FROM issue_d) BETWEEN {y0} AND {y1} THEN '{split}'"
        )
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def labelled_population_query() -> str:
    """train/validation/test: 36-month, individual, 2012-2015, with target."""
    return f"""
        SELECT *,
            {target_case_sql()} AS target,
            {split_case_sql()} AS split
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        WHERE term = 36
          AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2012 AND 2015
        ORDER BY id
    """


def replay_population_query() -> str:
    """2016-2018: same filters, all loan_status kept. target is forced NULL
    unconditionally, by design, not just where unresolved: some 2016-2018
    loans have already resolved (Fully Paid/Charged Off) and would
    otherwise get a real 0/1 target, but Section 2 of the brief defines
    replay as "no labels used" for drift monitoring. Computing a real
    target here would let a later phase accidentally compute performance
    metrics on replay data, which Phase 9 explicitly forbids for anything
    past 2015."""
    return f"""
        SELECT *,
            CAST(NULL AS INTEGER) AS target,
            'replay' AS split
        FROM read_parquet('{CLEANED_PARQUET.as_posix()}')
        WHERE term = 36
          AND application_type = 'Individual'
          AND EXTRACT(year FROM issue_d) BETWEEN 2016 AND 2018
        ORDER BY id
    """


def population_and_target_report(con: duckdb.DuckDBPyConnection) -> dict:
    """Counts and default rate per split, plus exclusion counts."""
    df = con.execute(
        f"""
        SELECT split, target, count(*) AS n
        FROM ({labelled_population_query()})
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()

    report: dict = {"splits": {}}
    for split in ("train", "validation", "test"):
        sub = df[df["split"] == split]
        n_default = int(sub.loc[sub["target"] == 1, "n"].sum())
        n_paid = int(sub.loc[sub["target"] == 0, "n"].sum())
        n_excluded = int(sub.loc[sub["target"].isna(), "n"].sum())
        n_total_in_year_range = n_default + n_paid + n_excluded
        report["splits"][split] = {
            "issue_years": SPLIT_YEARS[split],
            "n_fully_paid": n_paid,
            "n_default": n_default,
            "n_excluded_unresolved": n_excluded,
            "n_total_in_population_before_target_filter": n_total_in_year_range,
            "n_labelled": n_default + n_paid,
            "default_rate_pct": round(100.0 * n_default / (n_default + n_paid), 2)
            if (n_default + n_paid) > 0
            else None,
        }

    # exclusion breakdown by loan_status, across train+validation+test population
    excl_df = con.execute(
        f"""
        SELECT split, loan_status, count(*) AS n
        FROM ({labelled_population_query()})
        WHERE target IS NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()
    report["exclusions_by_status"] = (
        excl_df.groupby("loan_status")["n"].sum().sort_values(ascending=False).to_dict()
    )

    # replay population size (no target breakdown needed, just row count)
    replay_n = con.execute(f"SELECT count(*) FROM ({replay_population_query()})").fetchone()[0]
    report["replay"] = {"issue_years": SPLIT_YEARS["replay"], "n_rows": replay_n}

    return report


if __name__ == "__main__":
    import json

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    report = population_and_target_report(con)
    print(json.dumps(report, indent=2, default=str))
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/population_and_target.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
