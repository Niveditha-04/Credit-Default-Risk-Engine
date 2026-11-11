"""EDA for the modelling table. Default-rate-by-year uses the
already-computed train/validation/test numbers (reports/metrics/
population_and_target.json). Everything else (default rate by feature,
distributions, correlations) is computed on TRAIN ONLY, so nothing here
peeks at validation or test to make a modelling decision.
"""

import json

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.data.target import labelled_population_query

FIG_DIR = "reports/figures"
TRAIN_PARQUET = "data/processed/train.parquet"
TRAIN_SOURCE_QUERY = f"SELECT * FROM ({labelled_population_query()}) WHERE split = 'train' AND target IS NOT NULL"


def default_rate_by_year() -> None:
    with open("reports/metrics/population_and_target.json") as f:
        pop = json.load(f)
    splits = pop["splits"]
    years = {"train": "2012-2013", "validation": "2014", "test": "2015"}
    rates = [splits[s]["default_rate_pct"] for s in ("train", "validation", "test")]
    labels = [years[s] for s in ("train", "validation", "test")]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, rates, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by split (36-month, individual loans)")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.2, f"{r}%", ha="center")
    ax.set_ylim(0, max(rates) + 3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/default_rate_by_year.png", dpi=150)
    plt.close(fig)


def default_rate_by_feature(con: duckdb.DuckDBPyConnection, feature: str, fname: str) -> pd.DataFrame:
    # grade/sub_grade/int_rate are benchmark-only (not in train.parquet), so
    # query the labelled train population directly rather than the feature table.
    df = con.execute(
        f"""
        SELECT "{feature}" AS value, count(*) AS n, round(100.0 * avg(target), 2) AS default_rate_pct
        FROM ({TRAIN_SOURCE_QUERY})
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    df.to_csv(f"reports/metrics/{fname}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["value"].astype(str), df["default_rate_pct"], color="#4C72B0")
    ax.set_ylabel("Default rate (%)")
    ax.set_title(f"Train default rate by {feature}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/default_rate_by_{feature}.png", dpi=150)
    plt.close(fig)
    return df


def distributions(con: duckdb.DuckDBPyConnection) -> None:
    cols = ["loan_amnt", "annual_inc", "dti", "fico_mean", "revol_util", "credit_history_months"]
    df = con.execute(f"SELECT {', '.join(cols)} FROM read_parquet('{TRAIN_PARQUET}')").fetchdf()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=50, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("Train feature distributions")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/distributions.png", dpi=150)
    plt.close(fig)


def correlation_heatmap(con: duckdb.DuckDBPyConnection) -> None:
    cols = [
        "loan_amnt",
        "annual_inc",
        "dti",
        "fico_mean",
        "revol_bal",
        "revol_util",
        "open_acc",
        "total_acc",
        "credit_history_months",
        "loan_to_income",
        "revol_bal_to_income",
        "target",
    ]
    df = con.execute(f"SELECT {', '.join(cols)} FROM read_parquet('{TRAIN_PARQUET}')").fetchdf()
    corr = df.corr(numeric_only=True)
    corr.to_csv("reports/metrics/train_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    fig.colorbar(im)
    ax.set_title("Train correlation matrix (selected features)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    default_rate_by_year()
    default_rate_by_feature(con, "grade", "default_rate_by_grade")
    default_rate_by_feature(con, "purpose", "default_rate_by_purpose")
    default_rate_by_feature(con, "home_ownership", "default_rate_by_home_ownership")
    distributions(con)
    correlation_heatmap(con)
    print("EDA figures saved to reports/figures/")


if __name__ == "__main__":
    main()
