"""Convert the raw LendingClub CSV.gz into parquet (all columns as text).

Typing happens later. This script only moves the data from CSV.gz to
parquet and proves the row/column counts are preserved, using two
independent counting methods.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

RAW_CSV_GZ = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
OUT_PARQUET = Path("data/interim/accepted_2007_to_2018Q4_raw.parquet")
OUT_PROFILE = Path("reports/metrics/raw_data_profile.json")


def total_ram_gb() -> float:
    if platform.system() == "Darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        return int(out) / (1024**3)
    # Linux fallback
    return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)


def sha256_of_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows_cols_pandas(path: Path) -> tuple[int, int]:
    """Independent row/column count using pandas, reading in chunks."""
    header = pd.read_csv(path, nrows=0)
    n_cols = len(header.columns)
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000, dtype=str):
        n_rows += len(chunk)
    return n_rows, n_cols


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV_GZ.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_CSV_GZ}")

    file_size_bytes = RAW_CSV_GZ.stat().st_size

    print("Hashing raw file (SHA-256)...")
    t0 = time.time()
    file_hash = sha256_of_file(RAW_CSV_GZ)
    print(f"  done in {time.time() - t0:.1f}s: {file_hash}")

    ram_gb = total_ram_gb()
    duckdb_memory_limit_gb = round(ram_gb * 0.5, 1)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{duckdb_memory_limit_gb}GB'")
    con.execute("SET threads = 4")

    print("Counting rows/columns with DuckDB (method 1)...")
    t0 = time.time()
    n_rows_duckdb = con.execute(
        f"""
        SELECT count(*)
        FROM read_csv('{RAW_CSV_GZ.as_posix()}', all_varchar=true, header=true)
        """
    ).fetchone()[0]
    n_cols_duckdb = con.execute(
        f"""
        SELECT count(*) FROM (
            DESCRIBE SELECT * FROM read_csv(
                '{RAW_CSV_GZ.as_posix()}', all_varchar=true, header=true
            )
        )
        """
    ).fetchone()[0]
    print(f"  done in {time.time() - t0:.1f}s: rows={n_rows_duckdb}, cols={n_cols_duckdb}")

    print("Counting rows/columns with pandas (method 2, independent)...")
    t0 = time.time()
    n_rows_pandas, n_cols_pandas = count_rows_cols_pandas(RAW_CSV_GZ)
    print(f"  done in {time.time() - t0:.1f}s: rows={n_rows_pandas}, cols={n_cols_pandas}")

    assert n_rows_duckdb == n_rows_pandas, (
        f"Row count mismatch: duckdb={n_rows_duckdb} pandas={n_rows_pandas}"
    )
    assert n_cols_duckdb == n_cols_pandas, (
        f"Column count mismatch: duckdb={n_cols_duckdb} pandas={n_cols_pandas}"
    )

    print(f"Writing parquet to {OUT_PARQUET} (all columns as text)...")
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv(
                '{RAW_CSV_GZ.as_posix()}', all_varchar=true, header=true
            )
        ) TO '{OUT_PARQUET.as_posix()}' (FORMAT PARQUET)
        """
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Verifying parquet row/column counts (two independent methods)...")
    n_rows_parquet_duckdb = con.execute(
        f"SELECT count(*) FROM read_parquet('{OUT_PARQUET.as_posix()}')"
    ).fetchone()[0]
    pq_file = pq.ParquetFile(OUT_PARQUET)
    n_rows_parquet_pyarrow = pq_file.metadata.num_rows
    n_cols_parquet_pyarrow = pq_file.metadata.num_columns

    assert n_rows_parquet_duckdb == n_rows_parquet_pyarrow == n_rows_duckdb, (
        "Parquet row count does not match source CSV row count: "
        f"duckdb_parquet={n_rows_parquet_duckdb} "
        f"pyarrow_parquet={n_rows_parquet_pyarrow} "
        f"source_csv={n_rows_duckdb}"
    )
    assert n_cols_parquet_pyarrow == n_cols_duckdb, (
        f"Parquet column count mismatch: parquet={n_cols_parquet_pyarrow} csv={n_cols_duckdb}"
    )

    parquet_size_bytes = OUT_PARQUET.stat().st_size

    profile = {
        "raw_file": str(RAW_CSV_GZ),
        "raw_file_size_bytes": file_size_bytes,
        "raw_file_sha256": file_hash,
        "raw_row_count": n_rows_duckdb,
        "raw_column_count": n_cols_duckdb,
        "row_count_method_1": "duckdb read_csv count(*)",
        "row_count_method_2": "pandas read_csv chunksize=200000",
        "row_count_methods_agree": True,
        "column_count_methods_agree": True,
        "parquet_file": str(OUT_PARQUET),
        "parquet_file_size_bytes": parquet_size_bytes,
        "parquet_row_count": n_rows_parquet_duckdb,
        "parquet_column_count": n_cols_parquet_pyarrow,
        "parquet_row_count_verification": "duckdb read_parquet count(*) vs pyarrow ParquetFile metadata",
        "machine": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "total_ram_gb": round(ram_gb, 1),
            "duckdb_memory_limit_gb": duckdb_memory_limit_gb,
            "cpu_count": os.cpu_count(),
        },
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }

    with open(OUT_PROFILE, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nSaved profile to {OUT_PROFILE}")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
