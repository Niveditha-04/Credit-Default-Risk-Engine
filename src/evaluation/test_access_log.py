"""Append-only log of every access to test (2015) data. Test is touched
exactly once, and every touch is logged here."""

import json
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path("reports/metrics/test_access_log.json")


def log_access(script: str, row_count: int, action: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            entries = json.load(f)
    entries.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "script": script,
            "row_count": row_count,
            "action": action,
        }
    )
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[test_access_log] {script}: {row_count} rows - {action}")
