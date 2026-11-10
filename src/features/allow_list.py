"""Loads the explicit allow-list for all 151 raw columns (decision,
reason, whether known at loan origination) from config/allow_list.csv.

This is a positive list (ALLOW), not a block-list: a column only becomes
a model feature if it is explicitly marked decision="keep_feature" in
that table. Everything else is excluded by construction, including any
column we simply forgot to think about (the completeness assertion below
would fail first).

Decisions:
    keep_feature       -> goes into the model feature table
    keep_not_feature   -> kept in the dataset for filtering/splitting/joins/
                          stability analysis, never fed to a model as-is
    drop                -> excluded entirely
    target_source       -> this is where the target comes from, not a feature
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from src.data.columns import ALL_GROUPED_COLUMNS

ALLOW_LIST_CSV = Path("config/allow_list.csv")


@dataclass(frozen=True)
class ColumnDecision:
    decision: str  # keep_feature | keep_not_feature | drop | target_source
    reason: str
    known_at_origination: bool | None  # None = not applicable (not a feature)


def _parse_known_at_origination(value: str) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unexpected known_at_origination value: {value!r}")


def _load_allow_list(path: Path = ALLOW_LIST_CSV) -> dict[str, ColumnDecision]:
    allow_list: dict[str, ColumnDecision] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            col = row["column"]
            if col in allow_list:
                raise ValueError(f"Column {col} already has a decision ({allow_list[col]})")
            allow_list[col] = ColumnDecision(
                decision=row["decision"],
                reason=row["reason"],
                known_at_origination=_parse_known_at_origination(row["known_at_origination"]),
            )
    return allow_list


ALLOW_LIST: dict[str, ColumnDecision] = _load_allow_list()


def check_completeness() -> None:
    allow_set = set(ALLOW_LIST.keys())
    raw_set = set(ALL_GROUPED_COLUMNS)
    missing = raw_set - allow_set
    extra = allow_set - raw_set
    assert not missing, f"Columns with no allow-list decision: {sorted(missing)}"
    assert not extra, f"Allow-list references columns not in the raw file: {sorted(extra)}"


def banned_columns() -> set[str]:
    """Anything not explicitly marked keep_feature must never reach the model."""
    return {c for c, d in ALLOW_LIST.items() if d.decision != "keep_feature"}


def feature_columns() -> list[str]:
    return sorted(c for c, d in ALLOW_LIST.items() if d.decision == "keep_feature")


def write_feature_audit_csv(path: str = "reports/metrics/feature_audit.csv") -> None:
    check_completeness()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "decision", "known_at_origination", "reason"])
        for col in sorted(ALLOW_LIST):
            d = ALLOW_LIST[col]
            writer.writerow([col, d.decision, d.known_at_origination, d.reason])
    print(f"Wrote {path} ({len(ALLOW_LIST)} rows)")


if __name__ == "__main__":
    check_completeness()
    print(f"All {len(ALLOW_LIST)} columns have a decision.")
    feats = feature_columns()
    print(f"{len(feats)} columns marked keep_feature:")
    for c in feats:
        print(f"  {c}")
    write_feature_audit_csv()
