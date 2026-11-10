"""Runs the full test suite once, under coverage on the four core
modules (src.features, src.data.target, src.monitoring.psi, src.api),
and saves the test count and per-file coverage to
reports/metrics/test_coverage_summary.json - the source for the test
and coverage numbers quoted in README.md. Not collected by pytest
itself (does not match test_*.py)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

METRICS_DIR = Path("reports/metrics")
CORE_MODULES = ["src.features", "src.data.target", "src.monitoring.psi", "src.api"]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cov_json = Path(tmp) / "coverage.json"
        cov_args = [f"--cov={m}" for m in CORE_MODULES]
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", *cov_args, f"--cov-report=json:{cov_json}"],
            capture_output=True, text=True, check=False,
        )
        print(result.stdout[-2000:])
        with open(cov_json) as f:
            cov = json.load(f)

    n_passed = int(result.stdout.split(" passed")[0].rsplit(" ", 1)[-1].split("\n")[-1])

    summary = {
        "n_tests_passed": n_passed,
        "core_modules_checked": CORE_MODULES,
        "core_modules_combined_percent_covered": round(cov["totals"]["percent_covered"], 2),
        "per_file_percent_covered": {
            f: round(info["summary"]["percent_covered"], 2) for f, info in sorted(cov["files"].items())
        },
    }
    with open(METRICS_DIR / "test_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
