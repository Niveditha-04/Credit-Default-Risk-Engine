"""Phase 8 latency measurement: 1,000 sequential requests against the
RUNNING Docker container (not TestClient - a real HTTP round trip over
localhost), with and without reason codes. These are local laptop
numbers on one machine, not a production SLA measurement - no load
balancer, no concurrency, no network beyond localhost.
"""

import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

METRICS_DIR = Path("reports/metrics")
BASE_URL = "http://localhost:8000"
N_REQUESTS = 1000

PAYLOAD = {
    "loan_amnt": 15000.0, "annual_inc": 65000.0, "dti": 18.5,
    "fico_range_low": 680.0, "fico_range_high": 684.0,
    "earliest_cr_line": "2005-03-01",
    "home_ownership": "RENT", "purpose": "debt_consolidation",
    "verification_status": "Verified", "emp_length": 5.0,
    "revol_util": 45.0, "revol_bal": 8000.0, "open_acc": 10.0,
    "total_acc": 20.0, "pub_rec": 0.0, "pub_rec_bankruptcies": 0.0,
    "inq_last_6mths": 1.0, "mort_acc": 1.0, "acc_open_past_24mths": 3.0,
}


def run_sequential(client: httpx.Client, include_reason_codes: bool, n: int) -> np.ndarray:
    latencies_ms = np.empty(n)
    params = {"include_reason_codes": str(include_reason_codes).lower()}
    for i in range(n):
        t0 = time.perf_counter()
        resp = client.post(f"{BASE_URL}/score", json=PAYLOAD, params=params)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, resp.text
    return latencies_ms


def summarize(latencies_ms: np.ndarray) -> dict:
    return {
        "n": len(latencies_ms),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "min_ms": round(float(latencies_ms.min()), 3),
        "max_ms": round(float(latencies_ms.max()), 3),
    }


def main() -> None:
    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{BASE_URL}/health")
        assert health.status_code == 200, "container not reachable at " + BASE_URL
        print("Health check:", health.json())

        print(f"\nRunning {N_REQUESTS} sequential requests WITH reason codes...")
        with_reasons = run_sequential(client, True, N_REQUESTS)

        print(f"Running {N_REQUESTS} sequential requests WITHOUT reason codes...")
        without_reasons = run_sequential(client, False, N_REQUESTS)

    result = {
        "label": "local laptop numbers, not production - single container, no load balancer, sequential requests over localhost",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "n_requests_per_condition": N_REQUESTS,
        "with_reason_codes": summarize(with_reasons),
        "without_reason_codes": summarize(without_reasons),
    }
    with open(METRICS_DIR / "api_latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
