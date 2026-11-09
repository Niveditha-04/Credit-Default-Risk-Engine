"""FastAPI service for the credit default risk engine. Serves the frozen
champion (lightgbm_chosen) only - no training, no retraining, nothing
that touches data/ at request time beyond reading the baked artifacts.
"""

import uuid

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, ScoreRequest, ScoreResponse
from src.api.scoring import ScoringEngine

app = FastAPI(title="Credit Default Risk Engine", version="1.0.0")

engine = ScoringEngine()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Pydantic's default 422 body is a wall of internal error objects.
    Return a flat, readable message per bad field instead."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=engine.model_name,
        model_version=engine.model_version,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest, include_reason_codes: bool = True) -> ScoreResponse:
    result = engine.score(request.model_dump(), include_reason_codes=include_reason_codes)
    return ScoreResponse(
        request_id=str(uuid.uuid4()),
        calibrated_pd=result["calibrated_pd"],
        risk_band=result["risk_band"],
        top_3_risk_increasing_drivers_illustrative_only=result["top_3_risk_increasing_drivers_illustrative_only"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        warnings=result["warnings"],
    )
