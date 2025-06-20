"""FastAPI service exposing demand forecasts.

Run with::

    uvicorn app.api:app --host 0.0.0.0 --port 8000

Then POST a history series to ``/forecast``::

    curl -s localhost:8000/forecast -H 'content-type: application/json' \\
      -d '{"history": [3,0,0,5,0,2,0,4], "horizon": 7, "model": "croston_sba"}'
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.service import available_models, forecast

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Foundation, supervised and classical demand forecasts.",
    version="0.1.0",
)


class ForecastRequest(BaseModel):
    history: list[float] = Field(..., min_length=2, description="Past demand values")
    horizon: int = Field(28, ge=1, le=365, description="Steps to forecast")
    model: str = Field("croston_sba", description="Model name; see /models")
    quantile_levels: list[float] | None = Field(
        None, description="Quantiles in (0,1); defaults to deciles 0.1..0.9"
    )
    season_length: int = Field(7, ge=1, description="Seasonal period (7 = weekly)")


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    point: list[float]
    quantile_levels: list[float]
    quantiles: dict[str, list[float]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    return {"models": available_models()}


@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        result = forecast(
            history=req.history,
            horizon=req.horizon,
            model=req.model,
            quantile_levels=req.quantile_levels,
            season_length=req.season_length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForecastResponse(**result)
