"""ELASTIQ demand prediction API -- FastAPI application entry point.

Serves next-period demand forecasts computed by the batch pipeline's
step 06 (src/models/forecasting.py's XGBoost model, run via
src/pipelines/step_06_demand_forecasting.py). This service does NOT
train or run the model itself -- it loads that step's already-computed
artifact (xgboost_next_period_forecast.csv) and serves it, so
recommendations here always match what the batch pipeline produced.

Run locally (from this directory):

    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8001

See README.md for full setup, configuration, and Docker instructions.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title=settings.title,
    description=settings.description,
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
