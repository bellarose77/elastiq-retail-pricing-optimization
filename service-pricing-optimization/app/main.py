"""ELASTIQ pricing API -- FastAPI application entry point.

Exposes the batch pipeline's price-optimization step
(src/optimization/pricing.py's ``optimize_price_portfolio``) as a
request/response service. It does NOT re-run the batch pipeline: it loads
steps 01-07's already-computed artifacts (elasticity, causal IV,
promotion uplift, forecasts, market features) and scores requested items
against them with the same reconciliation and pricing-policy code the
batch pipeline uses, so recommendations here always match what
``python scripts/run_pipeline.py`` would produce.

Run locally (from this directory):

    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

See README.md for the full setup, configuration, and Docker instructions.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health_router, router
from app.config import settings

app = FastAPI(
    title=settings.title,
    description=settings.description,
    version=settings.version,
)

# The frontend (a static SPA, normally opened at http://127.0.0.1:5173 via
# the Vite dev server, or hosted with no backend at all on GitHub Pages)
# is a different origin from this API, so it needs CORS to call it
# directly. Configurable via PRICING_API_CORS_ALLOW_ORIGINS (see
# app/config.py) for other setups (a different dev port, a deployed
# frontend origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(router)
