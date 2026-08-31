"""ELASTIQ feature generation API -- FastAPI application entry point.

Serves structured market-intelligence features computed by the batch
pipeline's step 07 (src/features/text.py's retrieval-based feature
extraction, run via src/pipelines/step_07_rag_features.py). This service
does NOT run retrieval or feature extraction itself -- it loads that
step's already-computed artifact (retail_with_rag_features.csv) and
serves it, so recommendations here always match what the batch pipeline
produced.

Run locally (from this directory):

    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8002

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
