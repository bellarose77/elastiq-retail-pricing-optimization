"""HTTP routes for the feature generation API. Thin by design -- see
app/services/feature_service.py's module docstring.

Two routers, mirroring service-pricing-optimization/app/api/routes.py:
``health_router`` is mounted at the root (unversioned) since that's the
path hosting platforms/orchestrators are typically configured to poll
and shouldn't move across API versions. ``router`` holds every actual
resource endpoint and is mounted under ``/api/v1`` (see app/main.py), so
a future breaking change can ship as ``/api/v2`` alongside it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas import (
    FeatureRequest,
    FeatureResponse,
    ItemFeatures,
    ProductFeatures,
)
from app.services import feature_service
from app.services.feature_service import MissingArtifactsError

health_router = APIRouter()
router = APIRouter(prefix="/api/v1")


@health_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/items", response_model=list[ItemFeatures])
def get_items() -> list[dict[str, Any]]:
    """List every item with generated market-intelligence features."""

    try:
        return feature_service.list_items()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/features", response_model=FeatureResponse)
def post_features(request: FeatureRequest) -> FeatureResponse:
    """Look up generated features for the requested item_ids."""

    try:
        features, not_found, generated_from = feature_service.get_features(
            request.item_ids
        )
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return FeatureResponse(
        features=[ItemFeatures(**row) for row in features],
        not_found=not_found,
        generated_from=generated_from,
    )


@router.get("/products/features", response_model=list[ProductFeatures])
def get_product_features() -> list[dict[str, Any]]:
    """One feature summary per product, aggregated across its stores."""

    try:
        return feature_service.list_product_features()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/refresh")
def post_refresh() -> dict[str, str]:
    """Reload the feature artifact from disk after a new pipeline run."""

    try:
        generated_from = feature_service.refresh()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"status": "refreshed", "generated_from": generated_from}
