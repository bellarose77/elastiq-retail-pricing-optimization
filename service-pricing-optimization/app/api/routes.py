"""HTTP routes for the pricing API.

Thin by design: every handler here does request validation (via
``app/schemas.py``), delegates to ``app/services/pricing_service.py`` for
all actual work, and maps its results/errors onto HTTP responses. No
pricing logic, DataFrame handling, or filesystem access belongs in this
module.

Two routers: ``health_router`` is mounted at the root (unversioned) since
that's the path hosting platforms/orchestrators are typically configured
to poll and shouldn't move across API versions. ``router`` holds every
actual resource endpoint and is mounted under ``/api/v1`` (see
``app/main.py``), so a future breaking change can ship as ``/api/v2``
alongside it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ItemSummary,
    ProductRecommendation,
    ProductSummary,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services import pricing_service
from app.services.pricing_service import MissingArtifactsError

health_router = APIRouter()
router = APIRouter(prefix="/api/v1")


@health_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/items", response_model=list[ItemSummary])
def get_items() -> list[dict[str, Any]]:
    """List every priceable (product, store) decision unit."""

    try:
        return pricing_service.list_items()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/recommendations", response_model=RecommendationResponse)
def post_recommendations(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Score the requested (product, store) items and return recommendations.

    Items with no causally identified elasticity are returned with
    ``status="held_no_causal_elasticity"`` and a held (unchanged) price,
    matching the batch pipeline's guardrail against pricing on
    correlational evidence.
    """

    try:
        recommendations, not_found, generated_from = (
            pricing_service.get_recommendations(request.item_ids)
        )
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return RecommendationResponse(
        recommendations=[Recommendation(**row) for row in recommendations],
        not_found=not_found,
        generated_from=generated_from,
    )


@router.get("/products", response_model=list[ProductSummary])
def get_products() -> list[dict[str, Any]]:
    """Live equivalent of the frontend's bundled demo catalog (input data)."""

    try:
        return pricing_service.list_products()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get(
    "/products/recommendations",
    response_model=list[ProductRecommendation],
)
def get_product_recommendations() -> list[dict[str, Any]]:
    """One pricing decision per product, aggregated across its stores."""

    try:
        return pricing_service.list_product_recommendations()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/refresh")
def post_refresh() -> dict[str, str]:
    """Reload artifacts from disk after a new batch pipeline run."""

    try:
        generated_from = pricing_service.refresh()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"status": "refreshed", "generated_from": generated_from}
