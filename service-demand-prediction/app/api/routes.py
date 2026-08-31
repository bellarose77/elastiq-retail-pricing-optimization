"""HTTP routes for the demand prediction API. Thin by design -- see
app/services/forecast_service.py's module docstring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    ItemForecast,
    ProductForecast,
)
from app.services import forecast_service
from app.services.forecast_service import MissingArtifactsError

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/items", response_model=list[ItemForecast])
def get_items() -> list[dict[str, Any]]:
    """List every item with a next-period demand forecast."""

    try:
        return forecast_service.list_items()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/forecasts", response_model=ForecastResponse)
def post_forecasts(request: ForecastRequest) -> ForecastResponse:
    """Look up next-period demand forecasts for the requested item_ids."""

    try:
        forecasts, not_found, generated_from = forecast_service.get_forecasts(
            request.item_ids
        )
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ForecastResponse(
        forecasts=[ItemForecast(**row) for row in forecasts],
        not_found=not_found,
        generated_from=generated_from,
    )


@router.get("/products/forecasts", response_model=list[ProductForecast])
def get_product_forecasts() -> list[dict[str, Any]]:
    """One forecast per product, aggregated across its stores."""

    try:
        return forecast_service.list_product_forecasts()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/refresh")
def post_refresh() -> dict[str, str]:
    """Reload the forecast artifact from disk after a new pipeline run."""

    try:
        generated_from = forecast_service.refresh()
    except MissingArtifactsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"status": "refreshed", "generated_from": generated_from}
