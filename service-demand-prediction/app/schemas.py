"""Request/response models for the demand prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemForecast(BaseModel):
    """One (product, store) item's next-period demand forecast."""

    item_id: str
    product_id: str
    store_id: str
    category: str
    date: str
    forecast_quantity: float
    predicted_value: float
    prediction_interval_lower: float
    prediction_interval_upper: float


class ForecastRequest(BaseModel):
    """A batch request for item-level forecasts."""

    item_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "item_id values from GET /items, e.g. 'P001_S001'. Unknown "
            "ids are reported back under 'not_found' rather than failing "
            "the whole request."
        ),
    )


class ForecastResponse(BaseModel):
    """Forecasts for a requested batch, plus any unresolved ids."""

    forecasts: list[ItemForecast]
    not_found: list[str]
    generated_from: str = Field(
        description="Timestamp of the upstream forecast artifact used.",
    )


class ProductForecast(BaseModel):
    """One product's forecast, aggregated across its stores."""

    itemId: str
    forecastQuantity: float
    predictedValue: float
    predictionIntervalLower: float
    predictionIntervalUpper: float
    storeCount: int
