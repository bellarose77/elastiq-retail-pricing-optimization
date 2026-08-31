"""Demand-forecast business layer.

Thin by design: this service reuses no algorithm (the forecasting model
itself lives in src/models/forecasting.py and runs only as part of the
offline batch pipeline -- see artifact_store.py's docstring), so this
module's only job is slicing and aggregating the cached forecast table.
Every function returns plain JSON-safe Python data, never a DataFrame, so
api/routes.py stays a thin HTTP wrapper.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.artifact_store import MissingArtifactsError, get_store
from app.services.artifact_store import refresh as refresh_store

__all__ = [
    "MissingArtifactsError",
    "list_items",
    "get_forecasts",
    "list_product_forecasts",
    "refresh",
]


def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records (NaN/NaT -> None)."""

    if dataframe.empty:
        return []

    clean = dataframe.astype(object).where(pd.notnull(dataframe), None)

    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        for record in clean.to_dict(orient="records")
    ]


def list_items() -> list[dict[str, Any]]:
    """List every item with a next-period forecast."""

    store = get_store()

    return _records(store.forecasts)


def get_forecasts(item_ids: list[str]) -> tuple[list[dict[str, Any]], list[str], str]:
    """Look up the requested item_ids.

    Returns (forecasts, not_found_ids, artifact_generated_from).
    """

    store = get_store()
    requested = set(item_ids)
    matched = store.forecasts[store.forecasts["item_id"].isin(requested)]
    not_found = sorted(requested - set(matched["item_id"]))

    return _records(matched), not_found, store.generated_from


def list_product_forecasts() -> list[dict[str, Any]]:
    """One forecast per product, aggregated across its stores."""

    store = get_store()
    rows = []

    for product_id, group in store.forecasts.groupby("product_id", sort=True):
        rows.append(
            {
                "itemId": product_id,
                "forecastQuantity": round(
                    float(group["forecast_quantity"].sum()), 1
                ),
                "predictedValue": round(
                    float(group["predicted_value"].sum()), 1
                ),
                "predictionIntervalLower": round(
                    float(group["prediction_interval_lower"].sum()), 1
                ),
                "predictionIntervalUpper": round(
                    float(group["prediction_interval_upper"].sum()), 1
                ),
                "storeCount": int(len(group)),
            }
        )

    return rows


def refresh() -> str:
    """Reload the artifact from disk. Returns the new generated_from stamp."""

    store = refresh_store()

    return store.generated_from
