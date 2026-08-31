"""Pricing business layer.

This is the boundary between the HTTP API (``app/api/routes.py``, which
only imports from this module and ``app/schemas.py``) and the actual
pricing algorithm (``src/optimization``, owned by the monorepo's data/ML
code, reused here rather than reimplemented -- see that package for the
elasticity-precedence reconciliation, guardrails, and grid search this
module composes but never redefines).

Every function here returns plain JSON-safe Python data (dicts/lists),
never a DataFrame or a FastAPI/Pydantic type, so routes.py stays a thin
HTTP wrapper and this module stays framework-agnostic.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.artifact_store import (
    ArtifactStore,
    MissingArtifactsError,
    get_store,
    refresh as refresh_store,
)
from src.optimization import (
    attach_elasticity_provenance,
    build_product_recommendation_summary,
    build_product_summary,
    build_withheld_recommendations,
    default_profit_configuration,
    optimize_price_portfolio,
    split_priceable_items,
)

__all__ = [
    "MissingArtifactsError",
    "list_items",
    "get_recommendations",
    "list_products",
    "list_product_recommendations",
    "refresh",
]

# The production pricing policy (causal elasticity only, -10%/+20% change
# band, 15% minimum margin, 15% competitor band) is shared with the batch
# pipeline via this same factory -- see its docstring in
# src/optimization/dataset.py for why each guardrail exists.
CONFIGURATION = default_profit_configuration()


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


def _score(
    dataframe: pd.DataFrame,
    optimization_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run the shared pricing logic over a (sub)set of decision units."""

    priceable, withheld = split_priceable_items(dataframe, CONFIGURATION)

    if len(priceable):
        recommendations = optimize_price_portfolio(
            priceable,
            item_column="item_id",
            current_price_column="selling_price",
            baseline_quantity_column="baseline_quantity_for_pricing",
            elasticity_column="elasticity_for_pricing",
            unit_cost_column="unit_cost",
            category_column="category",
            competitor_price_column="competitor_price",
            promotion_active_column="planned_promotion_flag",
            promotion_uplift_rate_column="promotion_uplift_rate_for_pricing",
            available_inventory_column="available_inventory_for_pricing",
            elasticity_lower_column="elasticity_lower_for_pricing",
            elasticity_upper_column="elasticity_upper_for_pricing",
            configuration=CONFIGURATION,
        )
    else:
        recommendations = pd.DataFrame()

    if len(withheld):
        withheld_rows = build_withheld_recommendations(
            withheld,
            objective=CONFIGURATION.objective,
        )
        recommendations = pd.concat(
            [recommendations, withheld_rows],
            ignore_index=True,
        )

    return attach_elasticity_provenance(recommendations, optimization_data)


def list_items() -> list[dict[str, Any]]:
    """List every priceable (product, store) decision unit."""

    store = get_store()
    columns = [
        "item_id",
        "product_id",
        "store_id",
        "category",
        "selling_price",
        "elasticity_source",
        "elasticity_is_causal",
    ]

    return _records(store.optimization_data[columns])


def get_recommendations(
    item_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], str]:
    """Score the requested (product, store) items.

    Returns (recommendations, not_found_ids, artifact_generated_from).
    """

    store = get_store()
    optimization_data = store.optimization_data
    requested = set(item_ids)
    matched = optimization_data[optimization_data["item_id"].isin(requested)]
    not_found = sorted(requested - set(matched["item_id"]))

    if matched.empty:
        return [], not_found, store.generated_from

    recommendations = _score(matched, optimization_data)

    return _records(recommendations), not_found, store.generated_from


def list_products() -> list[dict[str, Any]]:
    """Live equivalent of the frontend's bundled demo catalog.

    Scores every item and aggregates successful recommendations to one
    row per product -- the shape scripts/export_demo_data.py writes to
    app/frontend/src/lib/demoData.js.
    """

    store = get_store()
    recommendations = _score(store.optimization_data, store.optimization_data)
    products = build_product_summary(
        recommendations,
        store.elasticity_estimates,
        store.retail_rag,
    )

    return _records(products)


def list_product_recommendations() -> list[dict[str, Any]]:
    """One pricing decision per product, aggregated across its stores."""

    store = get_store()
    recommendations = _score(store.optimization_data, store.optimization_data)
    summary = build_product_recommendation_summary(recommendations)

    return _records(summary)


def refresh() -> str:
    """Reload artifacts from disk. Returns the new generated_from stamp."""

    store: ArtifactStore = refresh_store()

    return store.generated_from
