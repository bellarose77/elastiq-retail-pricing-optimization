"""Aggregate per-store recommendations into one row per product.

Shared by ``scripts/export_demo_data.py`` (writes the frontend's bundled
offline snapshot, ``app/frontend/src/lib/demoData.js``) and the online
pricing API's ``GET /products`` (serves the same shape live). Both need
identical aggregation so the frontend's live and bundled data sources
never silently diverge in field meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_product_summary(
    recommendations: pd.DataFrame,
    elasticity_estimates: pd.DataFrame,
    rag_features: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate successful recommendations down to one row per product.

    Only ``status == "success"`` rows are aggregated -- held items (no
    causal elasticity) and infeasible items are excluded, matching what
    the frontend's demo catalog has always shown.
    """

    successful = recommendations.loc[
        recommendations["status"] == "success"
    ].copy()

    item_id_parts = successful["item_id"].str.split("_", n=1, expand=True)
    successful["product_id"] = item_id_parts[0]
    successful["store_id"] = item_id_parts[1]

    # Every recommendation now carries its own elasticity provenance
    # (product_iv / pooled_iv / shrunk_ols / default_fallback). Re-deriving
    # it here from step 03's `is_reliable` flag alone produces a weaker
    # answer, so it is only a fallback for a recommendations table that
    # predates that column.
    if "elasticity_source" in successful.columns:
        elasticity_source = (
            successful[["product_id", "elasticity_source"]]
            .drop_duplicates("product_id")
        )
    else:
        elasticity_source = elasticity_estimates[
            ["product_id", "is_reliable"]
        ].assign(
            elasticity_source=lambda df: np.where(
                df["is_reliable"], "product_estimate", "default_fallback"
            )
        )[["product_id", "elasticity_source"]]

    rag_context = (
        rag_features.groupby("product_id")
        .agg(data_as_of_date=("date", "max"))
        .reset_index()
    )

    # Avoid an `elasticity_source_x` / `_y` collision when the column is
    # already present on the recommendations (the normal case now).
    merged = (
        successful
        if "elasticity_source" in successful.columns
        else successful.merge(
            elasticity_source, on="product_id", how="left"
        )
    ).merge(rag_context, on="product_id", how="left")

    products = []

    for product_id, group in merged.groupby("product_id", sort=True):
        promotion_mode = group["promotion_active"].mode()
        products.append(
            {
                "itemId": product_id,
                "productId": product_id,
                "category": group["category"].iloc[0],
                "currentPrice": round(float(group["current_price"].mean()), 2),
                "unitCost": round(float(group["unit_cost"].mean()), 2),
                "competitorPrice": round(
                    float(group["competitor_price"].mean()), 2
                ),
                "inventory": int(round(group["available_inventory"].sum())),
                "baselineQuantity": round(
                    float(group["baseline_quantity"].sum()), 1
                ),
                "elasticity": round(float(group["elasticity"].iloc[0]), 3),
                "elasticitySource": group["elasticity_source"].iloc[0],
                "elasticityIsCausal": bool(
                    group["elasticity_is_causal"].iloc[0]
                ),
                "baselineSource": group.get(
                    "baseline_quantity_source",
                    pd.Series(["pipeline_forecast"]),
                ).iloc[0],
                "promotionUpliftRate": round(
                    float(group["promotion_uplift_rate"].iloc[0]), 3
                ),
                "promotionModelReliable": bool(
                    group.get(
                        "promotion_model_is_reliable",
                        pd.Series([False]),
                    ).iloc[0]
                ),
                "confidence": round(
                    float(group["confidence_dampening_factor"].iloc[0]), 3
                ),
                "ragSignal": round(
                    float(group.get(
                        "market_demand_adjustment_rate",
                        pd.Series([0.0]),
                    ).iloc[0] or 0.0),
                    3,
                ),
                # Normalized explicitly: callers may pass `rag_features`
                # with `date` already parsed to datetime64 (the online API
                # does, since it also needs real dates for the demand
                # baseline snapshot) or left as a plain ISO string (the
                # export script does). str() of a Timestamp includes a
                # spurious "00:00:00", so format both cases the same way.
                "dataAsOfDate": pd.Timestamp(
                    group["data_as_of_date"].iloc[0]
                ).strftime("%Y-%m-%d"),
                "promotionFlag": (
                    int(promotion_mode.iloc[0])
                    if not promotion_mode.empty
                    else 0
                ),
            }
        )

    return pd.DataFrame(products)


def build_product_recommendation_summary(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate every (product, store) recommendation to one row per product.

    Unlike ``build_product_summary`` (which describes the *input* catalog
    and only includes successfully priced rows), this aggregates the
    *decision* itself -- recommended price and expected outcome -- across
    every store a product is sold in, held items included, so a caller
    asking "what should product P001 cost" gets one answer that accounts
    for the whole product rather than one store.

    Only the few fields that can't be cheaply recomputed by a caller who
    already has the product's current price and cost are included
    (recommended price, expected quantity/revenue/profit, status). Price
    change, margin, and action classification are left to the caller so
    this stays a thin aggregation over ``optimize_price_portfolio``'s own
    output rather than a second decision layer.
    """

    working = recommendations.copy()
    item_id_parts = working["item_id"].str.split("_", n=1, expand=True)
    working["product_id"] = item_id_parts[0]

    rows = []

    for product_id, group in working.groupby("product_id", sort=True):
        statuses = group["status"].unique()
        status = statuses[0] if len(statuses) == 1 else "partial"

        # Held rows (see build_withheld_recommendations) carry no
        # expected_revenue/expected_profit -- summing skips them (NaN),
        # so a product with some held stores reports revenue/profit from
        # its successfully priced stores only, not zero or an error.
        rows.append(
            {
                "itemId": product_id,
                "status": status,
                "recommendedPrice": round(
                    float(group["recommended_price"].mean()), 2
                ),
                "expectedQuantity": round(
                    float(group["expected_quantity"].sum()), 1
                ),
                "expectedRevenue": round(
                    float(group["expected_revenue"].sum())
                    if "expected_revenue" in group.columns
                    else 0.0,
                    2,
                ),
                "expectedProfit": round(
                    float(group["expected_profit"].sum())
                    if "expected_profit" in group.columns
                    else 0.0,
                    2,
                ),
                "storeCount": int(len(group)),
            }
        )

    return pd.DataFrame(rows)
