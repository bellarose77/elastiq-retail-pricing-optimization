"""
Regenerate the frontend's demo seed data from real pipeline output.

app/frontend/src/lib/demoData.js feeds the ELASTIQ demo app's client-side
optimizer (app/frontend/src/lib/engine.js). The demo is a static site with
no backend, so "real data" here means: aggregate the actual, freshly
computed data/processed/price_optimization_recommendations.csv (plus its
elasticity and RAG-signal sources) down to one row per product and write
it out as a JS module, instead of hand-typed or stale numbers.

Run after `python -m src.pipelines.step_08_price_optimization` (and its
upstream steps) to keep the demo in sync with the real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT

OUTPUT_FILE = (
    PROJECT_ROOT / "app" / "frontend" / "src" / "lib" / "demoData.js"
)

def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommendations = pd.read_csv(
        PROCESSED_DIR / "price_optimization_recommendations.csv"
    )
    elasticity_estimates = pd.read_csv(
        PROCESSED_DIR / "product_elasticity_estimates.csv"
    )
    rag_features = pd.read_csv(
        PROCESSED_DIR / "retail_with_rag_features.csv"
    )
    return recommendations, elasticity_estimates, rag_features


def _build_product_rows(
    recommendations: pd.DataFrame,
    elasticity_estimates: pd.DataFrame,
    rag_features: pd.DataFrame,
) -> pd.DataFrame:
    successful = recommendations.loc[
        recommendations["status"] == "success"
    ].copy()

    item_id_parts = successful["item_id"].str.split("_", n=1, expand=True)
    successful["product_id"] = item_id_parts[0]
    successful["store_id"] = item_id_parts[1]

    # Step 08 now records the authoritative elasticity provenance on every
    # recommendation (product_iv / pooled_iv / shrunk_ols / default_fallback).
    # Re-deriving it here from step 03's `is_reliable` flag produced a
    # different, weaker answer and was how the committed demo data came to
    # claim "default_fallback" for 9 of 10 products.
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
                "dataAsOfDate": str(group["data_as_of_date"].iloc[0]),
                "promotionFlag": int(promotion_mode.iloc[0]) if not promotion_mode.empty else 0,
            }
        )

    return pd.DataFrame(products)


def _format_js_row(row: pd.Series) -> str:
    fields = [
        f'itemId: "{row.itemId}"',
        f'productId: "{row.productId}"',
        f'category: "{row.category}"',
        f"currentPrice: {row.currentPrice}",
        f"unitCost: {row.unitCost}",
        f"competitorPrice: {row.competitorPrice}",
        f"inventory: {row.inventory}",
        f"baselineQuantity: {row.baselineQuantity}",
        f"elasticity: {row.elasticity}",
        f'elasticitySource: "{row.elasticitySource}"',
        f"elasticityIsCausal: {str(bool(row.elasticityIsCausal)).lower()}",
        f'baselineSource: "{row.baselineSource}"',
        f"promotionUpliftRate: {row.promotionUpliftRate}",
        f"promotionModelReliable: {str(bool(row.promotionModelReliable)).lower()}",
        f"confidence: {row.confidence}",
        f"ragSignal: {row.ragSignal}",
        f"promotionFlag: {row.promotionFlag}",
        f'dataAsOfDate: "{row.dataAsOfDate}"',
    ]
    return "  { " + ", ".join(fields) + " },"


def _write_js_module(products: pd.DataFrame, output_file: Path) -> None:
    rows = "\n".join(
        _format_js_row(row) for _, row in products.iterrows()
    )

    content = (
        "/* ==========================================================================\n"
        "   DEMO DATA — auto-generated by scripts/export_demo_data.py from the real,\n"
        "   freshly computed data/processed/price_optimization_recommendations.csv\n"
        "   (plus product_elasticity_estimates.csv and retail_with_rag_features.csv),\n"
        "   aggregated to one row per product. Do not hand-edit; re-run the export\n"
        "   script after a fresh pipeline run instead.\n"
        "   ========================================================================== */\n"
        "export const DEMO_DATA = [\n"
        f"{rows}\n"
        "];\n"
    )

    output_file.write_text(content, encoding="utf-8")


def main() -> None:
    recommendations, elasticity_estimates, rag_features = _load_inputs()

    products = _build_product_rows(
        recommendations, elasticity_estimates, rag_features
    )

    if products.empty:
        raise ValueError(
            "No successful recommendations were found to export. Run "
            "the pipeline (steps 01-08) first."
        )

    _write_js_module(products, OUTPUT_FILE)

    print(f"Exported {len(products)} products to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
