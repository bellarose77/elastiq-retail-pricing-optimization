"""Build the reconciled per-item dataset that price optimization runs on.

This is the shared data-preparation logic behind step 08 of the batch
pipeline (``src/pipelines/step_08_price_optimization.py``) and the online
pricing API (``app/api``). Both need the exact same elasticity-precedence
and baseline-demand reconciliation -- duplicating it would let the batch
recommendations and the API recommendations quietly drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.optimization.pricing import PricingOptimizationConfig

# Used only when a product's category has no elasticity fit at all (e.g.
# a brand-new category with no data yet). Every other case is handled by
# the precedence chain in ``build_optimization_dataset``.
DEFAULT_ELASTICITY = -1.5
DEFAULT_ELASTICITY_HALF_WIDTH = 1.0

# An elasticity outside this range is not a measurement of ordinary demand;
# it is a symptom of residual confounding. Positive values in particular
# imply an upward-sloping demand curve.
PLAUSIBLE_ELASTICITY_RANGE = (-8.0, -0.05)


def default_profit_configuration() -> PricingOptimizationConfig:
    """The production pricing policy shared by the batch pipeline and API.

    Only price on causally identified elasticities, within a -10%/+20%
    change band, at a 15% minimum margin, within 15% of the competitor
    price. See ``PricingOptimizationConfig`` field docstrings for why each
    guardrail exists.
    """

    return PricingOptimizationConfig(
        objective="profit",
        minimum_price_change_rate=-0.10,
        maximum_price_change_rate=0.20,
        minimum_margin_rate=0.15,
        allow_non_causal_pricing=False,
        enable_confidence_dampening=True,
        competitor_price_tolerance=0.15,
        promotion_cost_rate=0.02,
        cross_price_elasticity=0.0,
        price_ending=0.99,
    )


def build_optimization_dataset(
    retail_rag: pd.DataFrame,
    elasticity_estimates: pd.DataFrame,
    promotion_uplift: pd.DataFrame,
    iv_product_elasticity: pd.DataFrame,
    iv_pooled: dict[str, Any],
    *,
    default_elasticity: float = DEFAULT_ELASTICITY,
    default_elasticity_half_width: float = DEFAULT_ELASTICITY_HALF_WIDTH,
    plausible_elasticity_range: tuple[float, float] = (
        PLAUSIBLE_ELASTICITY_RANGE
    ),
    verbose: bool = True,
) -> pd.DataFrame:
    """Reconcile every upstream pipeline output into one per-item table.

    Elasticity is sourced in strict order of causal credibility:

      1. product_iv    -- the product's own 2SLS estimate, when its
                           first-stage F clears the weak-instrument
                           threshold and the estimate is economically
                           plausible.
      2. pooled_iv     -- the portfolio-wide 2SLS estimate from step 04.
                           Loses cross-product heterogeneity but is still
                           causally identified.
      3. shrunk_ols    -- step 03's shrunk estimate, and ONLY if it is
                           economically plausible. Correlational, so it
                           travels with ``elasticity_is_causal=False``.
      4. flat default  -- last resort for a product with no usable
                           estimate at all.

    Returns one row per ``(product_id, store_id)`` decision unit, keyed by
    ``item_id``, carrying everything ``optimize_price_portfolio`` needs
    plus ``elasticity_source`` / ``elasticity_is_causal`` provenance.
    """

    iv_lookup = iv_product_elasticity.set_index("product_id")

    pooled_iv_elasticity = float(iv_pooled["pooled_iv_elasticity"])
    pooled_iv_lower = float(iv_pooled["confidence_interval_lower"])
    pooled_iv_upper = float(iv_pooled["confidence_interval_upper"])
    pooled_iv_is_usable = not bool(iv_pooled["weak_instrument_flag"])

    elasticity_lookup = {
        row["product_id"]: row
        for _, row in elasticity_estimates.iterrows()
        if pd.notna(row.get("shrunk_elasticity"))
    }

    def _resolve_elasticity(product_id: object) -> dict[str, object]:
        """Pick the most causally credible elasticity for one product."""

        if product_id in iv_lookup.index:
            row = iv_lookup.loc[product_id]

            if bool(row["is_reliable"]):
                return {
                    "elasticity_for_pricing": float(row["iv_elasticity"]),
                    "elasticity_lower_for_pricing": float(
                        row["confidence_interval_lower"]
                    ),
                    "elasticity_upper_for_pricing": float(
                        row["confidence_interval_upper"]
                    ),
                    "elasticity_source": "product_iv",
                    "elasticity_is_causal": True,
                    "first_stage_f_statistic": float(
                        row["first_stage_f_statistic"]
                    ),
                }

        if pooled_iv_is_usable:
            return {
                "elasticity_for_pricing": pooled_iv_elasticity,
                "elasticity_lower_for_pricing": pooled_iv_lower,
                "elasticity_upper_for_pricing": pooled_iv_upper,
                "elasticity_source": "pooled_iv",
                "elasticity_is_causal": True,
                "first_stage_f_statistic": float(
                    iv_pooled["first_stage_f_statistic"]
                ),
            }

        shrunk = elasticity_lookup.get(product_id)

        if shrunk is not None and plausible_elasticity_range[0] <= shrunk[
            "shrunk_elasticity"
        ] <= plausible_elasticity_range[1]:
            return {
                "elasticity_for_pricing": float(
                    shrunk["shrunk_elasticity"]
                ),
                "elasticity_lower_for_pricing": float(
                    shrunk["shrunk_confidence_interval_lower"]
                ),
                "elasticity_upper_for_pricing": float(
                    shrunk["shrunk_confidence_interval_upper"]
                ),
                "elasticity_source": "shrunk_ols",
                "elasticity_is_causal": False,
                "first_stage_f_statistic": np.nan,
            }

        return {
            "elasticity_for_pricing": default_elasticity,
            "elasticity_lower_for_pricing": (
                default_elasticity - default_elasticity_half_width
            ),
            "elasticity_upper_for_pricing": (
                default_elasticity + default_elasticity_half_width
            ),
            "elasticity_source": "default_fallback",
            "elasticity_is_causal": False,
            "first_stage_f_statistic": np.nan,
        }

    resolved_elasticity = pd.DataFrame(
        [
            {
                "product_id": product_id,
                **_resolve_elasticity(product_id),
            }
            for product_id in sorted(
                elasticity_estimates["product_id"].unique()
            )
        ]
    )

    if verbose:
        source_counts = resolved_elasticity["elasticity_source"].value_counts()

        for source in (
            "product_iv",
            "pooled_iv",
            "shrunk_ols",
            "default_fallback",
        ):
            count = int(source_counts.get(source, 0))

            if count:
                print(f"  {count:>3} product(s) priced from {source}")

        causal_share = float(
            resolved_elasticity["elasticity_is_causal"].mean()
        )

        print(
            f"  {causal_share:.0%} of products have a causally identified "
            "elasticity\n"
        )

    # Step 06 is the do-nothing demand baseline for a future price decision.
    has_forecast = (
        "forecast_quantity" in retail_rag.columns
        and pd.to_numeric(
            retail_rag["forecast_quantity"], errors="coerce"
        ).notna().any()
    )

    if has_forecast:
        snapshot_index = retail_rag.groupby(
            ["product_id", "store_id"]
        )["date"].idxmax()
        snapshot = retail_rag.loc[snapshot_index].reset_index(drop=True)
        snapshot["baseline_quantity_raw"] = pd.to_numeric(
            snapshot["forecast_quantity"], errors="coerce"
        )
        snapshot["baseline_quantity_source"] = "next_period_forecast"
        snapshot["baseline_is_censored"] = False

        if verbose:
            print(
                f"  Using {len(snapshot):,} next-period forecasts as the "
                "decision baseline"
            )
    else:
        stockout_flag = (
            retail_rag["stockout_flag"].fillna(0).astype(int)
            if "stockout_flag" in retail_rag.columns
            else pd.Series(0, index=retail_rag.index)
        )
        clean_rows = retail_rag.loc[stockout_flag.eq(0)]
        snapshot_index = clean_rows.groupby(
            ["product_id", "store_id"]
        )["date"].idxmax()
        snapshot = clean_rows.loc[snapshot_index].reset_index(drop=True)
        snapshot["baseline_quantity_raw"] = pd.to_numeric(
            snapshot["units_sold"], errors="coerce"
        )
        snapshot["baseline_quantity_source"] = "historical_non_stockout"
        snapshot["baseline_is_censored"] = False

        if verbose:
            print(
                "  WARNING: next-period forecasts were unavailable; using "
                "historical non-stockout demand"
            )

    if snapshot["baseline_quantity_raw"].isna().any():
        raise ValueError("Optimization baseline contains missing quantities")

    # Convert retrieved evidence into a bounded, transparent demand overlay.
    if "rag_weighted_impact_score" not in snapshot.columns:
        snapshot["rag_weighted_impact_score"] = 0.0
    snapshot["rag_weighted_impact_score"] = pd.to_numeric(
        snapshot["rag_weighted_impact_score"], errors="coerce"
    ).fillna(0.0)
    snapshot["market_demand_adjustment_rate"] = (
        snapshot["rag_weighted_impact_score"] * 0.10
    ).clip(lower=-0.15, upper=0.15)
    snapshot["baseline_quantity_for_pricing"] = (
        snapshot["baseline_quantity_raw"]
        * (1.0 + snapshot["market_demand_adjustment_rate"])
    ).clip(lower=0.0)

    if "available_inventory_for_pricing" not in snapshot.columns:
        inventory_source = (
            "inventory_level_lag_1"
            if "inventory_level_lag_1" in snapshot.columns
            else "inventory_level"
        )
        snapshot["available_inventory_for_pricing"] = pd.to_numeric(
            snapshot[inventory_source], errors="coerce"
        )

    optimization_data = (
        snapshot
        .merge(
            resolved_elasticity,
            on="product_id",
            how="left",
        )
        .merge(
            promotion_uplift[
                [
                    column
                    for column in (
                        "product_id",
                        "promotion_uplift_rate",
                        "promotion_uplift_rate_for_pricing",
                        "promotion_model_is_reliable",
                        "promotion_reliability_reason",
                    )
                    if column in promotion_uplift.columns
                ]
            ],
            on="product_id",
            how="left",
        )
    )

    optimization_data["elasticity_source"] = optimization_data[
        "elasticity_source"
    ].fillna("default_fallback")

    optimization_data["elasticity_is_causal"] = (
        optimization_data["elasticity_is_causal"].fillna(False)
    )

    optimization_data["elasticity_for_pricing"] = optimization_data[
        "elasticity_for_pricing"
    ].fillna(default_elasticity)

    optimization_data["elasticity_lower_for_pricing"] = optimization_data[
        "elasticity_lower_for_pricing"
    ].fillna(default_elasticity - default_elasticity_half_width)

    optimization_data["elasticity_upper_for_pricing"] = optimization_data[
        "elasticity_upper_for_pricing"
    ].fillna(default_elasticity + default_elasticity_half_width)

    if "promotion_uplift_rate_for_pricing" not in optimization_data.columns:
        optimization_data["promotion_uplift_rate_for_pricing"] = 0.0

    optimization_data["promotion_uplift_rate_for_pricing"] = pd.to_numeric(
        optimization_data["promotion_uplift_rate_for_pricing"],
        errors="coerce",
    ).fillna(0.0)
    if "promotion_model_is_reliable" not in optimization_data.columns:
        optimization_data["promotion_model_is_reliable"] = False
    optimization_data["promotion_model_is_reliable"] = optimization_data[
        "promotion_model_is_reliable"
    ].fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    if "promotion_reliability_reason" not in optimization_data.columns:
        optimization_data["promotion_reliability_reason"] = (
            "promotion_model_not_supplied"
        )
    optimization_data["planned_promotion_flag"] = (
        optimization_data["promotion_model_is_reliable"]
        & optimization_data["promotion_uplift_rate_for_pricing"].ge(0.20)
    ).astype(int)

    optimization_data["item_id"] = (
        optimization_data["product_id"].astype(str)
        + "_"
        + optimization_data["store_id"].astype(str)
    )

    return optimization_data


def split_priceable_items(
    optimization_data: pd.DataFrame,
    configuration: PricingOptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into items with a causal elasticity and items to hold.

    Items without a causally identified elasticity are not priced unless
    ``configuration.allow_non_causal_pricing`` opts in, mirroring the
    guardrail in ``optimize_price_portfolio``'s caller.
    """

    if configuration.allow_non_causal_pricing:
        return optimization_data, optimization_data.iloc[0:0]

    is_causal = optimization_data["elasticity_is_causal"].astype(bool)

    return (
        optimization_data.loc[is_causal],
        optimization_data.loc[~is_causal],
    )


def build_withheld_recommendations(
    withheld: pd.DataFrame,
    *,
    objective: str,
) -> pd.DataFrame:
    """Build explicit "held" rows for items with no causal elasticity.

    Reported as holds with a machine-readable reason so a caller sees
    every requested item accounted for rather than a silent gap.
    """

    if not len(withheld):
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "item_id": withheld["item_id"].to_numpy(),
            "category": withheld["category"].to_numpy(),
            "status": "held_no_causal_elasticity",
            "error_message": (
                "No causally identified elasticity available; the "
                "correlational estimate was not used. Re-run step 04 or "
                "supply a stronger instrument."
            ),
            "optimization_objective": objective,
            "current_price": withheld["selling_price"].to_numpy(),
            "recommended_price": withheld["selling_price"].to_numpy(),
            "price_change": 0.0,
            "price_change_rate": 0.0,
            "recommendation_action": "hold_price",
            "baseline_quantity": withheld[
                "baseline_quantity_for_pricing"
            ].to_numpy(),
            "expected_quantity": withheld[
                "baseline_quantity_for_pricing"
            ].to_numpy(),
            "elasticity": withheld["elasticity_for_pricing"].to_numpy(),
            "unit_cost": withheld["unit_cost"].to_numpy(),
            "elasticity_source": withheld["elasticity_source"].to_numpy(),
            "elasticity_is_causal": False,
        }
    )


def attach_elasticity_provenance(
    recommendations: pd.DataFrame,
    optimization_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge elasticity/baseline provenance onto a recommendations table."""

    provenance = optimization_data[
        [
            "item_id",
            "elasticity_source",
            "elasticity_is_causal",
            "baseline_is_censored",
            "baseline_quantity_source",
            "baseline_quantity_raw",
            "baseline_quantity_for_pricing",
            "market_demand_adjustment_rate",
            "rag_weighted_impact_score",
            "promotion_model_is_reliable",
            "promotion_reliability_reason",
        ]
    ]

    return recommendations.drop(
        columns=[
            column
            for column in ("elasticity_source", "elasticity_is_causal")
            if column in recommendations.columns
        ]
    ).merge(provenance, on="item_id", how="left")
