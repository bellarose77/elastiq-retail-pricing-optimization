"""
Pipeline Step 08: Constrained Price Optimization

This pipeline:
1. Loads all upstream pipeline outputs
2. Takes the latest observed snapshot per (product, store)
3. Reconciles elasticity estimates using step_03's empirical-Bayes
   shrinkage toward each product's category, with a flat default only
   for the pathological case of a product whose category has no
   elasticity fit at all
4. Runs constrained, uncertainty- and competitor-aware price optimization
5. Validates recommendations and generates a sensitivity analysis
6. Saves optimal pricing recommendations
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data import load_csv, save_csv, save_json
from src.optimization.pricing import (
    PricingOptimizationConfig,
    build_optimization_action_summary,
    build_optimization_category_summary,
    build_optimization_configuration,
    optimize_price_portfolio,
    run_price_sensitivity_analysis,
    summarize_optimization_portfolio,
    summarize_recommendation_validation,
    validate_price_recommendations,
    validate_pricing_configuration,
)
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)

# Used only when a product's category has no elasticity fit at all (e.g.
# a brand-new category with no data yet) -- step_03's shrinkage already
# handles every other case by blending toward the category baseline.
DEFAULT_ELASTICITY = -1.5
DEFAULT_ELASTICITY_HALF_WIDTH = 1.0

# An elasticity outside this range is not a measurement of ordinary demand;
# it is a symptom of residual confounding. Positive values in particular
# imply an upward-sloping demand curve.
PLAUSIBLE_ELASTICITY_RANGE = (-8.0, -0.05)

# Permitted price-change band for items we are willing to reprice.
PROFIT_CONFIGURATION_MIN_RATE = -0.10
PROFIT_CONFIGURATION_MAX_RATE = 0.20


def main() -> None:
    """Execute the price optimization pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 08 — CONSTRAINED PRICE OPTIMIZATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define required upstream inputs
    # ---------------------------------------------------------

    REQUIRED_FILES = {
        "retail_rag": PROCESSED_DIR / "retail_with_rag_features.csv",
        "elasticity": PROCESSED_DIR / "product_elasticity_estimates.csv",
        "promotion_uplift": PROCESSED_DIR / "promotion_uplift_by_product.csv",
        # Step 04's causal estimates are now a hard dependency: pricing on
        # correlational elasticities is what made this step recommend a
        # price rise for 41 of 43 items on data whose true elasticities
        # were uniformly elastic.
        "iv_product_elasticity": (
            PROCESSED_DIR / "iv_product_elasticity_estimates.csv"
        ),
        "iv_pooled_elasticity": PROCESSED_DIR / "iv_pooled_elasticity.json",
    }

    OUTPUT_FILES = {
        "recommendations": PROCESSED_DIR / "price_optimization_recommendations.csv",
        "portfolio_summary": PROCESSED_DIR / "price_optimization_portfolio_summary.csv",
        "validation_summary": PROCESSED_DIR / "price_optimization_validation_summary.csv",
        "action_summary": PROCESSED_DIR / "price_optimization_action_summary.csv",
        "category_summary": PROCESSED_DIR / "price_optimization_category_summary.csv",
        "sensitivity_analysis": PROCESSED_DIR / "price_optimization_sensitivity_analysis.csv",
        "configuration": PROCESSED_DIR / "price_optimization_configuration.json",
    }

    # ---------------------------------------------------------
    # Check dependencies and load upstream datasets
    # ---------------------------------------------------------

    print("\nChecking upstream pipeline dependencies...")

    check_pipeline_dependencies(
        REQUIRED_FILES,
        pipeline_name="step_08_price_optimization",
    )

    print("All required upstream outputs found")
    for name, path in REQUIRED_FILES.items():
        print(f"  - {path.name}")

    print("\nLoading upstream pipeline outputs...")

    retail_rag = load_csv(
        REQUIRED_FILES["retail_rag"],
        parse_dates=["date"],
        low_memory=False,
    )
    elasticity_estimates = load_csv(REQUIRED_FILES["elasticity"])
    promotion_uplift = load_csv(REQUIRED_FILES["promotion_uplift"])

    iv_product_elasticity = load_csv(
        REQUIRED_FILES["iv_product_elasticity"]
    )

    with open(
        REQUIRED_FILES["iv_pooled_elasticity"],
        encoding="utf-8",
    ) as handle:
        iv_pooled = json.load(handle)

    print(f"Retail + RAG data: {len(retail_rag):,} rows")
    print(f"Elasticity estimates: {len(elasticity_estimates):,} products")
    print(f"Promotion uplift: {len(promotion_uplift):,} products\n")

    # ---------------------------------------------------------
    # Define the optimization configuration
    # ---------------------------------------------------------
    #
    # Defined here, ahead of the elasticity reconciliation below, because
    # `allow_non_causal_pricing` gates whether non-causal elasticities are
    # priced at all. It lives on the config object -- not as a standalone
    # module constant -- so a policy change always shows up verbatim in
    # this run's saved price_optimization_configuration.json instead of
    # being an untracked local edit.

    print("Configuring optimization constraints...")

    PROFIT_CONFIGURATION = PricingOptimizationConfig(
        objective="profit",
        minimum_price_change_rate=PROFIT_CONFIGURATION_MIN_RATE,
        maximum_price_change_rate=PROFIT_CONFIGURATION_MAX_RATE,
        minimum_margin_rate=0.15,
        # Only price on causally identified elasticities by default. Set
        # this to True to also act on step 03's correlational (OLS/shrunk)
        # estimates -- which on this dataset are biased toward zero badly
        # enough to invert the pricing prescription.
        allow_non_causal_pricing=False,
        enable_confidence_dampening=True,
        competitor_price_tolerance=0.15,
        promotion_cost_rate=0.02,
        # The available cross-item adjustment is a fixed-point heuristic
        # (see PricingOptimizationConfig.cannibalization_max_iterations),
        # not a jointly solved demand system. Keep it disabled in the
        # decision pipeline until cross-elasticities are estimated and
        # the portfolio objective is solved jointly.
        cross_price_elasticity=0.0,
        price_ending=0.99,
    )

    validate_pricing_configuration(PROFIT_CONFIGURATION)

    print(
        "  Profit objective: "
        f"{PROFIT_CONFIGURATION.minimum_price_change_rate:.0%} to "
        f"{PROFIT_CONFIGURATION.maximum_price_change_rate:.0%} price change, "
        f"{PROFIT_CONFIGURATION.minimum_margin_rate:.0%} minimum margin, "
        f"competitor band ±{PROFIT_CONFIGURATION.competitor_price_tolerance:.0%}\n"
    )

    # ---------------------------------------------------------
    # Reconcile elasticity estimates with a confidence-aware fallback
    # ---------------------------------------------------------

    print("Reconciling elasticity estimates...")

    # ---------------------------------------------------------------------
    # Elasticity precedence chain
    # ---------------------------------------------------------------------
    #
    # Price is endogenous: retailers cut prices in weak weeks and raise
    # them in strong ones, so an ordinary least-squares slope absorbs the
    # demand shock. On this dataset that bias is not subtle -- step 03's
    # per-product OLS returns a POSITIVE elasticity for 8 of 10 products,
    # and the empirical-Bayes shrinkage then quietly replaces those with a
    # plausible-looking category baseline, so nothing downstream can tell
    # an estimate from a substitution.
    #
    # Elasticity is therefore now sourced in strict order of causal
    # credibility:
    #
    #   1. product_iv    -- the product's own 2SLS estimate, when its
    #                       first-stage F clears the weak-instrument
    #                       threshold and the estimate is economically
    #                       plausible.
    #   2. pooled_iv     -- the portfolio-wide 2SLS estimate from step 04.
    #                       Loses cross-product heterogeneity but is still
    #                       causally identified.
    #   3. shrunk_ols    -- step 03's shrunk estimate, and ONLY if it is
    #                       economically plausible. Correlational, so it is
    #                       flagged and (below) barred from driving a move
    #                       unless the operator opts in.
    #   4. flat default  -- last resort for a product with no usable
    #                       estimate at all.
    #
    # `elasticity_is_causal` travels with each row so the optimizer can
    # refuse to reprice on non-causal evidence.

    iv_lookup = iv_product_elasticity.set_index("product_id")

    pooled_iv_elasticity = float(iv_pooled["pooled_iv_elasticity"])
    pooled_iv_lower = float(iv_pooled["confidence_interval_lower"])
    pooled_iv_upper = float(iv_pooled["confidence_interval_upper"])
    pooled_iv_is_usable = not bool(iv_pooled["weak_instrument_flag"])

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

        if shrunk is not None and PLAUSIBLE_ELASTICITY_RANGE[0] <= shrunk[
            "shrunk_elasticity"
        ] <= PLAUSIBLE_ELASTICITY_RANGE[1]:
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
            "elasticity_for_pricing": DEFAULT_ELASTICITY,
            "elasticity_lower_for_pricing": (
                DEFAULT_ELASTICITY - DEFAULT_ELASTICITY_HALF_WIDTH
            ),
            "elasticity_upper_for_pricing": (
                DEFAULT_ELASTICITY + DEFAULT_ELASTICITY_HALF_WIDTH
            ),
            "elasticity_source": "default_fallback",
            "elasticity_is_causal": False,
            "first_stage_f_statistic": np.nan,
        }

    elasticity_lookup = {
        row["product_id"]: row
        for _, row in elasticity_estimates.iterrows()
        if pd.notna(row.get("shrunk_elasticity"))
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

    # ---------------------------------------------------------
    print("Preparing optimization dataset...")

    # Step 06 is the do-nothing demand baseline for a future price decision.
    # The earlier pipeline instead ignored `forecast_quantity` and optimized
    # from the latest historical units_sold row, so forecasting did not
    # actually feed the decision it was presented as supporting.
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
        print(
            "  WARNING: next-period forecasts were unavailable; using "
            "historical non-stockout demand"
        )

    if snapshot["baseline_quantity_raw"].isna().any():
        raise ValueError("Optimization baseline contains missing quantities")

    # Convert retrieved evidence into a bounded, transparent demand overlay.
    # The raw forecast remains in the output, so a reviewer can separate the
    # statistical baseline from the market-intelligence adjustment.
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

    if not PROFIT_CONFIGURATION.allow_non_causal_pricing:
        non_causal = ~optimization_data["elasticity_is_causal"].astype(bool)

        if non_causal.any():
            print(
                f"  {int(non_causal.sum())} item(s) have no causal "
                "elasticity and will be held at current price for review"
            )

    optimization_data["elasticity_for_pricing"] = optimization_data[
        "elasticity_for_pricing"
    ].fillna(DEFAULT_ELASTICITY)

    optimization_data["elasticity_lower_for_pricing"] = optimization_data[
        "elasticity_lower_for_pricing"
    ].fillna(DEFAULT_ELASTICITY - DEFAULT_ELASTICITY_HALF_WIDTH)

    optimization_data["elasticity_upper_for_pricing"] = optimization_data[
        "elasticity_upper_for_pricing"
    ].fillna(DEFAULT_ELASTICITY + DEFAULT_ELASTICITY_HALF_WIDTH)

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

    print(f"Optimization dataset: {len(optimization_data):,} future decision units")
    print(f"Unique products: {optimization_data['product_id'].nunique():,}")
    print(f"Categories: {optimization_data['category'].nunique()}\n")

    # ---------------------------------------------------------
    # Run price optimization
    # ---------------------------------------------------------

    print("Running price optimization...")

    # Items without a causally identified elasticity are not priced. They
    # are reported as explicit holds with a machine-readable reason, so the
    # output accounts for every item in the portfolio and a reviewer can
    # see exactly why a price was withheld rather than finding a silent gap.
    if PROFIT_CONFIGURATION.allow_non_causal_pricing:
        priceable = optimization_data
        withheld = optimization_data.iloc[0:0]
    else:
        is_causal = optimization_data["elasticity_is_causal"].astype(bool)
        priceable = optimization_data.loc[is_causal]
        withheld = optimization_data.loc[~is_causal]

    print(
        f"  Pricing {len(priceable):,} of {len(optimization_data):,} items "
        f"({len(withheld):,} withheld pending a causal elasticity)"
    )

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
        configuration=PROFIT_CONFIGURATION,
    )

    if len(withheld):
        withheld_rows = pd.DataFrame(
            {
                "item_id": withheld["item_id"].to_numpy(),
                "category": withheld["category"].to_numpy(),
                "status": "held_no_causal_elasticity",
                "error_message": (
                    "No causally identified elasticity available; "
                    "step 03's correlational estimate was not used. "
                    "Re-run step 04 or supply a stronger instrument."
                ),
                "optimization_objective": PROFIT_CONFIGURATION.objective,
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
                "elasticity": withheld[
                    "elasticity_for_pricing"
                ].to_numpy(),
                "unit_cost": withheld["unit_cost"].to_numpy(),
                "elasticity_source": withheld[
                    "elasticity_source"
                ].to_numpy(),
                "elasticity_is_causal": False,
            }
        )

        recommendations = pd.concat(
            [recommendations, withheld_rows],
            ignore_index=True,
        )

    # Attach elasticity provenance to every priced row as well, so the
    # recommendation file is self-documenting about the evidence behind it.
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

    recommendations = recommendations.drop(
        columns=[
            column
            for column in ("elasticity_source", "elasticity_is_causal")
            if column in recommendations.columns
        ]
    ).merge(provenance, on="item_id", how="left")

    print(f"Optimization complete: {len(recommendations):,} recommendations\n")

    # ---------------------------------------------------------
    # Validate recommendations
    # ---------------------------------------------------------

    print("Validating recommendations...")

    validation_data = validate_price_recommendations(
        recommendations,
        configuration=PROFIT_CONFIGURATION,
    )
    validation_summary = summarize_recommendation_validation(validation_data)

    print("Validation summary:")
    print(validation_summary.to_string(index=False))
    print()

    # ---------------------------------------------------------
    # Generate portfolio, action, and category summaries
    # ---------------------------------------------------------

    print("Generating portfolio summary...")

    portfolio_summary = summarize_optimization_portfolio(recommendations)
    action_summary = build_optimization_action_summary(recommendations)
    category_summary = build_optimization_category_summary(
        recommendations,
        category_column="category",
    )

    print("\nAction summary:")
    print(action_summary.to_string(index=False))
    print()

    # ---------------------------------------------------------
    # Run sensitivity analysis
    # ---------------------------------------------------------

    print("Running sensitivity analysis...")

    sensitivity_analysis = run_price_sensitivity_analysis(
        priceable,
        item_column="item_id",
        current_price_column="selling_price",
        baseline_quantity_column="baseline_quantity_for_pricing",
        elasticity_column="elasticity_for_pricing",
        unit_cost_column="unit_cost",
        category_column="category",
        configuration=PROFIT_CONFIGURATION,
    )

    print(f"Sensitivity scenarios: {len(sensitivity_analysis):,}\n")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving optimization results...")

    save_csv(recommendations, OUTPUT_FILES["recommendations"])
    save_csv(portfolio_summary, OUTPUT_FILES["portfolio_summary"])
    save_csv(validation_summary, OUTPUT_FILES["validation_summary"])
    save_csv(action_summary, OUTPUT_FILES["action_summary"])
    save_csv(category_summary, OUTPUT_FILES["category_summary"])
    save_csv(sensitivity_analysis, OUTPUT_FILES["sensitivity_analysis"])
    save_json(
        build_optimization_configuration(PROFIT_CONFIGURATION),
        OUTPUT_FILES["configuration"],
    )

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    portfolio_row = portfolio_summary.iloc[0]

    print("\n" + "=" * 72)
    print("PIPELINE STEP 08 COMPLETE")
    print("=" * 72)
    print(f"\nRecommendations generated: {len(recommendations):,} items")
    print(f"Successful: {portfolio_row['successful_items']:,}")
    print(f"Estimated profit lift: ${portfolio_row.get('total_profit_change', 0):,.0f}")
    print(f"Estimated revenue lift: ${portfolio_row.get('total_revenue_change', 0):,.0f}")
    print("\nPrimary output:")
    print(f"  {OUTPUT_FILES['recommendations'].name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
