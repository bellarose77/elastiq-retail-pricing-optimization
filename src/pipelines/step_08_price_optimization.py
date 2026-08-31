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

import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data import load_csv, save_csv, save_json
from src.optimization.dataset import (
    attach_elasticity_provenance,
    build_optimization_dataset,
    build_withheld_recommendations,
    default_profit_configuration,
    split_priceable_items,
)
from src.optimization.pricing import (
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
    # `default_profit_configuration` is shared with the online pricing API
    # (app/api) so both surfaces enforce the same guardrails -- causal
    # elasticity only, -10%/+20% change band, 15% minimum margin, 15%
    # competitor band. See its docstring and
    # `PricingOptimizationConfig.cannibalization_max_iterations` for why
    # cross-item cannibalization stays disabled here.

    print("Configuring optimization constraints...")

    PROFIT_CONFIGURATION = default_profit_configuration()

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
    # The precedence chain itself (product IV -> pooled IV -> shrunk OLS ->
    # flat default) lives in `build_optimization_dataset`, shared with the
    # online pricing API, so both surfaces price on identical evidence.
    # `elasticity_is_causal` travels with each row so the optimizer can
    # refuse to reprice on non-causal evidence.

    optimization_data = build_optimization_dataset(
        retail_rag,
        elasticity_estimates,
        promotion_uplift,
        iv_product_elasticity,
        iv_pooled,
    )

    if not PROFIT_CONFIGURATION.allow_non_causal_pricing:
        non_causal = ~optimization_data["elasticity_is_causal"].astype(bool)

        if non_causal.any():
            print(
                f"  {int(non_causal.sum())} item(s) have no causal "
                "elasticity and will be held at current price for review"
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
    priceable, withheld = split_priceable_items(
        optimization_data,
        PROFIT_CONFIGURATION,
    )

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
        withheld_rows = build_withheld_recommendations(
            withheld,
            objective=PROFIT_CONFIGURATION.objective,
        )

        recommendations = pd.concat(
            [recommendations, withheld_rows],
            ignore_index=True,
        )

    # Attach elasticity provenance to every priced row as well, so the
    # recommendation file is self-documenting about the evidence behind it.
    recommendations = attach_elasticity_provenance(
        recommendations,
        optimization_data,
    )

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
