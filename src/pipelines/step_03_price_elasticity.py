"""
Pipeline Step 03: Price Elasticity Estimation

This pipeline:
1. Loads cleaned retail data
2. Prepares data for elasticity estimation
3. Fits log-log elasticity models by product group
4. Estimates price elasticities and classifies products
5. Simulates price scenarios
6. Saves elasticity estimates and scenario results
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.config import (
    ELASTICITY_COLUMN_ALIASES,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.data import load_csv, save_csv, validate_dataframe
from src.features import add_calendar_features, add_promotion_features
from src.models import (
    fit_group_elasticities,
    prepare_elasticity_data,
    shrink_product_elasticities,
    simulate_price_scenarios,
)
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    """Execute the price elasticity pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 03 — PRICE ELASTICITY ESTIMATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    INPUT_FILE = INTERIM_DIR / "retail_demand_clean.csv"

    OUTPUT_FILES = {
        "elasticities": PROCESSED_DIR / "elasticity_estimates.csv",
        "product_elasticities": PROCESSED_DIR / "product_elasticity_estimates.csv",
        "scenarios": PROCESSED_DIR / "elasticity_scenario_simulation.csv",
    }

    # ---------------------------------------------------------
    # Check dependencies
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"retail_clean": INPUT_FILE},
        pipeline_name="step_03_price_elasticity",
    )

    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load and prepare data
    # ---------------------------------------------------------

    print("Loading retail data...")
    retail_data = load_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(retail_data, dataframe_name="retail data")
    print(f"Loaded {len(retail_data):,} records\n")

    # Add features
    print("Preparing elasticity features...")
    retail_df = add_calendar_features(retail_data, date_column="date")
    retail_df = add_promotion_features(retail_df, promotion_column="promotion_flag")

    # Apply column aliases for elasticity model
    rename_mapping = {
        source: target
        for source, target in ELASTICITY_COLUMN_ALIASES.items()
        if source in retail_df.columns and target not in retail_df.columns
    }

    if rename_mapping:
        retail_df = retail_df.rename(columns=rename_mapping)
        print(f"Applied {len(rename_mapping)} column aliases")

    # Prepare elasticity dataset
    elasticity_data = prepare_elasticity_data(
        retail_df,
        price_column="price",
        quantity_column="quantity",
        control_columns=["promotion", "weekend", "holiday"],
    )

    print(f"Prepared {len(elasticity_data):,} observations for elasticity estimation\n")

    # ---------------------------------------------------------
    # Fit elasticity models
    # ---------------------------------------------------------

    print("Fitting price elasticity models by category...")

    elasticity_results = fit_group_elasticities(
        retail_df,
        price_column="price",
        quantity_column="quantity",
        group_columns=["category"],
        control_columns=["promotion", "weekend", "holiday"],
    )

    print(f"Estimated elasticities for {len(elasticity_results)} categories")
    print("\nElasticity summary by category:")
    print(
        elasticity_results[
            ["category", "status", "elasticity", "elasticity_class", "observations"]
        ].to_string(index=False)
    )
    print()

    # ---------------------------------------------------------
    # Fit per-product elasticity models with reliability flags
    # ---------------------------------------------------------

    # step_08's uncertainty-aware pricing needs per-product elasticity
    # with a confidence interval, not just the category-level estimate
    # above. This used to come from a one-off notebook artifact that no
    # pipeline step reproduced; fitting it here closes that gap so the
    # whole chain is self-sufficient from raw data.
    print("Fitting price elasticity models by product...")

    product_elasticity_results = fit_group_elasticities(
        retail_df,
        price_column="price",
        quantity_column="quantity",
        group_columns=["product_id", "category"],
        control_columns=["promotion", "weekend", "holiday"],
    )

    product_elasticity_results["is_statistically_significant"] = (
        product_elasticity_results["status"] == "success"
    ) & (product_elasticity_results["p_value"] < 0.05)

    product_elasticity_results["is_economically_plausible"] = (
        product_elasticity_results["elasticity"] < 0
    )

    product_elasticity_results["is_reliable"] = (
        product_elasticity_results["is_statistically_significant"]
        & product_elasticity_results["is_economically_plausible"]
    )

    reliable_count = int(product_elasticity_results["is_reliable"].sum())

    print(f"Estimated elasticities for {len(product_elasticity_results)} products")
    print(
        f"{reliable_count} of {len(product_elasticity_results)} are statistically "
        "significant and negative-signed on their own (reliable)\n"
    )

    # ---------------------------------------------------------
    # Shrink noisy per-product estimates toward their category
    # ---------------------------------------------------------

    # Empirical-Bayes partial pooling: instead of an all-or-nothing "trust
    # the product estimate or fall back to a flat default" rule, blend
    # each product's own estimate with its category's elasticity, weighted
    # by how much the product's own data can be trusted relative to how
    # much products in that category naturally vary. See
    # shrink_product_elasticities's docstring for the statistics.
    print("Shrinking product elasticities toward their category baseline...")

    product_elasticity_results = shrink_product_elasticities(
        product_elasticity_results,
        elasticity_results,
        category_column="category",
    ).rename(columns={"elasticity": "price_elasticity"})

    average_shrinkage_weight = float(
        product_elasticity_results["shrinkage_weight"].mean()
    )

    print(
        f"Average shrinkage weight on the product's own estimate: "
        f"{average_shrinkage_weight:.2f} (1.0 = fully trust the product's "
        "own regression, 0.0 = fully use the category baseline)\n"
    )

    # ---------------------------------------------------------
    # Simulate price scenarios
    # ---------------------------------------------------------

    print("Simulating price change scenarios...")

    # Define price change scenarios
    price_changes = [-0.10, -0.05, 0.00, 0.05, 0.10]  # -10% to +10%

    scenario_results = []

    successful_categories = elasticity_results.loc[
        elasticity_results["status"] == "success"
    ]

    for category_data in successful_categories.to_dict("records"):
        category = category_data["category"]
        elasticity = category_data["elasticity"]

        category_scenarios = simulate_price_scenarios(
            base_price=10.0,  # Normalized baseline
            base_quantity=100.0,  # Normalized baseline
            elasticity=elasticity,
            price_change_rates=price_changes,
        )

        category_scenarios["category"] = category
        category_scenarios["elasticity"] = elasticity
        scenario_results.append(category_scenarios)

    scenario_simulation = pd.concat(scenario_results, ignore_index=True)

    print(f"Simulated {len(scenario_simulation)} price scenarios\n")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving elasticity estimates and scenarios...")

    save_csv(elasticity_results, OUTPUT_FILES["elasticities"])
    save_csv(product_elasticity_results, OUTPUT_FILES["product_elasticities"])
    save_csv(scenario_simulation, OUTPUT_FILES["scenarios"])

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 03 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
