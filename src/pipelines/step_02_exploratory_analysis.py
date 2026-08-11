"""
Pipeline Step 02: Exploratory Data Analysis

This pipeline:
1. Loads cleaned retail data
2. Adds engineered features
3. Generates summary statistics by category, product, store, region
4. Analyzes price-demand relationships
5. Saves EDA summaries to processed directory
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.analysis import (
    calculate_kpi_summary,
    calculate_price_demand_correlation,
    create_discount_bands,
    summarize_by_category,
    summarize_by_product,
    summarize_by_region,
    summarize_by_store,
)
from src.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RETAIL_NUMERIC_COLUMNS,
    RETAIL_REQUIRED_COLUMNS,
)
from src.data import (
    dataframe_quality_summary,
    load_csv,
    save_csv,
    validate_dataframe,
    validate_non_negative_columns,
    validate_positive_columns,
    validate_required_columns,
)
from src.features import (
    add_calendar_features,
    add_demand_features,
    add_price_features,
    add_promotion_features,
    safe_divide,
)
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    """Execute the exploratory analysis pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 02 — EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    INPUT_FILE = INTERIM_DIR / "retail_demand_clean.csv"

    OUTPUT_FILES = {
        "kpi_summary": PROCESSED_DIR / "eda_kpi_summary.csv",
        "numeric_summary": PROCESSED_DIR / "eda_numeric_summary.csv",
        "category_summary": PROCESSED_DIR / "eda_category_summary.csv",
        "product_summary": PROCESSED_DIR / "eda_product_summary.csv",
        "store_summary": PROCESSED_DIR / "eda_store_summary.csv",
        "region_summary": PROCESSED_DIR / "eda_region_summary.csv",
        "discount_summary": PROCESSED_DIR / "eda_discount_summary.csv",
        "price_demand_correlation": PROCESSED_DIR / "eda_price_demand_correlation.csv",
    }

    # ---------------------------------------------------------
    # Check dependencies
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"retail_clean": INPUT_FILE},
        pipeline_name="step_02_exploratory_analysis",
    )

    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading cleaned retail data...")
    retail_data = load_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(retail_data, dataframe_name="retail data")
    print(f"Loaded {len(retail_data):,} records")
    print(f"Columns: {retail_data.shape[1]}")
    print(
        f"Date range: {retail_data['date'].min().date()} to {retail_data['date'].max().date()}\n"
    )

    # ---------------------------------------------------------
    # Validate schema
    # ---------------------------------------------------------

    validate_required_columns(
        retail_data,
        RETAIL_REQUIRED_COLUMNS,
        dataframe_name="retail data",
    )

    validate_positive_columns(
        retail_data,
        ["regular_price", "selling_price", "competitor_price", "unit_cost"],
        allow_missing=False,
        dataframe_name="retail data",
    )

    validate_non_negative_columns(
        retail_data,
        ["discount_pct", "inventory_level", "units_sold", "revenue"],
        allow_missing=False,
        dataframe_name="retail data",
    )

    print("Schema validation passed\n")

    # ---------------------------------------------------------
    # Add engineered features
    # ---------------------------------------------------------

    print("Engineering features...")

    eda_data = add_calendar_features(retail_data, date_column="date")

    eda_data = add_price_features(
        eda_data,
        price_column="selling_price",
        reference_price_column="regular_price",
        cost_column="unit_cost",
    )

    eda_data = add_demand_features(
        eda_data,
        quantity_column="units_sold",
        price_column="selling_price",
        cost_column="unit_cost",
    )

    eda_data = add_promotion_features(
        eda_data,
        promotion_column="promotion_flag",
    )

    # Add derived features
    eda_data["competitor_price_gap"] = (
        eda_data["selling_price"] - eda_data["competitor_price"]
    )

    eda_data["competitor_price_gap_rate"] = safe_divide(
        eda_data["competitor_price_gap"],
        eda_data["competitor_price"],
    )

    eda_data["inventory_utilization"] = safe_divide(
        eda_data["units_sold"],
        eda_data["inventory_level"],
    )

    print(f"Engineered features: {eda_data.shape[1]} columns\n")

    # ---------------------------------------------------------
    # Generate summary statistics
    # ---------------------------------------------------------

    print("Generating summary statistics...")

    # Portfolio KPIs
    kpi_summary = calculate_kpi_summary(eda_data)

    # Numeric distributions
    available_numeric_columns = [
        col for col in RETAIL_NUMERIC_COLUMNS if col in eda_data.columns
    ]

    numeric_summary = (
        eda_data[available_numeric_columns]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .T.reset_index(names="variable")
    )

    # Category-level aggregation
    category_summary = summarize_by_category(eda_data)

    # Product-level aggregation (top 20)
    product_summary = summarize_by_product(eda_data, top_n=20)

    # Store-level aggregation
    store_summary = summarize_by_store(eda_data)

    # Region-level aggregation
    region_summary = summarize_by_region(eda_data)

    # Discount band analysis
    discount_summary = create_discount_bands(
        eda_data,
        discount_column="discount_rate",
    )

    # Price-demand correlation by category
    price_demand_correlation = calculate_price_demand_correlation(
        eda_data,
        group_by="category",
    )

    print(f"  - Portfolio KPIs: {len(kpi_summary)} metrics")
    print(f"  - Category summary: {len(category_summary)} categories")
    print(f"  - Product summary: {len(product_summary)} products")
    print(f"  - Store summary: {len(store_summary)} stores")
    print(f"  - Region summary: {len(region_summary)} regions")
    print(f"  - Price-demand correlations: {len(price_demand_correlation)} categories\n")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving EDA summaries...")

    save_csv(kpi_summary, OUTPUT_FILES["kpi_summary"])
    save_csv(numeric_summary, OUTPUT_FILES["numeric_summary"])
    save_csv(category_summary, OUTPUT_FILES["category_summary"])
    save_csv(product_summary, OUTPUT_FILES["product_summary"])
    save_csv(store_summary, OUTPUT_FILES["store_summary"])
    save_csv(region_summary, OUTPUT_FILES["region_summary"])
    save_csv(discount_summary, OUTPUT_FILES["discount_summary"])
    save_csv(price_demand_correlation, OUTPUT_FILES["price_demand_correlation"])

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 02 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
