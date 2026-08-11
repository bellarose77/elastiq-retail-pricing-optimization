"""
Pipeline Step 01: Data Understanding and Validation

This pipeline:
1. Loads or generates synthetic retail and market notes datasets
2. Validates data quality and schema
3. Performs consistency audits
4. Saves cleaned datasets and validation reports
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RANDOM_STATE,
    RETAIL_REQUIRED_COLUMNS,
)
from src.data import (
    build_consistency_audit,
    build_file_registry,
    convert_datetime_columns,
    convert_numeric_columns,
    dataframe_quality_summary,
    generate_dataset_overview,
    load_csv,
    save_csv,
    save_json,
    standardize_column_names,
    summarize_missing_values,
    validate_dataframe,
    validate_non_negative_columns,
    validate_positive_columns,
    validate_required_columns,
    validate_unique_columns,
)
from src.features import add_calendar_features

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    """Execute the data validation pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 01 — DATA UNDERSTANDING AND VALIDATION")
    print("=" * 72)
    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Raw data    : {RAW_DIR}")
    print(f"Interim data: {INTERIM_DIR}\n")

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    SYNTHETIC_DIR = RAW_DIR / "synthetic"
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    RETAIL_RAW_FILE = SYNTHETIC_DIR / "sample_retail_pricing.csv"
    MARKET_NOTES_RAW_FILE = SYNTHETIC_DIR / "market_notes.csv"

    RETAIL_CLEAN_FILE = INTERIM_DIR / "retail_demand_clean.csv"
    MARKET_NOTES_CLEAN_FILE = INTERIM_DIR / "market_notes_clean.csv"
    DATASET_OVERVIEW_FILE = INTERIM_DIR / "dataset_overview.csv"
    QUALITY_SUMMARY_FILE = INTERIM_DIR / "data_quality_summary.csv"
    CONSISTENCY_AUDIT_FILE = INTERIM_DIR / "retail_consistency_audit.csv"
    MANIFEST_FILE = INTERIM_DIR / "notebook_01_manifest.json"

    # ---------------------------------------------------------
    # Check for source files and generate if needed
    # ---------------------------------------------------------

    file_registry = build_file_registry(
        {
            "Synthetic retail pricing": (RETAIL_RAW_FILE, True),
            "Synthetic market notes": (MARKET_NOTES_RAW_FILE, True),
        },
        PROJECT_ROOT,
    )

    print("File registry:")
    print(file_registry.to_string(index=False))
    print()

    if not RETAIL_RAW_FILE.exists() or not MARKET_NOTES_RAW_FILE.exists():
        from src.data.generate_synthetic import (
            generate_market_notes,
            generate_retail_data,
        )

        print("Generating synthetic datasets...\n")

        retail_raw = generate_retail_data(
            num_days=365,
            num_stores=10,
            num_products=50,
            random_state=RANDOM_STATE,
        )

        market_notes_raw = generate_market_notes(
            num_notes=200,
            random_state=RANDOM_STATE,
        )

        save_csv(retail_raw, RETAIL_RAW_FILE)
        save_csv(market_notes_raw, MARKET_NOTES_RAW_FILE)

        print(f"Generated {len(retail_raw):,} retail records")
        print(f"Generated {len(market_notes_raw):,} market notes\n")

    # ---------------------------------------------------------
    # Load and standardize retail data
    # ---------------------------------------------------------

    print("Loading retail data...")
    retail_raw = load_csv(RETAIL_RAW_FILE, low_memory=False)

    retail = standardize_column_names(retail_raw)
    retail = convert_datetime_columns(retail, ["date"])
    retail = convert_numeric_columns(
        retail,
        [
            "regular_price",
            "selling_price",
            "competitor_price",
            "unit_cost",
            "discount_pct",
            "inventory_level",
            "units_sold",
            "revenue",
            "gross_profit",
        ],
    )

    validate_dataframe(retail, dataframe_name="retail data")
    validate_required_columns(
        retail,
        RETAIL_REQUIRED_COLUMNS,
        dataframe_name="retail data",
    )

    print(f"Loaded {len(retail):,} retail records")
    print(f"Columns: {retail.shape[1]}")
    print(f"Date range: {retail['date'].min().date()} to {retail['date'].max().date()}\n")

    # ---------------------------------------------------------
    # Validate numeric columns
    # ---------------------------------------------------------

    validate_positive_columns(
        retail,
        ["regular_price", "selling_price", "competitor_price", "unit_cost"],
        allow_missing=False,
        dataframe_name="retail data",
    )

    validate_non_negative_columns(
        retail,
        ["discount_pct", "inventory_level", "units_sold", "revenue"],
        allow_missing=False,
        dataframe_name="retail data",
    )

    print("Numeric validation passed")

    # ---------------------------------------------------------
    # Add calendar features for consistency checks
    # ---------------------------------------------------------

    retail = add_calendar_features(retail, date_column="date")

    calendar_reference = retail[["date", "year", "month", "day_of_week"]].copy()
    calendar_reference["is_weekend"] = retail["day_of_week"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)
    calendar_reference["day_name"] = retail["day_of_week"]

    # ---------------------------------------------------------
    # Build consistency audit
    # ---------------------------------------------------------

    calculated_discount_pct = 1 - (retail["selling_price"] / retail["regular_price"])
    calculated_revenue = retail["selling_price"] * retail["units_sold"]
    calculated_gross_profit = calculated_revenue - (
        retail["unit_cost"] * retail["units_sold"]
    )

    consistency_checks = [
        {
            "name": "Missing dates",
            "condition": retail["date"].isna(),
            "severity": "critical",
        },
        {
            "name": "Selling price above regular price",
            "condition": retail["selling_price"] > retail["regular_price"] + 0.01,
            "severity": "review",
        },
        {
            "name": "Units sold above inventory",
            "condition": retail["units_sold"] > retail["inventory_level"],
            "severity": "critical",
        },
        {
            "name": "Unit cost above selling price",
            "condition": retail["unit_cost"] > retail["selling_price"],
            "severity": "review",
        },
        {
            "name": "Discount differs by more than 2 pp",
            "condition": (retail["discount_pct"] - calculated_discount_pct).abs() > 0.02,
            "severity": "review",
        },
        {
            "name": "Revenue differs by more than $1",
            "condition": (retail["revenue"] - calculated_revenue).abs() > 1.00,
            "severity": "review",
        },
        {
            "name": "Gross profit differs by more than $1.50",
            "condition": (retail["gross_profit"] - calculated_gross_profit).abs() > 1.50,
            "severity": "review",
        },
    ]

    retail_consistency_audit = build_consistency_audit(retail, consistency_checks)

    print("\nConsistency audit:")
    print(retail_consistency_audit.to_string(index=False))

    critical_issues = retail_consistency_audit[
        retail_consistency_audit["severity"] == "critical"
    ]
    critical_issue_count = int(critical_issues["issue_count"].sum())

    if critical_issue_count > 0:
        raise ValueError(
            f"Critical retail consistency issues found: {critical_issue_count:,}"
        )

    print("\nNo critical consistency issues found")

    # ---------------------------------------------------------
    # Load and process market notes
    # ---------------------------------------------------------

    print("\nLoading market notes...")
    market_notes_raw = load_csv(MARKET_NOTES_RAW_FILE, low_memory=False)

    market_notes = standardize_column_names(market_notes_raw)
    market_notes = convert_datetime_columns(market_notes, ["date"])

    print(f"Loaded {len(market_notes):,} market notes\n")

    # ---------------------------------------------------------
    # Generate data quality summaries
    # ---------------------------------------------------------

    dataset_overview = generate_dataset_overview(
        {
            "retail": retail,
            "market_notes": market_notes,
        }
    )

    quality_summary = pd.concat(
        [
            summarize_missing_values(retail, "retail"),
            summarize_missing_values(market_notes, "market_notes"),
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving outputs...")

    save_csv(retail, RETAIL_CLEAN_FILE)
    save_csv(market_notes, MARKET_NOTES_CLEAN_FILE)
    save_csv(dataset_overview, DATASET_OVERVIEW_FILE)
    save_csv(quality_summary, QUALITY_SUMMARY_FILE)
    save_csv(retail_consistency_audit, CONSISTENCY_AUDIT_FILE)

    manifest = {
        "pipeline_step": "01_data_validation",
        "retail_clean_file": str(RETAIL_CLEAN_FILE.relative_to(PROJECT_ROOT)),
        "market_notes_clean_file": str(MARKET_NOTES_CLEAN_FILE.relative_to(PROJECT_ROOT)),
        "retail_rows": int(len(retail)),
        "retail_columns": int(retail.shape[1]),
        "retail_date_start": str(retail["date"].min().date()),
        "retail_date_end": str(retail["date"].max().date()),
        "market_notes_rows": int(len(market_notes)),
    }

    save_json(manifest, MANIFEST_FILE)

    print(f"\nOutputs saved to {INTERIM_DIR}")
    print(f"  - {RETAIL_CLEAN_FILE.name}")
    print(f"  - {MARKET_NOTES_CLEAN_FILE.name}")
    print(f"  - {DATASET_OVERVIEW_FILE.name}")
    print(f"  - {QUALITY_SUMMARY_FILE.name}")
    print(f"  - {CONSISTENCY_AUDIT_FILE.name}")
    print(f"  - {MANIFEST_FILE.name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 01 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
