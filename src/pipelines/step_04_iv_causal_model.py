"""
Pipeline Step 04: IV/2SLS Causal Price Model

This pipeline:
1. Loads cleaned retail data
2. Prepares data with log transformations
3. Fits IV 2SLS model with instruments for endogenous price
4. Compares OLS baseline with IV estimates
5. Saves causal effect estimates and diagnostics
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.data import load_csv, save_csv, save_json, validate_dataframe
from src.models.causal import (
    fit_group_iv_elasticities,
    compare_ols_and_iv,
    extract_causal_effect,
    fit_iv_2sls,
    summarize_iv_model,
)
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    """Execute the IV/2SLS causal model pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 04 — IV/2SLS CAUSAL PRICE MODEL")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define configuration
    # ---------------------------------------------------------

    SIGNIFICANCE_LEVEL = 0.05
    CONFIDENCE_LEVEL = 0.95
    WEAK_INSTRUMENT_THRESHOLD = 10.0

    OUTCOME_COLUMN = "log_units_sold"
    ENDOGENOUS_COLUMN = "log_selling_price"

    INSTRUMENT_COLUMNS = [
        "log_supplier_cost_index",
        "log_shipping_cost_index",
    ]

    EXOGENOUS_COLUMNS = [
        "promotion_flag",
        "holiday_flag",
        "weekend_flag",
        "weather_index",
        "log_customer_traffic",
        "log_advertising_spend",
    ]

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    INPUT_FILE = INTERIM_DIR / "retail_demand_clean.csv"

    OUTPUT_FILES = {
        "model_comparison": PROCESSED_DIR / "iv_2sls_model_comparison.csv",
        "first_stage_diagnostics": PROCESSED_DIR / "iv_first_stage_diagnostics.csv",
        "specification_diagnostics": PROCESSED_DIR / "iv_specification_diagnostics.csv",
        "product_elasticities": PROCESSED_DIR / "iv_product_elasticity_estimates.csv",
        "pooled_elasticity": PROCESSED_DIR / "iv_pooled_elasticity.json",
    }

    # ---------------------------------------------------------
    # Check dependencies
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"retail_clean": INPUT_FILE},
        pipeline_name="step_04_iv_causal_model",
    )

    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading retail data...")
    retail_data = load_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(retail_data, dataframe_name="retail data")
    print(f"Loaded {len(retail_data):,} records\n")

    # ---------------------------------------------------------
    # Prepare modeling dataset with log transformations
    # ---------------------------------------------------------

    print("Preparing IV modeling dataset...")

    # Create log transformations
    modeling_data = retail_data.copy()

    # Add small constant to avoid log(0)
    modeling_data["log_units_sold"] = np.log(
        modeling_data["units_sold"] + 1
    )
    modeling_data["log_selling_price"] = np.log(
        modeling_data["selling_price"]
    )

    # Instruments are evidence, not missing-value imputations.  Earlier
    # versions fabricated cost instruments from unit cost/regular price when
    # the source columns were absent.  That can mechanically create a strong
    # first stage while providing no defensible exclusion restriction.
    required_instrument_sources = [
        "supplier_cost_index",
        "shipping_cost_index",
    ]
    missing_instrument_sources = [
        column
        for column in required_instrument_sources
        if column not in modeling_data.columns
    ]

    if missing_instrument_sources:
        raise ValueError(
            "Causal elasticity cannot be estimated because declared "
            "instrument sources are missing: "
            f"{missing_instrument_sources}. Supply observed, externally "
            "justified instruments; they will not be synthesized."
        )

    nonpositive_instrument_sources = [
        column
        for column in required_instrument_sources
        if (pd.to_numeric(modeling_data[column], errors="coerce") <= 0).any()
    ]

    if nonpositive_instrument_sources:
        raise ValueError(
            "Instrument sources must be strictly positive before log "
            f"transformation: {nonpositive_instrument_sources}"
        )

    modeling_data["log_supplier_cost_index"] = np.log(
        modeling_data["supplier_cost_index"]
    )
    modeling_data["log_shipping_cost_index"] = np.log(
        modeling_data["shipping_cost_index"]
    )

    # Create log of other variables
    modeling_data["log_customer_traffic"] = np.log(
        modeling_data.get("customer_traffic", 100) + 1
    )
    modeling_data["log_advertising_spend"] = np.log(
        modeling_data.get("advertising_spend", 10) + 1
    )

    # Filter to complete cases
    all_model_columns = (
        [OUTCOME_COLUMN, ENDOGENOUS_COLUMN]
        + INSTRUMENT_COLUMNS
        + EXOGENOUS_COLUMNS
    )

    modeling_data = modeling_data.dropna(subset=all_model_columns)

    print(f"Prepared {len(modeling_data):,} complete observations")
    print(f"Outcome: {OUTCOME_COLUMN}")
    print(f"Endogenous: {ENDOGENOUS_COLUMN}")
    print(f"Instruments: {', '.join(INSTRUMENT_COLUMNS)}")
    print(f"Exogenous controls: {len(EXOGENOUS_COLUMNS)}\n")

    # ---------------------------------------------------------
    # Fit IV 2SLS model
    # ---------------------------------------------------------

    print("Fitting IV 2SLS model...")

    iv_model_bundle = fit_iv_2sls(
        modeling_data,
        outcome_column=OUTCOME_COLUMN,
        endogenous_column=ENDOGENOUS_COLUMN,
        instrument_columns=INSTRUMENT_COLUMNS,
        exogenous_columns=EXOGENOUS_COLUMNS,
    )

    print("IV 2SLS model fitted successfully")

    # Extract causal effect and full statistical summary
    iv_price_effect = extract_causal_effect(iv_model_bundle)
    iv_summary = summarize_iv_model(iv_model_bundle)

    print(f"\nIV Estimate of Price Elasticity: {iv_price_effect:.4f}")
    print(f"Standard Error: {iv_summary['standard_error']:.4f}")
    print(
        f"95% CI: [{iv_summary['confidence_interval_lower']:.4f}, "
        f"{iv_summary['confidence_interval_upper']:.4f}]"
    )
    print(f"P-value: {iv_summary['p_value']:.4f}\n")

    # ---------------------------------------------------------
    # Compare OLS vs IV
    # ---------------------------------------------------------

    print("Comparing OLS baseline with IV estimates...")

    ols_iv_comparison = compare_ols_and_iv(iv_model_bundle)

    print("\nOLS vs IV Comparison:")
    print(
        ols_iv_comparison[
            ["model", "estimate", "standard_error", "p_value", "observations"]
        ].to_string(index=False)
    )
    print()

    # ---------------------------------------------------------
    # First-stage / weak-instrument diagnostics
    # ---------------------------------------------------------

    first_stage_diagnostics_data = iv_model_bundle.first_stage_diagnostics

    print("IV Model Diagnostics:")
    print(f"  R-squared: {iv_summary['r_squared']:.4f}")
    print(f"  Observations: {iv_summary['observations']}")
    print(f"  First-stage F-statistic: {first_stage_diagnostics_data['f_statistic']:.4f}")
    print(f"  Weak instrument flag: {first_stage_diagnostics_data['weak_instrument_flag']}\n")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving IV model results...")

    save_csv(ols_iv_comparison, OUTPUT_FILES["model_comparison"])

    # First-stage diagnostics: the joint relevance test across every
    # instrument (per-instrument F-stats aren't computed by the model,
    # only the joint test used to flag weak instruments).
    first_stage_diagnostics = pd.DataFrame([
        {
            "instruments": ", ".join(INSTRUMENT_COLUMNS),
            "instrument_count": first_stage_diagnostics_data["instrument_count"],
            "observations": first_stage_diagnostics_data["observations"],
            "first_stage_r_squared": first_stage_diagnostics_data["first_stage_r_squared"],
            "first_stage_adjusted_r_squared": first_stage_diagnostics_data[
                "first_stage_adjusted_r_squared"
            ],
            "partial_r_squared": first_stage_diagnostics_data["partial_r_squared"],
            "f_statistic": first_stage_diagnostics_data["f_statistic"],
            "f_p_value": first_stage_diagnostics_data["f_p_value"],
            "weak_instrument_flag": first_stage_diagnostics_data["weak_instrument_flag"],
        }
    ])
    save_csv(first_stage_diagnostics, OUTPUT_FILES["first_stage_diagnostics"])

    # Specification diagnostics: only the weak-instrument F-test is
    # actually computed by src.models.causal. A formal endogeneity test
    # (e.g. Wu-Hausman) is not implemented there, so we report that
    # honestly instead of fabricating a statistic for it.
    specification_diagnostics = pd.DataFrame([
        {
            "test": "Weak Instruments (joint F-test)",
            "statistic": first_stage_diagnostics_data["f_statistic"],
            "threshold": WEAK_INSTRUMENT_THRESHOLD,
            "passes_threshold": (
                first_stage_diagnostics_data["f_statistic"] >= WEAK_INSTRUMENT_THRESHOLD
            ),
        },
        {
            "test": "Endogeneity (Wu-Hausman)",
            "statistic": np.nan,
            "threshold": np.nan,
            "passes_threshold": None,
        },
    ])
    save_csv(specification_diagnostics, OUTPUT_FILES["specification_diagnostics"])

    # ---------------------------------------------------------
    # Per-product causal elasticities
    # ---------------------------------------------------------
    #
    # The pooled estimate above establishes that price is endogenous and
    # by how much. Pricing decisions, however, are made per product, so
    # step 08 needs a causal elasticity per product rather than one
    # portfolio-wide number. The same instruments are applied within each
    # product, with store fixed effects, and every estimate carries its
    # own first-stage F and plausibility verdict so that step 08 can
    # decline to act on the weak ones.

    print("Fitting per-product IV elasticities...")

    product_iv_estimates = fit_group_iv_elasticities(
        modeling_data,
        group_columns=["product_id"],
        outcome_column=OUTCOME_COLUMN,
        endogenous_column=ENDOGENOUS_COLUMN,
        instrument_columns=INSTRUMENT_COLUMNS,
        exogenous_columns=EXOGENOUS_COLUMNS,
        categorical_columns=["store_id"],
        minimum_observations=60,
        weak_instrument_threshold=WEAK_INSTRUMENT_THRESHOLD,
        confidence_level=CONFIDENCE_LEVEL,
    )

    reliable_count = int(product_iv_estimates["is_reliable"].sum())

    print(
        f"  {reliable_count} of {len(product_iv_estimates)} products have a "
        "reliable causal elasticity "
        "(strong instrument, economically plausible)"
    )

    if reliable_count:
        reliable = product_iv_estimates.loc[
            product_iv_estimates["is_reliable"]
        ]
        print(
            "  Elasticity range: "
            f"{reliable['iv_elasticity'].min():.2f} to "
            f"{reliable['iv_elasticity'].max():.2f}"
            f"  |  first-stage F: {reliable['first_stage_f_statistic'].min():.0f} "
            f"to {reliable['first_stage_f_statistic'].max():.0f}"
        )

    unreliable = product_iv_estimates.loc[
        ~product_iv_estimates["is_reliable"]
    ]

    for _, row in unreliable.iterrows():
        print(
            f"  ! {row['product_id']}: not reliable "
            f"({row['failure_reason'] or row['status']})"
        )

    save_csv(product_iv_estimates, OUTPUT_FILES["product_elasticities"])

    # The pooled estimate is persisted as the portfolio-level fallback for
    # any product whose own causal estimate is unusable.
    save_json(
        {
            "pooled_iv_elasticity": float(iv_price_effect),
            "pooled_ols_elasticity": float(
                ols_iv_comparison.loc[
                    ols_iv_comparison["model"] == "OLS",
                    "estimate",
                ].iloc[0]
            ),
            "standard_error": float(iv_summary["standard_error"]),
            "confidence_interval_lower": float(
                iv_summary["confidence_interval_lower"]
            ),
            "confidence_interval_upper": float(
                iv_summary["confidence_interval_upper"]
            ),
            "first_stage_f_statistic": float(
                first_stage_diagnostics_data["f_statistic"]
            ),
            "weak_instrument_flag": bool(
                first_stage_diagnostics_data["weak_instrument_flag"]
            ),
            "observations": int(iv_summary["observations"]),
            "instruments": list(INSTRUMENT_COLUMNS),
        },
        OUTPUT_FILES["pooled_elasticity"],
    )

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 04 COMPLETE")
    print("=" * 72)
    print("\nKey Finding:")
    print(f"  IV Price Elasticity: {iv_price_effect:.4f}")
    print(f"  (Accounts for endogeneity of pricing decisions)")
    print("=" * 72)


if __name__ == "__main__":
    main()
