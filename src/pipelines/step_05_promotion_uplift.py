"""
Pipeline Step 05: Promotion Uplift Modeling

This pipeline:
1. Loads cleaned retail data
2. Prepares promotion data with propensity scores
3. Fits T-learner uplift model
4. Predicts individual promotion uplift
5. Ranks promotion opportunities
6. Evaluates model performance by segment
7. Saves predictions and opportunity rankings
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config import (
    INTERIM_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
)
from src.data import load_csv, save_csv, save_model, validate_dataframe
from src.features import add_calendar_features, safe_divide
from src.models import (
    PreparedPromotionData,
    calculate_propensity_scores,
    estimate_ipw_promotion_effect,
    evaluate_uplift_by_decile,
    fit_promotion_t_learner,
    fit_propensity_model,
    normalize_promotion_indicator,
    predict_promotion_uplift,
    prepare_promotion_data,
    rank_promotion_opportunities,
    summarize_promotion_predictions,
    trim_common_support,
)
from src.models.evaluation import summarize_segment_accuracy
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)

# Identifier columns needed downstream (segment aggregation, temporal
# split) that prepare_promotion_data does not retain, since it only
# keeps the outcome/treatment/feature columns. Reattaching them by
# position is only safe when preparation drops zero rows, which the
# guard below verifies rather than assuming.
IDENTIFIER_COLUMNS = ["product_id", "store_id", "date"]


def main() -> None:
    """Execute the promotion uplift modeling pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 05 — PROMOTION UPLIFT MODELING")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define configuration
    # ---------------------------------------------------------

    OUTCOME_COLUMN = "units_sold"
    TREATMENT_COLUMN = "promotion_flag"
    TOP_N_OPPORTUNITIES = 100

    # Only variables known before the promotion decision belong here.
    # selling_price and discount_pct are treatment definitions; inventory
    # and stockout are affected by realised demand; competitor_price is also
    # generated after promotion in the sample data.  Using any of them would
    # condition on a mediator or outcome and bias the estimated uplift.
    FEATURE_COLUMNS = [
        "regular_price",
        "unit_cost",
        "supplier_cost_index",
        "shipping_cost_index",
        "weekend_flag",
        "holiday_flag",
        "weather_index",
        "customer_traffic",
    ]

    CATEGORICAL_COLUMNS = ["product_id", "store_id"]
    MAX_ACCEPTABLE_STOCKOUT_RATE = 0.05

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    INPUT_FILE = INTERIM_DIR / "retail_demand_clean.csv"

    OUTPUT_FILES = {
        "predictions": PROCESSED_DIR / "promotion_uplift_predictions.csv",
        "by_product": PROCESSED_DIR / "promotion_uplift_by_product.csv",
        "by_store": PROCESSED_DIR / "promotion_uplift_by_store.csv",
        "deciles": PROCESSED_DIR / "promotion_uplift_deciles.csv",
        "metrics": PROCESSED_DIR / "promotion_uplift_model_metrics.csv",
        "top_opportunities": PROCESSED_DIR / "promotion_uplift_top_opportunities.csv",
        "propensity_diagnostics": PROCESSED_DIR / "promotion_propensity_diagnostics.csv",
    }

    MODEL_FILES = {
        "uplift_model": MODEL_DIR / "promotion_uplift_model.pkl",
    }

    # ---------------------------------------------------------
    # Check dependencies
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"retail_clean": INPUT_FILE},
        pipeline_name="step_05_promotion_uplift",
    )

    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load and prepare data
    # ---------------------------------------------------------

    print("Loading retail data...")
    retail = load_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(retail, dataframe_name="retail data")
    print(f"Loaded {len(retail):,} records")
    print(f"Promotion rate: {retail[TREATMENT_COLUMN].mean():.2%}\n")

    stockout_rate = float(
        retail.get("stockout_flag", pd.Series(0, index=retail.index))
        .fillna(0)
        .astype(float)
        .mean()
    )

    # Add calendar features and normalize the promotion indicator
    retail = add_calendar_features(retail, date_column="date")
    retail[TREATMENT_COLUMN] = normalize_promotion_indicator(
        retail[TREATMENT_COLUMN]
    )

    # ---------------------------------------------------------
    # Prepare promotion modeling data
    # ---------------------------------------------------------

    print("Preparing promotion modeling data...")

    available_features = [
        col for col in FEATURE_COLUMNS if col in retail.columns
    ]

    prepared_promotion_data = prepare_promotion_data(
        retail,
        outcome_column=OUTCOME_COLUMN,
        treatment_column=TREATMENT_COLUMN,
        feature_columns=available_features,
        categorical_columns=[
            column
            for column in CATEGORICAL_COLUMNS
            if column in retail.columns
        ],
    )

    if len(prepared_promotion_data.dataframe) != len(retail):
        raise ValueError(
            "prepare_promotion_data dropped rows with missing values, so "
            "identifier columns (product_id/store_id/date) can no longer "
            "be reattached by position. Rework this pipeline to carry an "
            "explicit join key through preparation instead."
        )

    prepared_promotion_data.dataframe[IDENTIFIER_COLUMNS] = retail[
        IDENTIFIER_COLUMNS
    ].reset_index(drop=True)

    print(f"Prepared {len(prepared_promotion_data.dataframe):,} observations")
    print(f"Features: {len(prepared_promotion_data.feature_columns)}\n")

    # ---------------------------------------------------------
    # Fit propensity model
    # ---------------------------------------------------------

    print("Fitting propensity score model...")

    propensity_model = fit_propensity_model(
        prepared_promotion_data,
        random_state=RANDOM_STATE,
    )

    propensity_scores = calculate_propensity_scores(
        prepared_promotion_data,
        propensity_model,
    )

    print(
        f"Propensity scores - mean: {propensity_scores.mean():.3f}, "
        f"std: {propensity_scores.std():.3f}\n"
    )

    # Trim to common support and estimate an IPW-adjusted effect as a
    # supplementary diagnostic. This is not required for the T-learner
    # uplift model below (this pipeline's primary output) and can
    # legitimately fail to find adequate overlap for some datasets —
    # in that case we skip it rather than crash the whole pipeline.
    common_support_data: object | None = None
    ipw_effect: dict[str, float | int] | None = None

    try:
        common_support_data, common_support_scores = trim_common_support(
            prepared_promotion_data,
            propensity_scores,
            lower_quantile=0.05,
            upper_quantile=0.95,
        )

        print(f"Common support: {len(common_support_data.dataframe):,} observations\n")

        ipw_effect = estimate_ipw_promotion_effect(
            common_support_data,
            common_support_scores,
        )

        print(
            "IPW-adjusted promotion effect: "
            f"{ipw_effect['average_treatment_effect']:.2f} units\n"
        )
    except ValueError as error:
        print(
            "Skipping IPW/common-support diagnostic (not required for the "
            f"T-learner model below): {error}\n"
        )

    # ---------------------------------------------------------
    # Split data for model validation
    # ---------------------------------------------------------

    print("Creating train/holdout split for validation...")

    # Use complete dates for the temporal holdout.  Row-position splits can
    # put some stores/products from one day in training and the rest of the
    # same day in validation.
    sorted_data = prepared_promotion_data.dataframe.sort_values("date")
    unique_dates = pd.Index(sorted_data["date"].drop_duplicates())

    if len(unique_dates) < 5:
        raise ValueError(
            "At least five distinct dates are required for promotion validation."
        )

    holdout_date_count = max(1, int(np.ceil(len(unique_dates) * 0.2)))
    holdout_start = unique_dates[-holdout_date_count]
    training_data = sorted_data.loc[sorted_data["date"] < holdout_start].copy()
    holdout_data = sorted_data.loc[sorted_data["date"] >= holdout_start].copy()

    training_prepared_data = PreparedPromotionData(
        dataframe=training_data,
        outcome_column=prepared_promotion_data.outcome_column,
        treatment_column=prepared_promotion_data.treatment_column,
        feature_columns=prepared_promotion_data.feature_columns,
    )

    holdout_prepared_data = PreparedPromotionData(
        dataframe=holdout_data,
        outcome_column=prepared_promotion_data.outcome_column,
        treatment_column=prepared_promotion_data.treatment_column,
        feature_columns=prepared_promotion_data.feature_columns,
    )

    print(f"Training: {len(training_data):,} observations")
    print(f"Holdout: {len(holdout_data):,} observations\n")

    # ---------------------------------------------------------
    # Fit T-learner uplift model on training data
    # ---------------------------------------------------------

    print("Fitting T-learner uplift model...")

    holdout_model_bundle = fit_promotion_t_learner(
        training_prepared_data,
        outcome_model=GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
        random_state=RANDOM_STATE,
    )

    # Predict on holdout (uses holdout_model_bundle's own prepared_data,
    # which is the training split's schema; swap in the holdout rows)
    holdout_model_bundle.prepared_data = holdout_prepared_data
    holdout_predictions = predict_promotion_uplift(holdout_model_bundle)

    print(f"Holdout predictions generated: {len(holdout_predictions):,}")
    print(f"Mean predicted uplift: {holdout_predictions['predicted_uplift'].mean():.2f} units\n")

    # ---------------------------------------------------------
    # Evaluate holdout performance
    # ---------------------------------------------------------

    print("Evaluating holdout performance...")

    # Evaluate by decile
    holdout_uplift_deciles = evaluate_uplift_by_decile(
        holdout_predictions,
        outcome_column=OUTCOME_COLUMN,
        treatment_column=TREATMENT_COLUMN,
        number_of_groups=10,
    )

    print("Holdout uplift by decile:")
    print(holdout_uplift_deciles[
        ["uplift_group", "observations", "average_predicted_uplift", "observed_uplift"]
    ].to_string(index=False))
    print()

    support_observations = (
        len(common_support_data.dataframe)
        if common_support_data is not None
        else 0
    )
    support_fraction = support_observations / max(len(prepared_promotion_data.dataframe), 1)
    top_decile_uplift = float(holdout_uplift_deciles.iloc[0]["observed_uplift"])
    bottom_decile_uplift = float(holdout_uplift_deciles.iloc[-1]["observed_uplift"])
    ranking_direction_ok = bool(top_decile_uplift > bottom_decile_uplift)

    reliability_reasons = []

    if common_support_data is None or support_fraction < 0.50:
        reliability_reasons.append("insufficient_propensity_overlap")

    if stockout_rate > MAX_ACCEPTABLE_STOCKOUT_RATE:
        reliability_reasons.append("sales_outcome_materially_censored_by_stockouts")

    if not ranking_direction_ok:
        reliability_reasons.append("holdout_uplift_ranking_failed")

    model_is_reliable = not reliability_reasons

    if model_is_reliable:
        print("Promotion model readiness: PASS - eligible for pricing input\n")
    else:
        print(
            "Promotion model readiness: HOLD - "
            + ", ".join(reliability_reasons)
            + "\n"
        )

    # Evaluate the promoted-outcome model's accuracy against actual
    # observed demand, restricted to rows that were actually promoted
    # (that's the only subset where "actual outcome under promotion" is
    # directly observed).
    treated_holdout = holdout_predictions.loc[
        holdout_predictions[TREATMENT_COLUMN] == 1
    ].rename(columns={"predicted_promoted_outcome": "forecast_quantity"})

    if not treated_holdout.empty:
        product_accuracy = summarize_segment_accuracy(
            treated_holdout,
            segment_column="product_id",
        )
        print(f"Product-level accuracy evaluated: {len(product_accuracy)} products\n")

    # ---------------------------------------------------------
    # Fit final model on all data
    # ---------------------------------------------------------

    print("Fitting final uplift model on full dataset...")

    final_model_bundle = fit_promotion_t_learner(
        prepared_promotion_data,
        outcome_model=GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
        random_state=RANDOM_STATE,
    )

    promotion_predictions = predict_promotion_uplift(final_model_bundle)
    promotion_predictions["promotion_model_is_reliable"] = model_is_reliable
    promotion_predictions["promotion_recommendation_eligible"] = (
        model_is_reliable
        & promotion_predictions["predicted_uplift"].gt(0)
    )

    print(f"Final predictions: {len(promotion_predictions):,}")
    print(f"Mean predicted uplift: {promotion_predictions['predicted_uplift'].mean():.2f} units\n")

    # ---------------------------------------------------------
    # Rank promotion opportunities
    # ---------------------------------------------------------

    print("Ranking promotion opportunities...")

    top_promotion_opportunities = rank_promotion_opportunities(
        promotion_predictions,
        top_n=TOP_N_OPPORTUNITIES,
    )

    prediction_summary = summarize_promotion_predictions(promotion_predictions)

    print("Overall prediction summary:")
    print(f"  Observations: {prediction_summary['observations']:,}")
    print(f"  Mean uplift: {prediction_summary['average_predicted_uplift']:.2f}")
    print(f"  Positive-uplift rate: {prediction_summary['positive_uplift_rate']:.2%}")
    eligible_recommendations = int(
        promotion_predictions["promotion_recommendation_eligible"].sum()
    )
    print(
        f"  Positive model scores: "
        f"{prediction_summary['recommended_promotions']:,}"
    )
    print(
        f"  Action-eligible promotion recommendations: "
        f"{eligible_recommendations:,}\n"
    )

    print(f"Top {TOP_N_OPPORTUNITIES} opportunities identified\n")

    # ---------------------------------------------------------
    # Aggregate by product and store
    # ---------------------------------------------------------

    print("Aggregating uplift by segment...")

    promotion_uplift_by_product = (
        promotion_predictions.groupby("product_id")
        .agg(
            {
                "predicted_uplift": ["mean", "sum", "count"],
                "predicted_promoted_outcome": "mean",
                "predicted_control_outcome": "mean",
            }
        )
        .reset_index()
    )
    promotion_uplift_by_product.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in promotion_uplift_by_product.columns.values
    ]

    # Uplift as a fraction of baseline (control) demand -- the
    # multiplicative rate consumed by the pricing optimizer downstream.
    # safe_divide avoids +/-inf for a low-volume product whose predicted
    # control outcome is at or near zero.
    promotion_uplift_by_product["promotion_uplift_rate"] = safe_divide(
        promotion_uplift_by_product["predicted_uplift_mean"],
        promotion_uplift_by_product["predicted_control_outcome_mean"],
    )
    promotion_uplift_by_product["promotion_model_is_reliable"] = (
        model_is_reliable
    )
    promotion_uplift_by_product["promotion_reliability_reason"] = (
        "ready" if model_is_reliable else ";".join(reliability_reasons)
    )
    promotion_uplift_by_product["promotion_uplift_rate_for_pricing"] = (
        promotion_uplift_by_product["promotion_uplift_rate"]
        if model_is_reliable
        else 0.0
    )

    promotion_uplift_by_store = (
        promotion_predictions.groupby("store_id")
        .agg(
            {
                "predicted_uplift": ["mean", "sum", "count"],
                "predicted_promoted_outcome": "mean",
                "predicted_control_outcome": "mean",
            }
        )
        .reset_index()
    )
    promotion_uplift_by_store.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in promotion_uplift_by_store.columns.values
    ]

    promotion_uplift_by_store["promotion_uplift_rate"] = safe_divide(
        promotion_uplift_by_store["predicted_uplift_mean"],
        promotion_uplift_by_store["predicted_control_outcome_mean"],
    )

    print(f"  Products: {len(promotion_uplift_by_product)}")
    print(f"  Stores: {len(promotion_uplift_by_store)}\n")

    # ---------------------------------------------------------
    # Create model metrics summary
    # ---------------------------------------------------------

    promotion_model_metrics = pd.DataFrame([
        {
            "metric": "Training observations",
            "value": len(training_data),
        },
        {
            "metric": "Holdout observations",
            "value": len(holdout_data),
        },
        {
            "metric": "Mean predicted uplift",
            "value": promotion_predictions["predicted_uplift"].mean(),
        },
        {
            "metric": "IPW average treatment effect",
            "value": (
                ipw_effect["average_treatment_effect"]
                if ipw_effect is not None
                else np.nan
            ),
        },
        {
            "metric": "Common-support fraction",
            "value": support_fraction,
        },
        {
            "metric": "Observed stockout rate",
            "value": stockout_rate,
        },
        {
            "metric": "Holdout ranking direction passed",
            "value": int(ranking_direction_ok),
        },
        {
            "metric": "Model eligible for pricing",
            "value": int(model_is_reliable),
        },
    ])

    # Propensity diagnostics
    propensity_diagnostics = pd.DataFrame([
        {
            "statistic": "Mean propensity",
            "value": propensity_scores.mean(),
        },
        {
            "statistic": "Std propensity",
            "value": propensity_scores.std(),
        },
        {
            "statistic": "Common support observations",
            "value": (
                len(common_support_data.dataframe)
                if common_support_data is not None
                else np.nan
            ),
        },
        {
            "statistic": "Common support fraction",
            "value": support_fraction,
        },
        {
            "statistic": "Model eligible for pricing",
            "value": int(model_is_reliable),
        },
    ])

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving promotion uplift results...")

    save_csv(promotion_predictions, OUTPUT_FILES["predictions"])
    save_csv(promotion_uplift_by_product, OUTPUT_FILES["by_product"])
    save_csv(promotion_uplift_by_store, OUTPUT_FILES["by_store"])
    save_csv(holdout_uplift_deciles, OUTPUT_FILES["deciles"])
    save_csv(promotion_model_metrics, OUTPUT_FILES["metrics"])
    save_csv(top_promotion_opportunities, OUTPUT_FILES["top_opportunities"])
    save_csv(propensity_diagnostics, OUTPUT_FILES["propensity_diagnostics"])

    save_model(final_model_bundle, MODEL_FILES["uplift_model"])

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    print(f"\nModel saved to {MODEL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  - {MODEL_FILES['uplift_model'].name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 05 COMPLETE")
    print("=" * 72)
    print(f"\nTop {TOP_N_OPPORTUNITIES} promotion opportunities identified")
    print(f"Action-eligible promotions: {eligible_recommendations:,}")
    print("=" * 72)


if __name__ == "__main__":
    main()
