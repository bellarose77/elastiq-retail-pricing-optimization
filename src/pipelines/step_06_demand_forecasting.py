"""
Pipeline Step 06: Demand Forecasting with XGBoost

This pipeline:
1. Loads cleaned retail data
2. Creates lag and rolling window features
3. Splits data chronologically into train/val/test
4. Fits XGBoost forecasting model
5. Evaluates forecast accuracy by segment
6. Generates next-period demand forecasts
7. Saves forecasts, metrics, and model
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_LAG_PERIODS,
    FORECAST_ROLLING_WINDOWS,
    INTERIM_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from src.data import load_csv, save_csv, save_model, validate_dataframe
from src.data.splitting import create_chronological_split
from src.features import add_lag_features, add_rolling_features
from src.models import (
    calculate_forecast_metrics,
    compare_forecast_with_baseline,
    fit_xgboost_forecast,
    get_feature_importance,
    predict_future_demand,
)
from src.models.evaluation import summarize_segment_accuracy
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)

GROUP_COLUMNS = ["product_id", "store_id"]

CATEGORICAL_FEATURE_COLUMNS = ["product_id", "store_id"]

# `inventory_level` is deliberately ABSENT from this list.
#
# It is a same-day, end-of-period stock reading that is not knowable when
# the forecast is made, and on this dataset it equals `units_sold` exactly
# on 25.9% of rows (every stockout day), correlating 0.933 with the target.
# Including it carried 61.5% of the model's feature importance and lifted
# reported accuracy from MAE 6.36 to 4.84 -- roughly a third of the
# headline number came from a feature that partly IS the target. The
# lagged value below is legitimately known at forecast time and is kept.
BASE_NUMERIC_FEATURE_COLUMNS = [
    "selling_price",
    "promotion_flag",
    "weekend_flag",
    "holiday_flag",
    "weather_index",
    "customer_traffic",
]

INVENTORY_LAG_COLUMN = "inventory_level_lag_1"


def main() -> None:
    """Execute the demand forecasting pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 06 — XGBOOST DEMAND FORECASTING")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    INPUT_FILE = INTERIM_DIR / "retail_demand_clean.csv"

    OUTPUT_FILES = {
        "test_predictions": PROCESSED_DIR / "xgboost_forecast_test_predictions.csv",
        "metrics": PROCESSED_DIR / "xgboost_forecast_metrics.csv",
        "baseline_comparison": PROCESSED_DIR / "xgboost_baseline_comparison.csv",
        "daily_test_forecast": PROCESSED_DIR / "xgboost_daily_test_forecast.csv",
        "feature_importance": PROCESSED_DIR / "xgboost_feature_importance.csv",
        "product_accuracy": PROCESSED_DIR / "xgboost_forecast_accuracy_by_product.csv",
        "store_accuracy": PROCESSED_DIR / "xgboost_forecast_accuracy_by_store.csv",
        "next_period_forecast": PROCESSED_DIR / "xgboost_next_period_forecast.csv",
    }

    MODEL_FILE = MODEL_DIR / "xgboost_demand_forecast.pkl"

    # ---------------------------------------------------------
    # Check dependencies
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"retail_clean": INPUT_FILE},
        pipeline_name="step_06_demand_forecasting",
    )

    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading retail data...")
    retail_data = load_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(retail_data, dataframe_name="retail data")
    print(f"Loaded {len(retail_data):,} records")
    print(f"Date range: {retail_data['date'].min().date()} to {retail_data['date'].max().date()}\n")

    # ---------------------------------------------------------
    # Prepare forecasting features
    # ---------------------------------------------------------

    print("Preparing forecasting features...")

    forecast_data = retail_data.sort_values(GROUP_COLUMNS + ["date"]).copy()

    forecast_data = add_lag_features(
        forecast_data,
        value_columns=["units_sold"],
        group_columns=GROUP_COLUMNS,
        date_column="date",
        lags=FORECAST_LAG_PERIODS,
    )

    forecast_data = add_rolling_features(
        forecast_data,
        value_columns=["units_sold"],
        group_columns=GROUP_COLUMNS,
        date_column="date",
        windows=FORECAST_ROLLING_WINDOWS,
    )

    forecast_data = add_lag_features(
        forecast_data,
        value_columns=["inventory_level"],
        group_columns=GROUP_COLUMNS,
        date_column="date",
        lags=[1],
    )

    lag_columns = [f"units_sold_lag_{lag}" for lag in FORECAST_LAG_PERIODS]
    rolling_columns = [
        f"units_sold_rolling_{stat}_{window}"
        for window in FORECAST_ROLLING_WINDOWS
        for stat in ("mean", "std")
    ]

    FEATURE_COLUMNS = [
        *lag_columns,
        *rolling_columns,
        *BASE_NUMERIC_FEATURE_COLUMNS,
        INVENTORY_LAG_COLUMN,
    ]

    initial_rows = len(forecast_data)
    forecast_data = forecast_data.dropna(
        subset=[*lag_columns, INVENTORY_LAG_COLUMN]
    )
    dropped_rows = initial_rows - len(forecast_data)

    print(f"Created lag features: {FORECAST_LAG_PERIODS}")
    print(f"Created rolling windows: {FORECAST_ROLLING_WINDOWS}")
    print(f"Dropped {dropped_rows:,} rows with missing lags")
    print(f"Forecasting dataset: {len(forecast_data):,} rows\n")

    # ---------------------------------------------------------
    # Split data chronologically
    # ---------------------------------------------------------

    print("Creating chronological train/val/test split...")

    train_data, val_data, test_data = create_chronological_split(
        forecast_data,
        date_column="date",
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    print(f"Training:   {len(train_data):,} rows ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
    print(f"Validation: {len(val_data):,} rows ({val_data['date'].min().date()} to {val_data['date'].max().date()})")
    print(f"Test:       {len(test_data):,} rows ({test_data['date'].min().date()} to {test_data['date'].max().date()})\n")

    # Train + val are combined into "development" data. fit_xgboost_forecast
    # re-splits this internally (time-based) for its own early-stopping
    # validation; the true test set below never touches model fitting.
    development_data = pd.concat([train_data, val_data], ignore_index=True)

    # ---------------------------------------------------------
    # Fit XGBoost model
    # ---------------------------------------------------------

    print("Fitting XGBoost forecasting model...")

    model_bundle = fit_xgboost_forecast(
        development_data,
        target_column="units_sold",
        date_column="date",
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_FEATURE_COLUMNS,
        number_of_estimators=100,
        learning_rate=0.1,
        maximum_depth=6,
        random_state=RANDOM_STATE,
    )

    print(f"Model trained on {len(model_bundle.training_data):,} observations")
    print(f"Model features: {len(model_bundle.prepared_data.feature_columns)}")
    print(
        "Internal validation MAPE: "
        f"{model_bundle.validation_metrics['mape_percent']:.2f}%\n"
    )

    # ---------------------------------------------------------
    # Generate test predictions on the true held-out test set
    # ---------------------------------------------------------

    print("Generating test predictions...")

    test_predictions = predict_future_demand(
        test_data,
        model_bundle,
        include_prediction_interval=True,
    )
    test_predictions["forecast_quantity"] = test_predictions["predicted_value"]

    test_metrics = calculate_forecast_metrics(
        test_predictions["units_sold"].to_numpy(),
        test_predictions["forecast_quantity"].to_numpy(),
    )

    print("Test performance:")
    print(f"  MAE:              {test_metrics['mae']:.2f}")
    print(f"  RMSE:             {test_metrics['rmse']:.2f}")
    print(f"  R-squared:        {test_metrics['r_squared']:.4f}")
    print(f"  MAPE:             {test_metrics['mape_percent']:.2f}%")
    print(f"  WAPE:             {test_metrics['wape_percent']:.2f}%")
    print(f"  SMAPE:            {test_metrics['smape_percent']:.2f}%\n")

    baseline_comparison = compare_forecast_with_baseline(model_bundle)

    # ---------------------------------------------------------
    # Evaluate by segment using shared function
    # ---------------------------------------------------------

    print("Evaluating forecast accuracy by segment...")

    product_accuracy = summarize_segment_accuracy(
        test_predictions,
        segment_column="product_id",
    )

    store_accuracy = summarize_segment_accuracy(
        test_predictions,
        segment_column="store_id",
    )

    print(f"Product-level accuracy: {len(product_accuracy)} products")
    print(f"Store-level accuracy: {len(store_accuracy)} stores\n")

    # ---------------------------------------------------------
    # Get feature importance
    # ---------------------------------------------------------

    print("Extracting feature importance...")

    feature_importance = get_feature_importance(
        model_bundle,
        top_n=20,
    )

    print("Top 20 features extracted\n")

    # ---------------------------------------------------------
    # Aggregate daily forecast
    # ---------------------------------------------------------

    daily_test_forecast = (
        test_predictions.groupby("date")
        .agg(
            {
                "units_sold": "sum",
                "forecast_quantity": "sum",
            }
        )
        .reset_index()
    )

    daily_test_forecast.rename(
        columns={"units_sold": "actual_quantity"},
        inplace=True,
    )

    # ---------------------------------------------------------
    # Generate next-period forecast
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Naive baselines
    # ---------------------------------------------------------
    #
    # The library's `compare_forecast_with_baseline` uses a MEAN baseline,
    # which is weaker than every naive forecaster and made the model look
    # ~68% better than "baseline". Seasonal-naive and rolling-mean are the
    # comparators a demand-forecasting reviewer will expect.

    print("Comparing against naive baselines...")

    naive_baselines = []

    for label, column in (
        ("Naive (lag 1)", "units_sold_lag_1"),
        ("Seasonal naive (lag 7)", "units_sold_lag_7"),
        ("Rolling mean (7)", "units_sold_rolling_mean_7"),
    ):
        if column not in test_predictions.columns:
            continue

        baseline_metrics = calculate_forecast_metrics(
            test_predictions["units_sold"],
            test_predictions[column],
        )

        naive_baselines.append(
            {
                "model": label,
                **baseline_metrics,
            }
        )

    naive_baseline_frame = pd.DataFrame(
        [
            *naive_baselines,
            {"model": "XGBoost", **test_metrics},
        ]
    )

    best_naive_mae = min(
        (row["mae"] for row in naive_baselines),
        default=float("nan"),
    )

    if naive_baselines:
        improvement = (
            100.0 * (best_naive_mae - test_metrics["mae"]) / best_naive_mae
        )

        print(
            f"  Best naive MAE: {best_naive_mae:.2f}  |  "
            f"XGBoost MAE: {test_metrics['mae']:.2f}  |  "
            f"improvement: {improvement:.1f}%"
        )

    save_csv(
        naive_baseline_frame,
        PROCESSED_DIR / "xgboost_naive_baseline_comparison.csv",
    )

    print()

    print("Generating next-period forecast...")

    next_forecast_date = retail_data["date"].max() + pd.Timedelta(days=1)

    history = retail_data.sort_values(GROUP_COLUMNS + ["date"])
    holiday_month_days = set(
        zip(
            retail_data.loc[retail_data["holiday_flag"].eq(1), "date"].dt.month,
            retail_data.loc[retail_data["holiday_flag"].eq(1), "date"].dt.day,
        )
    )

    def _build_next_period_row(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("date")
        actual_units = ordered["units_sold"]

        next_row = ordered.iloc[-1].copy()
        next_row["date"] = next_forecast_date

        # Explicit one-day-ahead assumptions.  Do not silently copy a past
        # promotion or stale calendar fields into tomorrow's forecast.
        next_row["year"] = next_forecast_date.year
        next_row["month"] = next_forecast_date.month
        next_row["day_of_week"] = next_forecast_date.day_name()
        next_row["weekend_flag"] = int(next_forecast_date.dayofweek >= 5)
        next_row["holiday_flag"] = int(
            (next_forecast_date.month, next_forecast_date.day)
            in holiday_month_days
        )
        next_row["quarter"] = next_forecast_date.quarter
        next_row["week_of_year"] = int(next_forecast_date.isocalendar().week)
        next_row["day_of_month"] = next_forecast_date.day
        next_row["day_name"] = next_forecast_date.day_name()
        next_row["is_weekend"] = int(next_forecast_date.dayofweek >= 5)
        next_row["is_month_start"] = int(next_forecast_date.is_month_start)
        next_row["is_month_end"] = int(next_forecast_date.is_month_end)
        next_row["promotion_flag"] = 0
        next_row["promotion_type"] = "none"
        next_row["discount_pct"] = 0.0
        next_row["selling_price"] = float(ordered["selling_price"].iloc[-1])
        next_row["weather_index"] = float(
            ordered["weather_index"].tail(7).mean()
        )

        same_weekday_history = ordered.loc[
            ordered["date"].dt.dayofweek.eq(next_forecast_date.dayofweek),
            "customer_traffic",
        ].tail(8)
        next_row["customer_traffic"] = float(
            same_weekday_history.mean()
            if not same_weekday_history.empty
            else ordered["customer_traffic"].tail(28).mean()
        )

        # Tomorrow's lag-1 stock level is today's observed stock level.
        # This is the whole point of using the lagged value: it is known at
        # forecast time, whereas the same-day reading is not.
        next_row[INVENTORY_LAG_COLUMN] = ordered["inventory_level"].iloc[-1]
        next_row["available_inventory_for_pricing"] = ordered[
            "inventory_level"
        ].iloc[-1]
        next_row["forecast_price_assumption"] = "last_observed_price"
        next_row["forecast_promotion_assumption"] = "no_planned_promotion"
        next_row["forecast_weather_assumption"] = "trailing_7_day_mean"
        next_row["forecast_traffic_assumption"] = "same_weekday_trailing_8"
        next_row["inventory_assumption"] = "last_observed_capacity"

        for lag in FORECAST_LAG_PERIODS:
            next_row[f"units_sold_lag_{lag}"] = (
                actual_units.iloc[-lag] if len(actual_units) >= lag else np.nan
            )

        for window in FORECAST_ROLLING_WINDOWS:
            recent_values = actual_units.iloc[-window:]
            next_row[f"units_sold_rolling_mean_{window}"] = recent_values.mean()
            next_row[f"units_sold_rolling_std_{window}"] = recent_values.std()

        return next_row

    # pandas 2.2 deprecated (and 3.0 removed) passing the grouping columns
    # into `apply`, so `ordered.iloc[-1]` silently stopped carrying
    # product_id/store_id and the next-period forecast crashed on any
    # modern pandas. Build the rows explicitly instead of relying on that.
    future_rows = []

    for group_key, group_frame in history.groupby(
        GROUP_COLUMNS,
        sort=True,
    ):
        next_row = _build_next_period_row(group_frame)

        for column, value in zip(GROUP_COLUMNS, group_key, strict=True):
            next_row[column] = value

        future_rows.append(next_row)

    future_data = pd.DataFrame(future_rows).reset_index(drop=True)

    next_period_forecast = predict_future_demand(
        future_data,
        model_bundle,
        include_prediction_interval=True,
    )
    next_period_forecast["forecast_quantity"] = next_period_forecast["predicted_value"]

    total_next_forecast = next_period_forecast["forecast_quantity"].sum()

    print(f"Next forecast date: {next_forecast_date.date()}")
    print(f"Store-product combinations: {len(next_period_forecast):,}")
    print(f"Total forecast: {total_next_forecast:,.0f} units\n")

    # ---------------------------------------------------------
    # Create metrics summary
    # ---------------------------------------------------------

    forecast_metrics = pd.DataFrame([
        {
            "metric": "Training observations",
            "value": len(model_bundle.training_data),
        },
        {
            "metric": "Test observations",
            "value": len(test_data),
        },
        {
            "metric": "Test MAE",
            "value": test_metrics["mae"],
        },
        {
            "metric": "Test RMSE",
            "value": test_metrics["rmse"],
        },
        {
            "metric": "Test R-squared",
            "value": test_metrics["r_squared"],
        },
        {
            "metric": "Test MAPE %",
            "value": test_metrics["mape_percent"],
        },
        {
            "metric": "Test WAPE %",
            "value": test_metrics["wape_percent"],
        },
        {
            "metric": "Test SMAPE %",
            "value": test_metrics["smape_percent"],
        },
    ])

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving forecasting outputs...")

    save_csv(test_predictions, OUTPUT_FILES["test_predictions"])
    save_csv(forecast_metrics, OUTPUT_FILES["metrics"])
    save_csv(baseline_comparison, OUTPUT_FILES["baseline_comparison"])
    save_csv(daily_test_forecast, OUTPUT_FILES["daily_test_forecast"])
    save_csv(feature_importance, OUTPUT_FILES["feature_importance"])
    save_csv(product_accuracy, OUTPUT_FILES["product_accuracy"])
    save_csv(store_accuracy, OUTPUT_FILES["store_accuracy"])
    save_csv(next_period_forecast, OUTPUT_FILES["next_period_forecast"])

    save_model(model_bundle, MODEL_FILE)

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    print(f"\nModel saved to {MODEL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  - {MODEL_FILE.name}")

    print("\n" + "=" * 72)
    print("PIPELINE STEP 06 COMPLETE")
    print("=" * 72)
    print(f"\nTest MAPE: {test_metrics['mape_percent']:.2f}%")
    print(f"Next-period forecast: {total_next_forecast:,.0f} units for {next_forecast_date.date()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
