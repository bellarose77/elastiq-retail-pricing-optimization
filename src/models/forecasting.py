"""XGBoost forecasting, evaluation, and prediction utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.validation import validate_required_columns


@dataclass(slots=True)
class PreparedForecastData:
    """Prepared dataset and metadata for forecasting."""

    dataframe: pd.DataFrame
    target_column: str
    date_column: str
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_columns: list[str]


@dataclass(slots=True)
class ForecastModelBundle:
    """Store a fitted forecasting model and its validation results."""

    model: Any
    prepared_data: PreparedForecastData
    training_data: pd.DataFrame
    validation_data: pd.DataFrame
    validation_predictions: pd.DataFrame
    validation_metrics: dict[str, float | int]
    baseline_metrics: dict[str, float | int]
    residual_lower_quantile: float
    residual_upper_quantile: float


def prepare_forecasting_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
) -> PreparedForecastData:
    """Prepare numeric and categorical features for forecasting."""

    numeric_feature_columns = list(feature_columns)
    categorical_columns = list(
        categorical_columns or []
    )

    if not numeric_feature_columns and not categorical_columns:
        raise ValueError(
            "At least one forecasting feature is required."
        )

    overlapping_columns = (
        set(numeric_feature_columns)
        & set(categorical_columns)
    )

    if overlapping_columns:
        raise ValueError(
            "Columns cannot be included as both numeric and "
            f"categorical features: {sorted(overlapping_columns)}"
        )

    required_columns = [
        target_column,
        date_column,
        *numeric_feature_columns,
        *categorical_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="forecasting data",
    )

    prepared_dataframe = pd.DataFrame(
        index=dataframe.index
    )

    prepared_dataframe[date_column] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    )

    prepared_dataframe[target_column] = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    for column in numeric_feature_columns:
        prepared_dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    generated_feature_columns: list[str] = []

    if categorical_columns:
        categorical_features = pd.get_dummies(
            dataframe[categorical_columns].astype("string"),
            columns=categorical_columns,
            prefix=categorical_columns,
            drop_first=True,
            dtype=float,
        )

        generated_feature_columns = list(
            categorical_features.columns
        )

        prepared_dataframe = pd.concat(
            [
                prepared_dataframe,
                categorical_features,
            ],
            axis=1,
        )

    prepared_dataframe = prepared_dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    prepared_dataframe = prepared_dataframe.dropna(
        subset=[
            date_column,
            target_column,
        ]
    )

    final_feature_columns = [
        *numeric_feature_columns,
        *generated_feature_columns,
    ]

    constant_columns = [
        column
        for column in final_feature_columns
        if prepared_dataframe[column].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:
        prepared_dataframe = prepared_dataframe.drop(
            columns=constant_columns
        )

        final_feature_columns = [
            column
            for column in final_feature_columns
            if column not in constant_columns
        ]

    if not final_feature_columns:
        raise ValueError(
            "No usable forecasting features remain after preparation."
        )

    prepared_dataframe = (
        prepared_dataframe
        .sort_values(
            date_column,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return PreparedForecastData(
        dataframe=prepared_dataframe,
        target_column=target_column,
        date_column=date_column,
        feature_columns=final_feature_columns,
        numeric_feature_columns=numeric_feature_columns,
        categorical_columns=categorical_columns,
    )


def time_based_train_validation_split(
    prepared_data: PreparedForecastData,
    *,
    validation_fraction: float = 0.20,
    validation_start_date: str | pd.Timestamp | None = None,
    minimum_training_observations: int = 50,
    minimum_validation_observations: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecasting data chronologically without random shuffling."""

    dataframe = prepared_data.dataframe

    date_column = prepared_data.date_column

    if validation_start_date is not None:
        cutoff_date = pd.Timestamp(
            validation_start_date
        )

        training_data = dataframe.loc[
            dataframe[date_column] < cutoff_date
        ].copy()

        validation_data = dataframe.loc[
            dataframe[date_column] >= cutoff_date
        ].copy()

    else:
        if not 0 < validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        unique_dates = pd.Index(
            dataframe[date_column]
            .sort_values()
            .drop_duplicates()
        )

        if len(unique_dates) < 2:
            raise ValueError(
                "At least two distinct dates are required for validation."
            )

        validation_date_count = max(
            1,
            int(np.ceil(len(unique_dates) * validation_fraction)),
        )
        validation_date_count = min(
            validation_date_count,
            len(unique_dates) - 1,
        )
        cutoff_date = unique_dates[-validation_date_count]

        training_data = dataframe.loc[
            dataframe[date_column] < cutoff_date
        ].copy()

        validation_data = dataframe.loc[
            dataframe[date_column] >= cutoff_date
        ].copy()

    if len(training_data) < minimum_training_observations:
        raise ValueError(
            "Insufficient training observations. "
            f"Required: {minimum_training_observations:,}; "
            f"available: {len(training_data):,}."
        )

    if len(validation_data) < minimum_validation_observations:
        raise ValueError(
            "Insufficient validation observations. "
            f"Required: {minimum_validation_observations:,}; "
            f"available: {len(validation_data):,}."
        )

    if (
        training_data[date_column].max()
        >= validation_data[date_column].min()
    ):
        raise ValueError(
            "The chronological training and validation periods overlap."
        )

    return (
        training_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
    )


def calculate_forecast_metrics(
    actual_values: Sequence[float] | pd.Series | np.ndarray,
    predicted_values: Sequence[float] | pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate standard regression and forecasting metrics."""

    actual = np.asarray(
        actual_values,
        dtype=float,
    )

    predicted = np.asarray(
        predicted_values,
        dtype=float,
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape."
        )

    valid_mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        raise ValueError(
            "No valid observations are available for evaluation."
        )

    errors = actual - predicted

    absolute_actual_sum = float(np.abs(actual).sum())
    wape = (
        float(np.abs(errors).sum() / absolute_actual_sum * 100)
        if absolute_actual_sum > 0
        else np.nan
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    nonzero_actual_mask = actual != 0

    if nonzero_actual_mask.any():
        mape = np.mean(
            np.abs(
                errors[nonzero_actual_mask]
                / actual[nonzero_actual_mask]
            )
        ) * 100
    else:
        mape = np.nan

    smape_denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid_smape_mask = (
        smape_denominator != 0
    )

    if valid_smape_mask.any():
        smape = np.mean(
            2
            * np.abs(errors[valid_smape_mask])
            / smape_denominator[
                valid_smape_mask
            ]
        ) * 100
    else:
        smape = np.nan

    if len(actual) > 1:
        r_squared = r2_score(
            actual,
            predicted,
        )
    else:
        r_squared = np.nan

    mean_actual = float(
        np.mean(actual)
    )

    if mean_actual == 0:
        normalized_mae = np.nan
    else:
        normalized_mae = (
            float(mae)
            / abs(mean_actual)
        )

    return {
        "observations": int(
            len(actual)
        ),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "wape_percent": float(wape),
        "smape_percent": float(smape),
        "r_squared": float(r_squared),
        "mean_actual": mean_actual,
        "mean_prediction": float(
            np.mean(predicted)
        ),
        "normalized_mae": float(
            normalized_mae
        ),
    }


def create_xgboost_regressor(
    *,
    number_of_estimators: int = 400,
    learning_rate: float = 0.05,
    maximum_depth: int = 6,
    minimum_child_weight: float = 1.0,
    subsample: float = 0.85,
    column_subsample: float = 0.85,
    l1_regularization: float = 0.0,
    l2_regularization: float = 1.0,
    random_state: int = 42,
    number_of_jobs: int = -1,
) -> Any:
    """Create a configured XGBoost regression model."""

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "The 'xgboost' package is required. "
            "Install it with: python -m pip install xgboost"
        ) from error

    if number_of_estimators <= 0:
        raise ValueError(
            "number_of_estimators must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if maximum_depth <= 0:
        raise ValueError(
            "maximum_depth must be greater than zero."
        )

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=number_of_estimators,
        learning_rate=learning_rate,
        max_depth=maximum_depth,
        min_child_weight=minimum_child_weight,
        subsample=subsample,
        colsample_bytree=column_subsample,
        reg_alpha=l1_regularization,
        reg_lambda=l2_regularization,
        random_state=random_state,
        n_jobs=number_of_jobs,
        tree_method="hist",
    )


def fit_xgboost_forecast(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
    validation_fraction: float = 0.20,
    validation_start_date: str | pd.Timestamp | None = None,
    model: Any | None = None,
    number_of_estimators: int = 400,
    learning_rate: float = 0.05,
    maximum_depth: int = 6,
    random_state: int = 42,
) -> ForecastModelBundle:
    """Prepare data, fit XGBoost, and evaluate a time-based holdout."""

    prepared_data = prepare_forecasting_data(
        dataframe,
        target_column=target_column,
        date_column=date_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )

    training_data, validation_data = (
        time_based_train_validation_split(
            prepared_data,
            validation_fraction=validation_fraction,
            validation_start_date=validation_start_date,
        )
    )

    forecasting_model = (
        model
        if model is not None
        else create_xgboost_regressor(
            number_of_estimators=number_of_estimators,
            learning_rate=learning_rate,
            maximum_depth=maximum_depth,
            random_state=random_state,
        )
    )

    training_features = training_data[
        prepared_data.feature_columns
    ].astype(float)

    training_target = training_data[
        target_column
    ].astype(float)

    validation_features = validation_data[
        prepared_data.feature_columns
    ].astype(float)

    validation_target = validation_data[
        target_column
    ].astype(float)

    forecasting_model.fit(
        training_features,
        training_target,
        eval_set=[
            (
                validation_features,
                validation_target,
            )
        ],
        verbose=False,
    )

    predicted_values = forecasting_model.predict(
        validation_features
    )

    validation_predictions = validation_data[
        [
            date_column,
            target_column,
        ]
    ].copy()

    validation_predictions[
        "predicted_value"
    ] = predicted_values

    validation_predictions["residual"] = (
        validation_predictions[
            target_column
        ]
        - validation_predictions[
            "predicted_value"
        ]
    )

    validation_predictions[
        "absolute_error"
    ] = validation_predictions[
        "residual"
    ].abs()

    validation_metrics = calculate_forecast_metrics(
        validation_target,
        predicted_values,
    )

    baseline_prediction = np.repeat(
        training_target.mean(),
        len(validation_target),
    )

    baseline_metrics = calculate_forecast_metrics(
        validation_target,
        baseline_prediction,
    )

    residual_lower_quantile = float(
        validation_predictions[
            "residual"
        ].quantile(0.05)
    )

    residual_upper_quantile = float(
        validation_predictions[
            "residual"
        ].quantile(0.95)
    )

    validation_predictions[
        "prediction_interval_lower"
    ] = (
        validation_predictions[
            "predicted_value"
        ]
        + residual_lower_quantile
    )

    validation_predictions[
        "prediction_interval_upper"
    ] = (
        validation_predictions[
            "predicted_value"
        ]
        + residual_upper_quantile
    )

    return ForecastModelBundle(
        model=forecasting_model,
        prepared_data=prepared_data,
        training_data=training_data,
        validation_data=validation_data,
        validation_predictions=validation_predictions,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        residual_lower_quantile=(
            residual_lower_quantile
        ),
        residual_upper_quantile=(
            residual_upper_quantile
        ),
    )


def prepare_future_features(
    dataframe: pd.DataFrame,
    model_bundle: ForecastModelBundle,
) -> pd.DataFrame:
    """Prepare new observations using the training feature structure."""

    prepared_data = model_bundle.prepared_data

    required_columns = [
        *prepared_data.numeric_feature_columns,
        *prepared_data.categorical_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="future forecasting data",
    )

    future_features = pd.DataFrame(
        index=dataframe.index
    )

    for column in (
        prepared_data.numeric_feature_columns
    ):
        future_features[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if prepared_data.categorical_columns:
        categorical_features = pd.get_dummies(
            dataframe[
                prepared_data.categorical_columns
            ].astype("string"),
            columns=(
                prepared_data.categorical_columns
            ),
            prefix=(
                prepared_data.categorical_columns
            ),
            drop_first=True,
            dtype=float,
        )

        future_features = pd.concat(
            [
                future_features,
                categorical_features,
            ],
            axis=1,
        )

    future_features = future_features.reindex(
        columns=prepared_data.feature_columns,
        fill_value=0.0,
    )

    future_features = future_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return future_features.astype(float)


def predict_future_demand(
    dataframe: pd.DataFrame,
    model_bundle: ForecastModelBundle,
    *,
    include_prediction_interval: bool = True,
) -> pd.DataFrame:
    """Generate demand forecasts for new observations."""

    prediction_features = prepare_future_features(
        dataframe,
        model_bundle,
    )

    predictions = model_bundle.model.predict(
        prediction_features
    )

    result = dataframe.copy()

    result["predicted_value"] = predictions

    if include_prediction_interval:
        result["prediction_interval_lower"] = (
            result["predicted_value"]
            + model_bundle.residual_lower_quantile
        )

        result["prediction_interval_upper"] = (
            result["predicted_value"]
            + model_bundle.residual_upper_quantile
        )

    return result


def get_feature_importance(
    model_bundle: ForecastModelBundle,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return model feature importance ranked from highest to lowest."""

    model = model_bundle.model

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise TypeError(
            "The forecasting model does not expose "
            "feature_importances_."
        )

    importance = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    feature_columns = (
        model_bundle
        .prepared_data
        .feature_columns
    )

    if len(importance) != len(feature_columns):
        raise ValueError(
            "The feature-importance array does not match "
            "the fitted feature columns."
        )

    importance_dataframe = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    importance_dataframe = (
        importance_dataframe
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_dataframe[
        "importance_rank"
    ] = np.arange(
        1,
        len(importance_dataframe) + 1,
    )

    total_importance = float(
        importance_dataframe[
            "importance"
        ].sum()
    )

    if total_importance > 0:
        importance_dataframe[
            "importance_percent"
        ] = (
            importance_dataframe[
                "importance"
            ]
            / total_importance
            * 100
        )
    else:
        importance_dataframe[
            "importance_percent"
        ] = 0.0

    if top_n is not None:
        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        importance_dataframe = (
            importance_dataframe.head(
                top_n
            )
        )

    return importance_dataframe


def compare_forecast_with_baseline(
    model_bundle: ForecastModelBundle,
) -> pd.DataFrame:
    """Compare XGBoost validation performance with a mean baseline."""

    model_metrics = (
        model_bundle.validation_metrics
    )

    baseline_metrics = (
        model_bundle.baseline_metrics
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Mean Baseline",
                **baseline_metrics,
            },
            {
                "model": "XGBoost",
                **model_metrics,
            },
        ]
    )

    baseline_mae = float(
        baseline_metrics["mae"]
    )

    baseline_rmse = float(
        baseline_metrics["rmse"]
    )

    comparison["mae_improvement_vs_baseline"] = (
        baseline_mae
        - comparison["mae"]
    )

    comparison[
        "rmse_improvement_vs_baseline"
    ] = (
        baseline_rmse
        - comparison["rmse"]
    )

    if baseline_mae == 0:
        comparison[
            "mae_improvement_percent"
        ] = np.nan
    else:
        comparison[
            "mae_improvement_percent"
        ] = (
            comparison[
                "mae_improvement_vs_baseline"
            ]
            / baseline_mae
            * 100
        )

    return comparison
