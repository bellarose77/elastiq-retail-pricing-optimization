"""
Model Evaluation Functions

This module provides functions for evaluating model performance,
particularly for forecasting and uplift models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.forecasting import calculate_forecast_metrics


def summarize_segment_accuracy(
    dataframe: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    """
    Summarize forecast accuracy metrics by segment.

    This function computes comprehensive accuracy metrics for each segment,
    including MAE, RMSE, R², WAPE, MAPE, and bias.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Predictions with columns: units_sold, forecast_quantity, and segment_column
    segment_column : str
        Column name to group by for segment-level metrics

    Returns
    -------
    pd.DataFrame
        Segment-level accuracy summary sorted by WAPE (best to worst)

    Notes
    -----
    Expected columns in dataframe:
    - units_sold: Actual demand values
    - forecast_quantity: Predicted demand values
    - {segment_column}: Segment identifier

    Examples
    --------
    >>> predictions = pd.DataFrame({
    ...     "product_id": ["A", "A", "B", "B"],
    ...     "units_sold": [100, 120, 200, 180],
    ...     "forecast_quantity": [105, 115, 195, 185],
    ... })
    >>> summary = summarize_segment_accuracy(predictions, "product_id")
    """
    records = []

    for segment_value, segment_data in dataframe.groupby(
        segment_column,
        dropna=False,
    ):
        actual_values = segment_data["units_sold"].astype(float).to_numpy()

        predicted_values = (
            segment_data["forecast_quantity"].astype(float).to_numpy()
        )

        metrics = calculate_forecast_metrics(
            actual_values,
            predicted_values,
        )

        forecast_errors = predicted_values - actual_values

        actual_absolute_total = float(np.abs(actual_values).sum())

        actual_total = float(actual_values.sum())

        wape_percent = (
            float(
                np.abs(forecast_errors).sum() / actual_absolute_total * 100
            )
            if actual_absolute_total != 0
            else np.nan
        )

        bias_percent = (
            float(forecast_errors.sum() / actual_total * 100)
            if actual_total != 0
            else np.nan
        )

        records.append(
            {
                segment_column: segment_value,
                "observations": int(len(segment_data)),
                "actual_quantity": float(actual_values.sum()),
                "forecast_quantity": float(predicted_values.sum()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r_squared": metrics["r_squared"],
                "wape_percent": wape_percent,
                "mape_percent": metrics["mape_percent"],
                "bias_percent": bias_percent,
                "forecast_accuracy_percent": (
                    100 - wape_percent if pd.notna(wape_percent) else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "wape_percent",
            ascending=True,
        )
        .reset_index(drop=True)
    )
