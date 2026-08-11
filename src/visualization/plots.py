# %%
"""Reusable visualization functions for retail analytics."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data.validation import validate_required_columns


def format_percent_axis(
    axis: Axes,
    *,
    decimals: int = 0,
) -> Axes:
    """Format the vertical axis as percentages."""

    from matplotlib.ticker import PercentFormatter

    axis.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=decimals,
        )
    )

    return axis


def format_currency_axis(
    axis: Axes,
    *,
    symbol: str = "$",
    decimals: int = 0,
) -> Axes:
    """Format the vertical axis as currency."""

    from matplotlib.ticker import StrMethodFormatter

    axis.yaxis.set_major_formatter(
        StrMethodFormatter(
            symbol
            + "{x:,.%df}" % decimals
        )
    )

    return axis


def finalize_figure(
    figure: Figure,
    *,
    tight_layout: bool = True,
) -> Figure:
    """Apply final layout settings to a Matplotlib figure."""

    if tight_layout:
        figure.tight_layout()

    return figure


# %%
def plot_price_scenarios(
    scenarios: pd.DataFrame,
    *,
    objective_column: str = "expected_profit",
    title: str = "Price Optimization Scenarios",
    current_price: float | None = None,
    recommended_price: float | None = None,
    figure_size: tuple[float, float] = (
        10.0,
        6.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot the optimization objective across candidate prices."""

    validate_required_columns(
        scenarios,
        [
            "candidate_price",
            objective_column,
        ],
        dataframe_name="price scenarios",
    )

    plot_data = scenarios[
        [
            "candidate_price",
            objective_column,
        ]
    ].copy()

    plot_data["candidate_price"] = pd.to_numeric(
        plot_data["candidate_price"],
        errors="coerce",
    )

    plot_data[objective_column] = pd.to_numeric(
        plot_data[objective_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            "candidate_price"
        )
    )

    if plot_data.empty:
        raise ValueError(
            "No valid price-scenario observations are available."
        )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    axis.plot(
        plot_data["candidate_price"],
        plot_data[objective_column],
        linewidth=2,
        marker="o",
        markersize=3,
    )

    if current_price is not None:
        axis.axvline(
            float(current_price),
            linestyle="--",
            linewidth=1.5,
            label="Current price",
        )

    if recommended_price is not None:
        axis.axvline(
            float(recommended_price),
            linestyle=":",
            linewidth=2,
            label="Recommended price",
        )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Candidate Price"
    )

    axis.set_ylabel(
        objective_column
        .replace(
            "_",
            " ",
        )
        .title()
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    if (
        current_price is not None
        or recommended_price is not None
    ):
        axis.legend()

    finalize_figure(
        figure
    )

    return figure, axis


# %%
def plot_optimization_actions(
    action_summary: pd.DataFrame,
    *,
    action_column: str = "recommendation_action",
    count_column: str = "item_count",
    title: str = "Price Recommendation Actions",
    figure_size: tuple[float, float] = (
        9.0,
        5.5,
    ),
) -> tuple[Figure, Axes]:
    """Plot the number of items associated with each pricing action."""

    validate_required_columns(
        action_summary,
        [
            action_column,
            count_column,
        ],
        dataframe_name="optimization action summary",
    )

    plot_data = action_summary[
        [
            action_column,
            count_column,
        ]
    ].copy()

    plot_data[count_column] = pd.to_numeric(
        plot_data[count_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            count_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid optimization-action data is available."
        )

    readable_actions = (
        plot_data[action_column]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.title()
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    bars = axis.bar(
        readable_actions,
        plot_data[count_column],
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Recommendation Action"
    )

    axis.set_ylabel(
        "Number of Items"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    for bar, value in zip(
        bars,
        plot_data[count_column],
        strict=True,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    finalize_figure(
        figure
    )

    return figure, axis


# %%
def plot_category_optimization_impact(
    category_summary: pd.DataFrame,
    *,
    category_column: str = "category",
    value_column: str = "profit_change",
    title: str = "Expected Optimization Impact by Category",
    figure_size: tuple[float, float] = (
        10.0,
        6.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot the expected financial impact for each product category."""

    validate_required_columns(
        category_summary,
        [
            category_column,
            value_column,
        ],
        dataframe_name="optimization category summary",
    )

    plot_data = category_summary[
        [
            category_column,
            value_column,
        ]
    ].copy()

    plot_data[value_column] = pd.to_numeric(
        plot_data[value_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            value_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid category optimization data is available."
        )

    readable_categories = (
        plot_data[category_column]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.title()
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    bars = axis.barh(
        readable_categories,
        plot_data[value_column],
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        value_column
        .replace(
            "_",
            " ",
        )
        .title()
    )

    axis.set_ylabel(
        "Category"
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        plot_data[value_column],
        strict=True,
    ):
        horizontal_position = (
            value
            if value >= 0
            else 0
        )

        alignment = (
            "left"
            if value >= 0
            else "right"
        )

        axis.text(
            horizontal_position,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:,.2f} ",
            ha=alignment,
            va="center",
        )

    finalize_figure(
        figure
    )

    return figure, axis


# %%
def plot_price_recommendations(
    recommendations: pd.DataFrame,
    *,
    item_column: str = "item_id",
    current_price_column: str = "current_price",
    recommended_price_column: str = "recommended_price",
    top_n: int = 20,
    title: str = "Current vs Recommended Prices",
    figure_size: tuple[float, float] = (
        12.0,
        7.0,
    ),
) -> tuple[Figure, Axes]:
    """Compare current and recommended prices for selected items."""

    validate_required_columns(
        recommendations,
        [
            item_column,
            current_price_column,
            recommended_price_column,
        ],
        dataframe_name="price recommendations",
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    plot_data = recommendations[
        [
            item_column,
            current_price_column,
            recommended_price_column,
        ]
    ].copy()

    plot_data[current_price_column] = pd.to_numeric(
        plot_data[current_price_column],
        errors="coerce",
    )

    plot_data[recommended_price_column] = pd.to_numeric(
        plot_data[recommended_price_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    plot_data["absolute_price_change"] = (
        plot_data[recommended_price_column]
        - plot_data[current_price_column]
    ).abs()

    plot_data = (
        plot_data
        .sort_values(
            "absolute_price_change",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            recommended_price_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid price recommendations are available."
        )

    item_positions = np.arange(
        len(plot_data)
    )

    bar_height = 0.36

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    axis.barh(
        item_positions - bar_height / 2,
        plot_data[current_price_column],
        height=bar_height,
        label="Current price",
    )

    axis.barh(
        item_positions + bar_height / 2,
        plot_data[recommended_price_column],
        height=bar_height,
        label="Recommended price",
    )

    axis.set_yticks(
        item_positions
    )

    axis.set_yticklabels(
        plot_data[item_column].astype(str)
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Price"
    )

    axis.set_ylabel(
        "Item"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    axis.legend()

    finalize_figure(
        figure
    )

    return figure, axis


# %%
def plot_forecast_performance(
    forecast_data: pd.DataFrame,
    *,
    date_column: str = "date",
    actual_column: str = "quantity",
    predicted_column: str = "predicted_value",
    title: str = "Actual vs Predicted Demand",
    figure_size: tuple[float, float] = (
        12.0,
        6.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot actual and predicted demand across time."""

    validate_required_columns(
        forecast_data,
        [
            date_column,
            actual_column,
            predicted_column,
        ],
        dataframe_name="forecast results",
    )

    plot_data = forecast_data[
        [
            date_column,
            actual_column,
            predicted_column,
        ]
    ].copy()

    plot_data[date_column] = pd.to_datetime(
        plot_data[date_column],
        errors="coerce",
    )

    plot_data[actual_column] = pd.to_numeric(
        plot_data[actual_column],
        errors="coerce",
    )

    plot_data[predicted_column] = pd.to_numeric(
        plot_data[predicted_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            date_column,
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid forecasting observations are available."
        )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    axis.plot(
        plot_data[date_column],
        plot_data[actual_column],
        linewidth=2,
        label="Actual",
    )

    axis.plot(
        plot_data[date_column],
        plot_data[predicted_column],
        linewidth=2,
        linestyle="--",
        label="Predicted",
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Demand"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.autofmt_xdate()

    finalize_figure(
        figure
    )

    return figure, axis

# %%
def plot_forecast_residuals(
    forecast_data: pd.DataFrame,
    *,
    predicted_column: str = "predicted_value",
    residual_column: str = "residual",
    title: str = "Forecast Residual Analysis",
    figure_size: tuple[float, float] = (
        10.0,
        6.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot forecast residuals against predicted demand."""

    validate_required_columns(
        forecast_data,
        [
            predicted_column,
            residual_column,
        ],
        dataframe_name="forecast results",
    )

    plot_data = forecast_data[
        [
            predicted_column,
            residual_column,
        ]
    ].copy()

    plot_data[predicted_column] = pd.to_numeric(
        plot_data[predicted_column],
        errors="coerce",
    )

    plot_data[residual_column] = pd.to_numeric(
        plot_data[residual_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid forecast residual observations are available."
        )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    axis.scatter(
        plot_data[predicted_column],
        plot_data[residual_column],
        alpha=0.6,
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Predicted Demand"
    )

    axis.set_ylabel(
        "Residual"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    finalize_figure(
        figure
    )

    return figure, axis

# %%
def plot_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 15,
    title: str = "Forecasting Feature Importance",
    figure_size: tuple[float, float] = (
        10.0,
        7.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot the most influential forecasting features."""

    validate_required_columns(
        feature_importance,
        [
            feature_column,
            importance_column,
        ],
        dataframe_name="feature importance",
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    plot_data = feature_importance[
        [
            feature_column,
            importance_column,
        ]
    ].copy()

    plot_data[importance_column] = pd.to_numeric(
        plot_data[importance_column],
        errors="coerce",
    )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            importance_column,
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            importance_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid feature-importance data is available."
        )

    readable_features = (
        plot_data[feature_column]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.title()
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    bars = axis.barh(
        readable_features,
        plot_data[importance_column],
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        plot_data[importance_column],
        strict=True,
    ):
        axis.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.4f}",
            ha="left",
            va="center",
        )

    finalize_figure(
        figure
    )

    return figure, axis

# %%
def plot_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 15,
    title: str = "Forecasting Feature Importance",
    figure_size: tuple[float, float] = (
        10.0,
        7.0,
    ),
) -> tuple[Figure, Axes]:
    """Plot the most influential forecasting features."""

    validate_required_columns(
        feature_importance,
        [
            feature_column,
            importance_column,
        ],
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    plot_data = feature_importance[
        [
            feature_column,
            importance_column,
        ]
    ].copy()

    plot_data[feature_column] = (
        plot_data[feature_column]
        .astype(str)
        .str.strip()
    )

    plot_data[importance_column] = pd.to_numeric(
        plot_data[importance_column],
        errors="coerce",
    )

    plot_data = plot_data.dropna(
        subset=[
            feature_column,
            importance_column,
        ]
    )

    plot_data = plot_data[
        plot_data[feature_column] != ""
    ]

    if plot_data.empty:
        raise ValueError(
            "No valid feature-importance records "
            "are available for plotting."
        )

    plot_data = (
        plot_data.sort_values(
            importance_column,
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            importance_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    figure, axis = plt.subplots(
        figsize=figure_size,
    )

    bars = axis.barh(
        plot_data[feature_column],
        plot_data[importance_column],
    )

    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.set_axisbelow(True)

    maximum_importance = float(
        plot_data[importance_column].abs().max()
    )

    label_offset = (
        maximum_importance * 0.01
        if maximum_importance > 0
        else 0.01
    )

    for bar, importance in zip(
        bars,
        plot_data[importance_column],
        strict=False,
    ):
        axis.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.4f}",
            va="center",
            fontsize=9,
        )

    finalize_figure(figure)

    return figure, axis