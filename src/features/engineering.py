"""Reusable feature-engineering functions for retail modelling."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.validation import validate_required_columns
from src.features.text import normalize_category_key  # noqa: F401 (re-exported)


def normalize_name(value: object) -> str:
    """Convert a value into a lowercase underscore-separated name."""

    if pd.isna(value):
        return "unknown"

    normalized_value = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")

    return normalized_value or "unknown"


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_value: float = 0.0,
) -> pd.Series:
    """Divide two Series while safely handling zero and missing values."""

    numeric_numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    numeric_denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = numeric_numerator.div(
        numeric_denominator.replace(0, np.nan)
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result.fillna(fill_value)


def add_calendar_features(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add calendar-based features from a date column."""

    validate_required_columns(
        dataframe,
        [date_column],
    )

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["year"] = result[date_column].dt.year
    result["quarter"] = result[date_column].dt.quarter
    result["month"] = result[date_column].dt.month
    result["week_of_year"] = (
        result[date_column]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )
    result["day_of_month"] = result[date_column].dt.day
    result["day_of_week"] = result[date_column].dt.dayofweek
    result["day_name"] = result[date_column].dt.day_name()

    result["is_weekend"] = (
        result["day_of_week"]
        .isin([5, 6])
        .astype("int8")
    )

    result["is_month_start"] = (
        result[date_column]
        .dt.is_month_start
        .astype("int8")
    )

    result["is_month_end"] = (
        result[date_column]
        .dt.is_month_end
        .astype("int8")
    )

    result["days_from_start"] = (
        result[date_column] - result[date_column].min()
    ).dt.days

    return result


def add_price_features(
    dataframe: pd.DataFrame,
    *,
    price_column: str = "price",
    reference_price_column: str | None = None,
    cost_column: str | None = None,
) -> pd.DataFrame:
    """Add price, discount, margin, and markup features."""

    required_columns = [price_column]

    if reference_price_column is not None:
        required_columns.append(reference_price_column)

    if cost_column is not None:
        required_columns.append(cost_column)

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = dataframe.copy()

    result[price_column] = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    result["log_price"] = np.log1p(
        result[price_column].clip(lower=0)
    )

    if reference_price_column is not None:
        result[reference_price_column] = pd.to_numeric(
            result[reference_price_column],
            errors="coerce",
        )

        result["price_difference"] = (
            result[price_column]
            - result[reference_price_column]
        )

        result["price_ratio"] = safe_divide(
            result[price_column],
            result[reference_price_column],
            fill_value=1.0,
        )

        result["discount_amount"] = (
            result[reference_price_column]
            - result[price_column]
        ).clip(lower=0)

        result["discount_rate"] = safe_divide(
            result["discount_amount"],
            result[reference_price_column],
        )

        result["is_discounted"] = (
            result["discount_amount"] > 0
        ).astype("int8")

    if cost_column is not None:
        result[cost_column] = pd.to_numeric(
            result[cost_column],
            errors="coerce",
        )

        result["unit_margin"] = (
            result[price_column]
            - result[cost_column]
        )

        result["margin_rate"] = safe_divide(
            result["unit_margin"],
            result[price_column],
        )

        result["markup_rate"] = safe_divide(
            result["unit_margin"],
            result[cost_column],
        )

    return result


def add_demand_features(
    dataframe: pd.DataFrame,
    *,
    quantity_column: str = "quantity",
    price_column: str = "price",
    cost_column: str | None = None,
) -> pd.DataFrame:
    """Add revenue, demand, and profit-related features."""

    required_columns = [
        quantity_column,
        price_column,
    ]

    if cost_column is not None:
        required_columns.append(cost_column)

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = dataframe.copy()

    result[quantity_column] = pd.to_numeric(
        result[quantity_column],
        errors="coerce",
    )

    result[price_column] = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    result["revenue"] = (
        result[quantity_column]
        * result[price_column]
    )

    result["log_quantity"] = np.log1p(
        result[quantity_column].clip(lower=0)
    )

    if cost_column is not None:
        result[cost_column] = pd.to_numeric(
            result[cost_column],
            errors="coerce",
        )

        result["total_cost"] = (
            result[quantity_column]
            * result[cost_column]
        )

        result["gross_profit"] = (
            result["revenue"]
            - result["total_cost"]
        )

        result["gross_margin_rate"] = safe_divide(
            result["gross_profit"],
            result["revenue"],
        )

    return result


def add_promotion_features(
    dataframe: pd.DataFrame,
    *,
    promotion_column: str = "promotion",
) -> pd.DataFrame:
    """Standardize a promotion indicator and add promotion features."""

    validate_required_columns(
        dataframe,
        [promotion_column],
    )

    result = dataframe.copy()

    promotion_values = result[promotion_column]

    if pd.api.types.is_bool_dtype(promotion_values):
        promotion_indicator = promotion_values.astype("int8")

    elif pd.api.types.is_numeric_dtype(promotion_values):
        promotion_indicator = (
            pd.to_numeric(
                promotion_values,
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype("int8")
        )

    else:
        normalized_values = (
            promotion_values
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        positive_values = {
            "1",
            "true",
            "yes",
            "y",
            "promotion",
            "promoted",
            "promo",
            "active",
        }

        promotion_indicator = (
            normalized_values
            .isin(positive_values)
            .astype("int8")
        )

    result["is_promotion"] = promotion_indicator

    return result


def add_lag_features(
    dataframe: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    group_columns: Sequence[str],
    date_column: str = "date",
    lags: Iterable[int] = (1, 7, 14, 28),
) -> pd.DataFrame:
    """Add grouped lag features for selected numeric columns."""

    required_columns = [
        *group_columns,
        date_column,
        *value_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["_original_row_order"] = np.arange(len(result))

    result = result.sort_values(
        [*group_columns, date_column],
        kind="stable",
    )

    grouped = result.groupby(
        list(group_columns),
        dropna=False,
        sort=False,
    )

    for value_column in value_columns:
        numeric_values = pd.to_numeric(
            result[value_column],
            errors="coerce",
        )

        result[value_column] = numeric_values

        for lag in lags:
            if lag <= 0:
                raise ValueError(
                    "Lag values must be greater than zero."
                )

            result[f"{value_column}_lag_{lag}"] = (
                grouped[value_column]
                .shift(lag)
            )

    result = (
        result
        .sort_values("_original_row_order")
        .drop(columns="_original_row_order")
        .reset_index(drop=True)
    )

    return result


def add_rolling_features(
    dataframe: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    group_columns: Sequence[str],
    date_column: str = "date",
    windows: Iterable[int] = (7, 14, 28),
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add leakage-safe rolling averages and standard deviations."""

    required_columns = [
        *group_columns,
        date_column,
        *value_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["_original_row_order"] = np.arange(len(result))

    result = result.sort_values(
        [*group_columns, date_column],
        kind="stable",
    )

    for value_column in value_columns:
        result[value_column] = pd.to_numeric(
            result[value_column],
            errors="coerce",
        )

        shifted_values = (
            result.groupby(
                list(group_columns),
                dropna=False,
                sort=False,
            )[value_column]
            .shift(1)
        )

        for window in windows:
            if window <= 0:
                raise ValueError(
                    "Rolling window values must be greater than zero."
                )

            grouped_shifted = shifted_values.groupby(
                [
                    result[column]
                    for column in group_columns
                ],
                dropna=False,
                sort=False,
            )

            result[
                f"{value_column}_rolling_mean_{window}"
            ] = grouped_shifted.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=min_periods,
                ).mean()
            )

            result[
                f"{value_column}_rolling_std_{window}"
            ] = grouped_shifted.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=min_periods,
                ).std()
            )

    result = (
        result
        .sort_values("_original_row_order")
        .drop(columns="_original_row_order")
        .reset_index(drop=True)
    )

    return result


def add_price_change_features(
    dataframe: pd.DataFrame,
    *,
    price_column: str = "price",
    group_columns: Sequence[str],
    date_column: str = "date",
) -> pd.DataFrame:
    """Add previous-price and percentage-change features."""

    required_columns = [
        *group_columns,
        date_column,
        price_column,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["_original_row_order"] = np.arange(len(result))

    result = result.sort_values(
        [*group_columns, date_column],
        kind="stable",
    )

    result[price_column] = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    grouped_price = result.groupby(
        list(group_columns),
        dropna=False,
        sort=False,
    )[price_column]

    result["previous_price"] = grouped_price.shift(1)

    result["price_change"] = (
        result[price_column]
        - result["previous_price"]
    )

    result["price_change_rate"] = safe_divide(
        result["price_change"],
        result["previous_price"],
    )

    result["price_increased"] = (
        result["price_change"] > 0
    ).astype("int8")

    result["price_decreased"] = (
        result["price_change"] < 0
    ).astype("int8")

    result = (
        result
        .sort_values("_original_row_order")
        .drop(columns="_original_row_order")
        .reset_index(drop=True)
    )

    return result


def build_retail_features(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "date",
    item_column: str = "item_id",
    quantity_column: str = "quantity",
    price_column: str = "price",
    reference_price_column: str | None = None,
    cost_column: str | None = None,
    promotion_column: str | None = None,
    include_lags: bool = True,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Build a standard modelling dataset for retail analysis."""

    required_columns = [
        date_column,
        item_column,
        quantity_column,
        price_column,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    result = add_calendar_features(
        dataframe,
        date_column=date_column,
    )

    result = add_price_features(
        result,
        price_column=price_column,
        reference_price_column=reference_price_column,
        cost_column=cost_column,
    )

    result = add_demand_features(
        result,
        quantity_column=quantity_column,
        price_column=price_column,
        cost_column=cost_column,
    )

    if promotion_column is not None:
        result = add_promotion_features(
            result,
            promotion_column=promotion_column,
        )

    result = add_price_change_features(
        result,
        price_column=price_column,
        group_columns=[item_column],
        date_column=date_column,
    )

    if include_lags:
        result = add_lag_features(
            result,
            value_columns=[
                quantity_column,
                price_column,
            ],
            group_columns=[item_column],
            date_column=date_column,
        )

    if include_rolling:
        result = add_rolling_features(
            result,
            value_columns=[quantity_column],
            group_columns=[item_column],
            date_column=date_column,
        )

    return result