"""
Data Splitting Functions

This module provides functions for splitting datasets, particularly
for time-series forecasting scenarios.
"""

from __future__ import annotations

import pandas as pd


def create_chronological_split(
    df: pd.DataFrame,
    date_column: str,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with a date column
    date_column : str
        Name of the date column to use for chronological ordering
    train_fraction : float, default=0.7
        Fraction of data to use for training
    val_fraction : float, default=0.15
        Fraction of data to use for validation

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_data, val_data, test_data)

    Raises
    ------
    ValueError
        If train_fraction + val_fraction >= 1.0

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "date": pd.date_range("2023-01-01", periods=100),
    ...     "value": range(100),
    ... })
    >>> train, val, test = create_chronological_split(df, "date")
    >>> len(train), len(val), len(test)
    (70, 15, 15)
    """
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be less than 1.0"
        )

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' was not found")

    # Split on complete calendar dates, not row positions.  A panel dataset
    # commonly has many products and stores on the same date.  Cutting at a
    # row offset leaks same-day information across train/validation/test and
    # makes the reported holdout accuracy optimistic.
    df_sorted = df.copy()
    df_sorted[date_column] = pd.to_datetime(
        df_sorted[date_column], errors="coerce"
    )

    if df_sorted[date_column].isna().any():
        raise ValueError(f"Date column '{date_column}' contains invalid values")

    df_sorted = df_sorted.sort_values(date_column).reset_index(drop=True)
    unique_dates = pd.Index(df_sorted[date_column].drop_duplicates())

    if len(unique_dates) < 3:
        raise ValueError(
            "At least three distinct dates are required for a "
            "train/validation/test split"
        )

    train_date_count = max(1, int(len(unique_dates) * train_fraction))
    validation_date_count = max(1, int(len(unique_dates) * val_fraction))

    if train_date_count + validation_date_count >= len(unique_dates):
        validation_date_count = 1
        train_date_count = len(unique_dates) - 2

    validation_start = unique_dates[train_date_count]
    test_start = unique_dates[train_date_count + validation_date_count]

    train_data = df_sorted.loc[
        df_sorted[date_column] < validation_start
    ].copy()
    val_data = df_sorted.loc[
        df_sorted[date_column].ge(validation_start)
        & df_sorted[date_column].lt(test_start)
    ].copy()
    test_data = df_sorted.loc[
        df_sorted[date_column] >= test_start
    ].copy()

    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError("Chronological split produced an empty partition")

    return train_data, val_data, test_data
