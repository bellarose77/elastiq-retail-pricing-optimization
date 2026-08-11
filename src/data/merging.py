"""
Data Merging Functions

This module provides functions for merging datasets with temporal
awareness and leakage prevention.
"""

from __future__ import annotations

import pandas as pd


def merge_temporal_features(
    target_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    on: list[str],
    date_column: str,
    feature_columns: list[str] | None = None,
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge features with temporal awareness to prevent data leakage.

    This function ensures that features are merged based on exact date
    matches to prevent future information from leaking into past records.

    Parameters
    ----------
    target_df : pd.DataFrame
        Target dataset to merge features into
    feature_df : pd.DataFrame
        Feature dataset to merge from
    on : list[str]
        Column names to join on (must include date_column)
    date_column : str
        Name of the date column for temporal validation
    feature_columns : list[str] | None
        Specific feature columns to merge. If None, merge all columns
        from feature_df except join keys
    how : str, default='left'
        Type of merge (left, right, inner, outer)

    Returns
    -------
    pd.DataFrame
        Merged dataset with features

    Raises
    ------
    ValueError
        If date_column not in 'on' list

    Examples
    --------
    >>> target = pd.DataFrame({
    ...     "date": pd.date_range("2023-01-01", periods=3),
    ...     "product": ["A", "B", "C"],
    ...     "sales": [10, 20, 30],
    ... })
    >>> features = pd.DataFrame({
    ...     "date": pd.date_range("2023-01-01", periods=3),
    ...     "product": ["A", "B", "C"],
    ...     "feature": [1.0, 2.0, 3.0],
    ... })
    >>> result = merge_temporal_features(
    ...     target, features, on=["date", "product"], date_column="date"
    ... )
    """
    if date_column not in on:
        raise ValueError(
            f"date_column '{date_column}' must be in 'on' list for temporal merge"
        )

    # Select columns to merge
    if feature_columns is None:
        feature_columns = [
            col for col in feature_df.columns if col not in on
        ]

    merge_cols = on + [
        col for col in feature_columns if col not in on
    ]
    feature_subset = feature_df[merge_cols].copy()

    # Perform the merge
    result = target.merge(
        feature_subset,
        on=on,
        how=how,
        validate="m:1",
    )

    return result
