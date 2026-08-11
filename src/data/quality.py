"""
Data Quality and Validation Functions

This module provides functions for building file registries, dataset overviews,
and data quality assessments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_file_registry(
    file_paths: dict[str, tuple[Path, bool]],
    project_root: Path,
) -> pd.DataFrame:
    """
    Build a registry of dataset files with their existence status.

    Parameters
    ----------
    file_paths : dict[str, tuple[Path, bool]]
        Dictionary mapping dataset names to (file_path, required) tuples
    project_root : Path
        Project root directory for computing relative paths

    Returns
    -------
    pd.DataFrame
        Registry with columns: dataset, required, path, exists

    Examples
    --------
    >>> file_paths = {
    ...     "Retail data": (Path("/data/retail.csv"), True),
    ...     "Market notes": (Path("/data/notes.csv"), False),
    ... }
    >>> registry = build_file_registry(file_paths, Path("/project"))
    """
    records = []

    for dataset_name, (file_path, required) in file_paths.items():
        records.append(
            {
                "dataset": dataset_name,
                "required": required,
                "path": str(file_path.relative_to(project_root)),
                "exists": file_path.exists(),
            }
        )

    return pd.DataFrame(records)


def generate_dataset_overview(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate statistical overview of multiple datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary mapping dataset names to DataFrames

    Returns
    -------
    pd.DataFrame
        Overview with columns: dataset, rows, columns, memory_mb,
        missing_cells, missing_percent

    Examples
    --------
    >>> datasets = {
    ...     "retail": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    ...     "sales": pd.DataFrame({"x": [5, 6, 7]}),
    ... }
    >>> overview = generate_dataset_overview(datasets)
    """
    records = []

    for dataset_name, df in datasets.items():
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isna().sum().sum())

        records.append(
            {
                "dataset": dataset_name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "missing_cells": missing_cells,
                "missing_percent": (
                    100.0 * missing_cells / total_cells
                    if total_cells > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_missing_values(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Summarize missing values by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    dataset_name : str, default='dataset'
        Name of the dataset for the summary

    Returns
    -------
    pd.DataFrame
        Missing value summary with columns: dataset, column,
        missing_count, missing_percent

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    >>> summary = summarize_missing_values(df, "retail")
    """
    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (100.0 * df.isna().sum() / len(df)).values,
        }
    )

    missing_summary.insert(0, "dataset", dataset_name)

    return missing_summary[missing_summary["missing_count"] > 0].sort_values(
        "missing_count",
        ascending=False,
    )
