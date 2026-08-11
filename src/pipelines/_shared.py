"""
Shared Pipeline Utilities

This module provides common utilities for pipeline scripts,
including dependency checking and data loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_csv


def check_pipeline_dependencies(
    required_files: dict[str, Path],
    pipeline_name: str = "pipeline",
) -> None:
    """
    Verify that all required upstream pipeline outputs exist.

    Parameters
    ----------
    required_files : dict[str, Path]
        Mapping of dependency names to file paths
    pipeline_name : str, default='pipeline'
        Name of the pipeline for error messages

    Raises
    ------
    FileNotFoundError
        If any required file is missing

    Examples
    --------
    >>> from pathlib import Path
    >>> deps = {
    ...     "retail_data": Path("data/interim/retail_clean.csv"),
    ...     "elasticities": Path("data/processed/elasticities.csv"),
    ... }
    >>> check_pipeline_dependencies(deps, "optimization")
    """
    missing_files = {
        name: path
        for name, path in required_files.items()
        if not path.exists()
    }

    if missing_files:
        missing_list = "\n".join(
            f"  - {name}: {path}" for name, path in missing_files.items()
        )

        raise FileNotFoundError(
            f"{pipeline_name} requires the following upstream outputs "
            f"that are missing:\n{missing_list}\n\n"
            f"Please run the prerequisite pipeline steps first."
        )


def load_pipeline_outputs(
    file_mapping: dict[str, Path],
    parse_dates: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load multiple pipeline output files into a dictionary of DataFrames.

    Parameters
    ----------
    file_mapping : dict[str, Path]
        Mapping of dataset names to file paths
    parse_dates : list[str] | None
        Column names to parse as dates (applied to all files)

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of dataset names to loaded DataFrames

    Examples
    --------
    >>> from pathlib import Path
    >>> files = {
    ...     "retail": Path("data/interim/retail_clean.csv"),
    ...     "elasticities": Path("data/processed/elasticities.csv"),
    ... }
    >>> datasets = load_pipeline_outputs(files, parse_dates=["date"])
    >>> retail_df = datasets["retail"]
    """
    datasets = {}

    for name, file_path in file_mapping.items():
        datasets[name] = load_csv(
            file_path,
            parse_dates=parse_dates,
            low_memory=False,
        )

    return datasets
