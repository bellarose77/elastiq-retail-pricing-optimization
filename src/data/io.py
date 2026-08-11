"""Reusable functions for loading and saving project data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PathLike = str | Path


def ensure_parent_directory(file_path: PathLike) -> Path:
    """Create the parent directory of a file and return its Path."""

    path = Path(file_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def require_file(file_path: PathLike) -> Path:
    """Confirm that a required file exists."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "Required file was not found:\n"
            f"{path.resolve()}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            "Expected a file but found another object:\n"
            f"{path.resolve()}"
        )

    return path


def load_csv(
    file_path: PathLike,
    *,
    parse_dates: list[str] | None = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV file after confirming that it exists."""

    path = require_file(file_path)

    dataframe = pd.read_csv(
        path,
        parse_dates=parse_dates,
        low_memory=low_memory,
        **kwargs,
    )

    return dataframe


def save_csv(
    dataframe: pd.DataFrame,
    file_path: PathLike,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Save a DataFrame as a CSV file."""

    path = ensure_parent_directory(file_path)

    dataframe.to_csv(
        path,
        index=index,
        **kwargs,
    )

    return path


def load_json(file_path: PathLike) -> dict[str, Any]:
    """Load a JSON configuration or results file."""

    path = require_file(file_path)

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "The JSON file must contain a dictionary-like object."
        )

    return data


def save_json(
    data: dict[str, Any],
    file_path: PathLike,
    *,
    indent: int = 4,
) -> Path:
    """Save a dictionary as a JSON file."""

    path = ensure_parent_directory(file_path)

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    return path


def load_model(file_path: PathLike) -> Any:
    """Load a model or Python object saved with joblib."""

    path = require_file(file_path)

    return joblib.load(path)


def save_model(
    model: Any,
    file_path: PathLike,
) -> Path:
    """Save a model or Python object with joblib."""

    path = ensure_parent_directory(file_path)

    joblib.dump(
        model,
        path,
    )

    return path