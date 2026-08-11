"""Reusable validation and data-cleaning utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Confirm that an object is a non-empty pandas DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"{dataframe_name} must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            f"{dataframe_name} is empty."
        )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Confirm that all required columns exist in a DataFrame."""

    validate_dataframe(
        dataframe,
        dataframe_name=dataframe_name,
    )

    required = list(required_columns)

    missing_columns = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_columns}"
        )


def validate_unique_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Confirm that selected columns uniquely identify each row."""

    selected_columns = list(columns)

    validate_required_columns(
        dataframe,
        selected_columns,
        dataframe_name=dataframe_name,
    )

    duplicate_mask = dataframe.duplicated(
        subset=selected_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            f"{dataframe_name} contains "
            f"{duplicate_count:,} duplicated rows based on "
            f"{selected_columns}."
        )


def standardize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with lowercase underscore-separated column names."""

    validated_dataframe = dataframe.copy()

    validated_dataframe.columns = (
        validated_dataframe.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    return validated_dataframe


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    errors: str = "coerce",
) -> pd.DataFrame:
    """Convert selected columns to numeric values."""

    converted_dataframe = dataframe.copy()
    selected_columns = list(columns)

    validate_required_columns(
        converted_dataframe,
        selected_columns,
    )

    for column in selected_columns:
        converted_dataframe[column] = pd.to_numeric(
            converted_dataframe[column],
            errors=errors,
        )

    return converted_dataframe


def convert_datetime_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    errors: str = "coerce",
) -> pd.DataFrame:
    """Convert selected columns to pandas datetime values."""

    converted_dataframe = dataframe.copy()
    selected_columns = list(columns)

    validate_required_columns(
        converted_dataframe,
        selected_columns,
    )

    for column in selected_columns:
        converted_dataframe[column] = pd.to_datetime(
            converted_dataframe[column],
            errors=errors,
        )

    return converted_dataframe


def validate_non_negative_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    allow_missing: bool = True,
    dataframe_name: str = "dataframe",
) -> None:
    """Confirm that selected numeric columns contain no negative values."""

    selected_columns = list(columns)

    validate_required_columns(
        dataframe,
        selected_columns,
        dataframe_name=dataframe_name,
    )

    for column in selected_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if not allow_missing and numeric_values.isna().any():
            missing_count = int(numeric_values.isna().sum())

            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{missing_count:,} missing or non-numeric values."
            )

        negative_mask = numeric_values < 0

        if negative_mask.any():
            negative_count = int(negative_mask.sum())

            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{negative_count:,} negative values."
            )


def validate_positive_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    allow_missing: bool = True,
    dataframe_name: str = "dataframe",
) -> None:
    """Confirm that selected columns contain values greater than zero."""

    selected_columns = list(columns)

    validate_required_columns(
        dataframe,
        selected_columns,
        dataframe_name=dataframe_name,
    )

    for column in selected_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if not allow_missing and numeric_values.isna().any():
            missing_count = int(numeric_values.isna().sum())

            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{missing_count:,} missing or non-numeric values."
            )

        invalid_mask = numeric_values <= 0

        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())

            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{invalid_count:,} values that are not positive."
            )


def dataframe_quality_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a column-level data-quality summary."""

    validate_dataframe(dataframe)

    row_count = len(dataframe)

    summary = pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": [
                str(dataframe[column].dtype)
                for column in dataframe.columns
            ],
            "non_null_count": [
                int(dataframe[column].notna().sum())
                for column in dataframe.columns
            ],
            "missing_count": [
                int(dataframe[column].isna().sum())
                for column in dataframe.columns
            ],
            "missing_percent": [
                round(
                    dataframe[column].isna().mean() * 100,
                    2,
                )
                for column in dataframe.columns
            ],
            "unique_count": [
                int(dataframe[column].nunique(dropna=True))
                for column in dataframe.columns
            ],
        }
    )

    summary["row_count"] = row_count

    return summary[
        [
            "column",
            "dtype",
            "row_count",
            "non_null_count",
            "missing_count",
            "missing_percent",
            "unique_count",
        ]
    ]


def validate_binary_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> pd.DataFrame:
    """
    Validate that columns contain only 0, 1, or missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset
    columns : Iterable[str]
        Column names to validate as binary
    dataframe_name : str, default='dataframe'
        Name for error messages

    Returns
    -------
    pd.DataFrame
        Summary with columns: column, invalid_count

    Raises
    ------
    ValueError
        If any column has non-binary values
    """
    columns_list = list(columns)
    validate_required_columns(
        dataframe,
        columns_list,
        dataframe_name=dataframe_name,
    )

    validation_rows = []

    for column in columns_list:
        invalid_mask = dataframe[column].notna() & ~dataframe[column].isin(
            [0, 1]
        )

        invalid_count = int(invalid_mask.sum())

        validation_rows.append(
            {
                "column": column,
                "invalid_count": invalid_count,
            }
        )

        if invalid_count > 0:
            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{invalid_count:,} non-binary values (expected 0, 1, or missing)"
            )

    return pd.DataFrame(validation_rows)


def validate_bounded_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    *,
    inclusive: str = "both",
    dataframe_name: str = "dataframe",
) -> pd.DataFrame:
    """
    Validate that columns are within specified bounds.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset
    columns : Iterable[str]
        Column names to validate
    lower_bound : float, default=0.0
        Lower bound for valid values
    upper_bound : float, default=1.0
        Upper bound for valid values
    inclusive : str, default='both'
        Whether bounds are inclusive ('both', 'neither', 'left', 'right')
    dataframe_name : str, default='dataframe'
        Name for error messages

    Returns
    -------
    pd.DataFrame
        Summary with columns: column, expected_range, invalid_count

    Raises
    ------
    ValueError
        If any column has out-of-bounds values
    """
    columns_list = list(columns)
    validate_required_columns(
        dataframe,
        columns_list,
        dataframe_name=dataframe_name,
    )

    validation_rows = []

    for column in columns_list:
        invalid_mask = dataframe[column].notna() & ~dataframe[column].between(
            lower_bound,
            upper_bound,
            inclusive=inclusive,
        )

        invalid_count = int(invalid_mask.sum())

        validation_rows.append(
            {
                "column": column,
                "expected_range": f"[{lower_bound}, {upper_bound}]",
                "invalid_count": invalid_count,
            }
        )

        if invalid_count > 0:
            raise ValueError(
                f"{dataframe_name}.{column} contains "
                f"{invalid_count:,} out-of-bounds values "
                f"(expected [{lower_bound}, {upper_bound}])"
            )

    return pd.DataFrame(validation_rows)


def build_consistency_audit(
    dataframe: pd.DataFrame,
    checks: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Run consistency checks on a dataset and return audit results.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to audit
    checks : list[dict] | None
        List of check definitions, each with:
        - name: str - Check description
        - condition: pd.Series[bool] - Boolean mask of violations
        - severity: str - 'critical' or 'review'

    Returns
    -------
    pd.DataFrame
        Audit results with columns: check, issue_count, severity

    Examples
    --------
    >>> df = pd.DataFrame({"price": [10, 20], "cost": [5, 25]})
    >>> checks = [
    ...     {
    ...         "name": "Cost exceeds price",
    ...         "condition": df["cost"] > df["price"],
    ...         "severity": "critical"
    ...     }
    ... ]
    >>> audit = build_consistency_audit(df, checks)
    """
    if checks is None:
        checks = []

    records = []

    for check_def in checks:
        records.append(
            {
                "check": check_def["name"],
                "issue_count": int(check_def["condition"].sum()),
                "severity": check_def.get("severity", "review"),
            }
        )

    return pd.DataFrame(records)