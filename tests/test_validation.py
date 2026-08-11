"""Unit tests for src/data/validation.py"""

import pandas as pd
import pytest

from src.data.validation import (
    build_consistency_audit,
    convert_datetime_columns,
    convert_numeric_columns,
    dataframe_quality_summary,
    standardize_column_names,
    validate_binary_columns,
    validate_bounded_columns,
    validate_dataframe,
    validate_non_negative_columns,
    validate_positive_columns,
    validate_required_columns,
    validate_unique_columns,
)


class TestValidateDataFrame:
    """Test validate_dataframe function."""

    def test_valid_dataframe_passes(self):
        """Test that a valid dataframe passes validation."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        validate_dataframe(df)  # Should not raise

    def test_empty_dataframe_raises_error(self):
        """Test that an empty dataframe raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="is empty"):
            validate_dataframe(df)

    def test_non_dataframe_raises_type_error(self):
        """Test that a non-DataFrame object raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            validate_dataframe([1, 2, 3])

    def test_custom_dataframe_name_in_error(self):
        """Test that custom dataframe_name appears in error message."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="my_data"):
            validate_dataframe(df, dataframe_name="my_data")


class TestValidateRequiredColumns:
    """Test validate_required_columns function."""

    def test_all_columns_present_passes(self):
        """Test that validation passes when all required columns are present."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_required_columns(df, ["a", "b"])  # Should not raise

    def test_missing_single_column_raises_error(self):
        """Test that missing a single column raises ValueError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_missing_multiple_columns_raises_error(self):
        """Test that missing multiple columns raises ValueError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns.*b.*c"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_empty_required_list_passes(self):
        """Test that empty required columns list passes."""
        df = pd.DataFrame({"a": [1]})
        validate_required_columns(df, [])  # Should not raise


class TestValidateUniqueColumns:
    """Test validate_unique_columns function."""

    def test_unique_columns_passes(self):
        """Test that unique column combinations pass validation."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        validate_unique_columns(df, ["a"])  # Should not raise
        validate_unique_columns(df, ["a", "b"])  # Should not raise

    def test_duplicate_rows_raises_error(self):
        """Test that duplicate rows raise ValueError."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        with pytest.raises(ValueError, match="duplicated rows"):
            validate_unique_columns(df, ["a"])

    def test_missing_column_raises_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_unique_columns(df, ["a", "b"])


class TestStandardizeColumnNames:
    """Test standardize_column_names function."""

    def test_lowercase_conversion(self):
        """Test that column names are converted to lowercase."""
        df = pd.DataFrame({"UPPER": [1], "Mixed": [2]})
        result = standardize_column_names(df)
        assert list(result.columns) == ["upper", "mixed"]

    def test_special_character_replacement(self):
        """Test that special characters are replaced with underscores."""
        df = pd.DataFrame({"col-name": [1], "col.name": [2], "col name": [3]})
        result = standardize_column_names(df)
        assert all("_" in col for col in result.columns)

    def test_strip_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are stripped."""
        df = pd.DataFrame({"_column_": [1]})
        result = standardize_column_names(df)
        assert result.columns[0] == "column"

    def test_original_dataframe_unchanged(self):
        """Test that the original dataframe is not modified."""
        df = pd.DataFrame({"Original": [1]})
        original_columns = df.columns.tolist()
        standardize_column_names(df)
        assert df.columns.tolist() == original_columns


class TestConvertNumericColumns:
    """Test convert_numeric_columns function."""

    def test_string_to_numeric_conversion(self):
        """Test that string numbers are converted to numeric."""
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = convert_numeric_columns(df, ["a"])
        assert pd.api.types.is_numeric_dtype(result["a"])

    def test_coerce_errors_by_default(self):
        """Test that invalid values are coerced to NaN by default."""
        df = pd.DataFrame({"a": ["1", "invalid", "3"]})
        result = convert_numeric_columns(df, ["a"])
        assert result["a"].isna().sum() == 1

    def test_missing_column_raises_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            convert_numeric_columns(df, ["b"])

    def test_original_dataframe_unchanged(self):
        """Test that the original dataframe is not modified."""
        df = pd.DataFrame({"a": ["1", "2"]})
        original_dtypes = df.dtypes.tolist()
        convert_numeric_columns(df, ["a"])
        assert df.dtypes.tolist() == original_dtypes


class TestConvertDatetimeColumns:
    """Test convert_datetime_columns function."""

    def test_string_to_datetime_conversion(self):
        """Test that string dates are converted to datetime."""
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
        result = convert_datetime_columns(df, ["date"])
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_coerce_invalid_dates(self):
        """Test that invalid dates are coerced to NaT."""
        df = pd.DataFrame({"date": ["2023-01-01", "invalid"]})
        result = convert_datetime_columns(df, ["date"])
        assert result["date"].isna().sum() == 1

    def test_original_dataframe_unchanged(self):
        """Test that the original dataframe is not modified."""
        df = pd.DataFrame({"date": ["2023-01-01"]})
        original_dtypes = df.dtypes.tolist()
        convert_datetime_columns(df, ["date"])
        assert df.dtypes.tolist() == original_dtypes


class TestValidateNonNegativeColumns:
    """Test validate_non_negative_columns function."""

    def test_all_non_negative_passes(self):
        """Test that all non-negative values pass validation."""
        df = pd.DataFrame({"a": [0, 1, 2, 3]})
        validate_non_negative_columns(df, ["a"])  # Should not raise

    def test_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        df = pd.DataFrame({"a": [1, -1, 3]})
        with pytest.raises(ValueError, match="negative values"):
            validate_non_negative_columns(df, ["a"])

    def test_missing_values_allowed_by_default(self):
        """Test that missing values are allowed by default."""
        df = pd.DataFrame({"a": [1, None, 3]})
        validate_non_negative_columns(df, ["a"], allow_missing=True)  # Should not raise

    def test_missing_values_raise_error_when_not_allowed(self):
        """Test that missing values raise error when allow_missing=False."""
        df = pd.DataFrame({"a": [1, None, 3]})
        with pytest.raises(ValueError, match="missing or non-numeric"):
            validate_non_negative_columns(df, ["a"], allow_missing=False)


class TestValidatePositiveColumns:
    """Test validate_positive_columns function."""

    def test_all_positive_passes(self):
        """Test that all positive values pass validation."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        validate_positive_columns(df, ["a"])  # Should not raise

    def test_zero_values_raise_error(self):
        """Test that zero values raise ValueError."""
        df = pd.DataFrame({"a": [1, 0, 3]})
        with pytest.raises(ValueError, match="not positive"):
            validate_positive_columns(df, ["a"])

    def test_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        df = pd.DataFrame({"a": [1, -1, 3]})
        with pytest.raises(ValueError, match="not positive"):
            validate_positive_columns(df, ["a"])


class TestDataFrameQualitySummary:
    """Test dataframe_quality_summary function."""

    def test_summary_has_expected_columns(self):
        """Test that summary contains expected columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})
        summary = dataframe_quality_summary(df)
        expected_cols = [
            "column",
            "dtype",
            "row_count",
            "non_null_count",
            "missing_count",
            "missing_percent",
            "unique_count",
        ]
        assert all(col in summary.columns for col in expected_cols)

    def test_summary_row_count(self):
        """Test that summary reports correct row count."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = dataframe_quality_summary(df)
        assert summary["row_count"].iloc[0] == 3

    def test_summary_missing_count(self):
        """Test that summary reports correct missing count."""
        df = pd.DataFrame({"a": [1, None, 3]})
        summary = dataframe_quality_summary(df)
        assert summary["missing_count"].iloc[0] == 1

    def test_summary_unique_count(self):
        """Test that summary reports correct unique count."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        summary = dataframe_quality_summary(df)
        assert summary["unique_count"].iloc[0] == 2


class TestValidateBinaryColumns:
    """Test validate_binary_columns function."""

    def test_binary_columns_pass(self):
        """Test that columns with only 0 and 1 pass validation."""
        df = pd.DataFrame({"flag": [0, 1, 0, 1]})
        result = validate_binary_columns(df, ["flag"])
        assert result["invalid_count"].iloc[0] == 0

    def test_non_binary_values_raise_error(self):
        """Test that non-binary values raise ValueError."""
        df = pd.DataFrame({"flag": [0, 1, 2]})
        with pytest.raises(ValueError, match="non-binary values"):
            validate_binary_columns(df, ["flag"])

    def test_missing_values_allowed(self):
        """Test that missing values are allowed."""
        df = pd.DataFrame({"flag": [0, 1, None]})
        result = validate_binary_columns(df, ["flag"])
        assert result["invalid_count"].iloc[0] == 0


class TestValidateBoundedColumns:
    """Test validate_bounded_columns function."""

    def test_values_within_bounds_pass(self):
        """Test that values within bounds pass validation."""
        df = pd.DataFrame({"rate": [0.0, 0.5, 1.0]})
        result = validate_bounded_columns(df, ["rate"], 0.0, 1.0)
        assert result["invalid_count"].iloc[0] == 0

    def test_values_outside_bounds_raise_error(self):
        """Test that values outside bounds raise ValueError."""
        df = pd.DataFrame({"rate": [0.0, 1.5, 1.0]})
        with pytest.raises(ValueError, match="out-of-bounds values"):
            validate_bounded_columns(df, ["rate"], 0.0, 1.0)

    def test_exclusive_bounds(self):
        """Test that exclusive bounds work correctly."""
        df = pd.DataFrame({"rate": [0.0, 0.5, 1.0]})
        with pytest.raises(ValueError):
            validate_bounded_columns(df, ["rate"], 0.0, 1.0, inclusive="neither")


class TestBuildConsistencyAudit:
    """Test build_consistency_audit function."""

    def test_empty_checks_returns_empty_dataframe(self):
        """Test that empty checks return an empty DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        result = build_consistency_audit(df, checks=[])
        assert len(result) == 0

    def test_single_check_returns_correct_result(self):
        """Test that a single check returns correct results."""
        df = pd.DataFrame({"price": [10, 20], "cost": [5, 25]})
        checks = [
            {
                "name": "Cost exceeds price",
                "condition": df["cost"] > df["price"],
                "severity": "critical",
            }
        ]
        result = build_consistency_audit(df, checks)
        assert len(result) == 1
        assert result["issue_count"].iloc[0] == 1
        assert result["severity"].iloc[0] == "critical"

    def test_multiple_checks(self):
        """Test that multiple checks are processed correctly."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        checks = [
            {"name": "Check 1", "condition": df["a"] > 2, "severity": "critical"},
            {"name": "Check 2", "condition": df["b"] < 5, "severity": "review"},
        ]
        result = build_consistency_audit(df, checks)
        assert len(result) == 2
