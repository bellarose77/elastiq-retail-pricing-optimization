"""Unit tests for src/features/engineering.py"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_calendar_features,
    add_demand_features,
    add_lag_features,
    add_price_change_features,
    add_price_features,
    add_promotion_features,
    add_rolling_features,
    normalize_name,
    safe_divide,
)


class TestNormalizeName:
    """Test normalize_name function."""

    def test_lowercase_conversion(self):
        """Test that values are converted to lowercase."""
        assert normalize_name("UPPER") == "upper"
        assert normalize_name("MixedCase") == "mixedcase"

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores."""
        assert normalize_name("hello-world") == "hello_world"
        assert normalize_name("hello world") == "hello_world"
        assert normalize_name("hello.world") == "hello_world"

    def test_nan_returns_unknown(self):
        """Test that NaN returns 'unknown'."""
        assert normalize_name(np.nan) == "unknown"
        assert normalize_name(pd.NA) == "unknown"

    def test_empty_string_returns_unknown(self):
        """Test that empty string returns 'unknown'."""
        assert normalize_name("") == "unknown"
        assert normalize_name("   ") == "unknown"


class TestSafeDivide:
    """Test safe_divide function."""

    def test_normal_division(self):
        """Test normal division."""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 4, 5])
        result = safe_divide(numerator, denominator)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_divide_by_zero_returns_fill_value(self):
        """Test that division by zero returns fill value."""
        numerator = pd.Series([10, 20])
        denominator = pd.Series([2, 0])
        result = safe_divide(numerator, denominator, fill_value=0.0)
        assert result.iloc[1] == 0.0

    def test_handles_missing_values(self):
        """Test that missing values are handled."""
        numerator = pd.Series([10, None, 30])
        denominator = pd.Series([2, 4, 5])
        result = safe_divide(numerator, denominator, fill_value=0.0)
        assert result.iloc[1] == 0.0

    def test_infinity_replaced_with_fill_value(self):
        """Test that infinity is replaced with fill value."""
        numerator = pd.Series([10, 20])
        denominator = pd.Series([0, 2])
        result = safe_divide(numerator, denominator, fill_value=-1.0)
        assert result.iloc[0] == -1.0

    def test_string_numbers_converted(self):
        """Test that string numbers are converted."""
        numerator = pd.Series(["10", "20"])
        denominator = pd.Series(["2", "4"])
        result = safe_divide(numerator, denominator)
        assert result.iloc[0] == 5.0


class TestAddCalendarFeatures:
    """Test add_calendar_features function."""

    def test_adds_calendar_columns(self):
        """Test that calendar features are added."""
        df = pd.DataFrame({"date": ["2023-01-15", "2023-02-20"]})
        result = add_calendar_features(df)

        expected_cols = [
            "year",
            "quarter",
            "month",
            "week_of_year",
            "day_of_month",
            "day_of_week",
            "day_name",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "days_from_start",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_year_extraction(self):
        """Test that year is correctly extracted."""
        df = pd.DataFrame({"date": ["2023-01-15"]})
        result = add_calendar_features(df)
        assert result["year"].iloc[0] == 2023

    def test_month_extraction(self):
        """Test that month is correctly extracted."""
        df = pd.DataFrame({"date": ["2023-02-15"]})
        result = add_calendar_features(df)
        assert result["month"].iloc[0] == 2

    def test_weekend_flag(self):
        """Test that weekend flag is correct."""
        # 2023-01-14 is Saturday, 2023-01-15 is Sunday, 2023-01-16 is Monday
        df = pd.DataFrame({"date": ["2023-01-14", "2023-01-15", "2023-01-16"]})
        result = add_calendar_features(df)
        assert result["is_weekend"].iloc[0] == 1  # Saturday
        assert result["is_weekend"].iloc[1] == 1  # Sunday
        assert result["is_weekend"].iloc[2] == 0  # Monday

    def test_days_from_start(self):
        """Test that days from start is calculated correctly."""
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-08"]})
        result = add_calendar_features(df)
        assert result["days_from_start"].iloc[0] == 0
        assert result["days_from_start"].iloc[1] == 7

    def test_custom_date_column(self):
        """Test that custom date column name works."""
        df = pd.DataFrame({"my_date": ["2023-01-15"]})
        result = add_calendar_features(df, date_column="my_date")
        assert "year" in result.columns


class TestAddPriceFeatures:
    """Test add_price_features function."""

    def test_adds_log_price(self):
        """Test that log price is added."""
        df = pd.DataFrame({"price": [10, 20, 30]})
        result = add_price_features(df)
        assert "log_price" in result.columns
        assert all(result["log_price"] > 0)

    def test_adds_price_difference(self):
        """Test that price difference is calculated."""
        df = pd.DataFrame({"price": [10, 20], "reference_price": [12, 18]})
        result = add_price_features(df, reference_price_column="reference_price")
        assert "price_difference" in result.columns
        assert result["price_difference"].iloc[0] == -2
        assert result["price_difference"].iloc[1] == 2

    def test_adds_discount_features(self):
        """Test that discount features are added."""
        df = pd.DataFrame({"price": [8, 20], "reference_price": [10, 20]})
        result = add_price_features(df, reference_price_column="reference_price")
        assert "discount_amount" in result.columns
        assert "discount_rate" in result.columns
        assert "is_discounted" in result.columns
        assert result["discount_amount"].iloc[0] == 2
        assert result["is_discounted"].iloc[0] == 1

    def test_adds_margin_features(self):
        """Test that margin features are added when cost is provided."""
        df = pd.DataFrame({"price": [10, 20], "cost": [6, 12]})
        result = add_price_features(df, cost_column="cost")
        assert "unit_margin" in result.columns
        assert "margin_rate" in result.columns
        assert result["unit_margin"].iloc[0] == 4

    def test_custom_column_names(self):
        """Test that custom column names work."""
        df = pd.DataFrame({"my_price": [10, 20]})
        result = add_price_features(df, price_column="my_price")
        assert "log_price" in result.columns


class TestAddDemandFeatures:
    """Test add_demand_features function."""

    def test_adds_revenue(self):
        """Test that revenue is calculated."""
        df = pd.DataFrame({"quantity": [10, 20], "price": [5, 10]})
        result = add_demand_features(df)
        assert "revenue" in result.columns
        assert result["revenue"].iloc[0] == 50
        assert result["revenue"].iloc[1] == 200

    def test_adds_log_quantity(self):
        """Test that log quantity is added."""
        df = pd.DataFrame({"quantity": [10, 20], "price": [5, 10]})
        result = add_demand_features(df)
        assert "log_quantity" in result.columns

    def test_adds_profit_features(self):
        """Test that profit features are added when cost is provided."""
        df = pd.DataFrame({"quantity": [10, 20], "price": [10, 15], "cost": [6, 9]})
        result = add_demand_features(df, cost_column="cost")
        assert "gross_profit" in result.columns
        assert "total_cost" in result.columns
        assert result["gross_profit"].iloc[0] == 40  # (10-6) * 10


class TestAddPromotionFeatures:
    """Test add_promotion_features function."""

    def test_boolean_promotion_column(self):
        """Test that boolean promotion values work."""
        df = pd.DataFrame({"promotion": [True, False, True]})
        result = add_promotion_features(df)
        assert "is_promotion" in result.columns
        assert result["is_promotion"].iloc[0] == 1
        assert result["is_promotion"].iloc[1] == 0

    def test_numeric_promotion_column(self):
        """Test that numeric promotion values work."""
        df = pd.DataFrame({"promotion": [1, 0, 2]})
        result = add_promotion_features(df)
        assert result["is_promotion"].iloc[0] == 1
        assert result["is_promotion"].iloc[1] == 0
        assert result["is_promotion"].iloc[2] == 1  # Any positive number

    def test_string_promotion_column(self):
        """Test that string promotion values work."""
        df = pd.DataFrame({"promotion": ["yes", "no", "promotion"]})
        result = add_promotion_features(df)
        assert result["is_promotion"].iloc[0] == 1
        assert result["is_promotion"].iloc[1] == 0
        assert result["is_promotion"].iloc[2] == 1


class TestAddLagFeatures:
    """Test add_lag_features function."""

    def test_adds_lag_columns(self):
        """Test that lag columns are added."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "product": ["A"] * 10,
                "sales": range(10, 20),
            }
        )
        result = add_lag_features(
            df,
            value_columns=["sales"],
            group_columns=["product"],
            lags=[1, 2],
        )
        assert "sales_lag_1" in result.columns
        assert "sales_lag_2" in result.columns

    def test_lag_values_correct(self):
        """Test that lag values are correct."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "product": ["A"] * 5,
                "sales": [10, 20, 30, 40, 50],
            }
        )
        result = add_lag_features(
            df,
            value_columns=["sales"],
            group_columns=["product"],
            lags=[1],
        )
        assert pd.isna(result["sales_lag_1"].iloc[0])
        assert result["sales_lag_1"].iloc[1] == 10
        assert result["sales_lag_1"].iloc[2] == 20

    def test_grouped_lags(self):
        """Test that lags are calculated within groups."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4).tolist() * 2,
                "product": ["A"] * 4 + ["B"] * 4,
                "sales": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        result = add_lag_features(
            df,
            value_columns=["sales"],
            group_columns=["product"],
            lags=[1],
        )
        # First row of each product should have NaN lag
        product_a = result[result["product"] == "A"]
        product_b = result[result["product"] == "B"]
        assert pd.isna(product_a["sales_lag_1"].iloc[0])
        assert pd.isna(product_b["sales_lag_1"].iloc[0])

    def test_zero_lag_raises_error(self):
        """Test that zero lag raises ValueError."""
        df = pd.DataFrame({"date": ["2023-01-01"], "product": ["A"], "sales": [10]})
        with pytest.raises(ValueError, match="greater than zero"):
            add_lag_features(
                df,
                value_columns=["sales"],
                group_columns=["product"],
                lags=[0],
            )


class TestAddRollingFeatures:
    """Test add_rolling_features function."""

    def test_adds_rolling_columns(self):
        """Test that rolling columns are added."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "product": ["A"] * 10,
                "sales": range(10, 20),
            }
        )
        result = add_rolling_features(
            df,
            value_columns=["sales"],
            group_columns=["product"],
            windows=[3],
        )
        assert "sales_rolling_mean_3" in result.columns
        assert "sales_rolling_std_3" in result.columns

    def test_rolling_mean_calculation(self):
        """Test that rolling mean is calculated correctly."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "product": ["A"] * 5,
                "sales": [10, 20, 30, 40, 50],
            }
        )
        result = add_rolling_features(
            df,
            value_columns=["sales"],
            group_columns=["product"],
            windows=[2],
            min_periods=1,
        )
        # Rolling mean should exclude current value (shifted by 1)
        # Row 2: mean of [10, 20] = 15
        assert result["sales_rolling_mean_2"].iloc[2] == 15.0

    def test_zero_window_raises_error(self):
        """Test that zero window raises ValueError."""
        df = pd.DataFrame({"date": ["2023-01-01"], "product": ["A"], "sales": [10]})
        with pytest.raises(ValueError, match="greater than zero"):
            add_rolling_features(
                df,
                value_columns=["sales"],
                group_columns=["product"],
                windows=[0],
            )


class TestAddPriceChangeFeatures:
    """Test add_price_change_features function."""

    def test_adds_price_change_columns(self):
        """Test that price change columns are added."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "product": ["A"] * 5,
                "price": [10, 12, 11, 13, 12],
            }
        )
        result = add_price_change_features(
            df,
            group_columns=["product"],
        )
        expected_cols = [
            "previous_price",
            "price_change",
            "price_change_rate",
            "price_increased",
            "price_decreased",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_price_change_calculation(self):
        """Test that price change is calculated correctly."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "product": ["A"] * 3,
                "price": [10, 12, 11],
            }
        )
        result = add_price_change_features(df, group_columns=["product"])
        assert pd.isna(result["price_change"].iloc[0])
        assert result["price_change"].iloc[1] == 2
        assert result["price_change"].iloc[2] == -1

    def test_price_change_flags(self):
        """Test that price increase/decrease flags are correct."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4),
                "product": ["A"] * 4,
                "price": [10, 12, 12, 11],
            }
        )
        result = add_price_change_features(df, group_columns=["product"])
        assert result["price_increased"].iloc[1] == 1
        assert result["price_decreased"].iloc[3] == 1
        assert result["price_increased"].iloc[2] == 0  # No change
        assert result["price_decreased"].iloc[2] == 0  # No change
