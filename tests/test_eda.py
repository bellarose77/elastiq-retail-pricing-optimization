"""Unit tests for src/analysis/eda.py"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import (
    calculate_kpi_summary,
    calculate_price_demand_correlation,
    create_discount_bands,
    summarize_by_category,
    summarize_by_product,
    summarize_by_region,
    summarize_by_store,
)


@pytest.fixture
def sample_retail_data():
    """Create sample retail data for testing."""
    return pd.DataFrame(
        {
            "store_id": ["S1", "S1", "S2", "S2"],
            "region": ["North", "North", "South", "South"],
            "product_id": ["P1", "P2", "P1", "P2"],
            "category": ["Beverage", "Grocery", "Beverage", "Grocery"],
            "units_sold": [100, 200, 150, 250],
            "revenue": [1000, 2000, 1500, 2500],
            "gross_profit": [300, 600, 450, 750],
            "selling_price": [10.0, 10.0, 10.0, 10.0],
            "is_promotion": [0, 1, 0, 1],
            "discount_rate": [0.0, 0.15, 0.0, 0.20],
            "stockout_flag": [0, 0, 1, 0],
        }
    )


class TestCalculateKpiSummary:
    """Test calculate_kpi_summary function."""

    def test_returns_dataframe_with_expected_columns(self, sample_retail_data):
        """Test that the function returns a DataFrame with metric and value columns."""
        result = calculate_kpi_summary(sample_retail_data)
        assert isinstance(result, pd.DataFrame)
        assert "metric" in result.columns
        assert "value" in result.columns

    def test_number_of_observations_correct(self, sample_retail_data):
        """Test that number of observations is correctly calculated."""
        result = calculate_kpi_summary(sample_retail_data)
        obs_row = result[result["metric"] == "Number of observations"]
        assert obs_row["value"].iloc[0] == 4

    def test_total_revenue_correct(self, sample_retail_data):
        """Test that total revenue is correctly calculated."""
        result = calculate_kpi_summary(sample_retail_data)
        revenue_row = result[result["metric"] == "Total revenue"]
        assert revenue_row["value"].iloc[0] == 7000

    def test_promotion_rate_correct(self, sample_retail_data):
        """Test that promotion rate is correctly calculated."""
        result = calculate_kpi_summary(sample_retail_data)
        promo_row = result[result["metric"] == "Promotion rate"]
        assert promo_row["value"].iloc[0] == 0.5

    def test_stockout_rate_correct(self, sample_retail_data):
        """Test that stockout rate is correctly calculated."""
        result = calculate_kpi_summary(sample_retail_data)
        stockout_row = result[result["metric"] == "Stockout rate"]
        assert stockout_row["value"].iloc[0] == 0.25


class TestSummarizeByCategory:
    """Test summarize_by_category function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = summarize_by_category(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_category(self, sample_retail_data):
        """Test that data is grouped by category."""
        result = summarize_by_category(sample_retail_data)
        assert "category" in result.columns
        assert len(result) == 2  # Beverage and Grocery

    def test_calculates_margin_rate(self, sample_retail_data):
        """Test that margin rate is calculated."""
        result = summarize_by_category(sample_retail_data)
        assert "margin_rate" in result.columns
        # Margin rate = gross_profit / revenue
        assert all(result["margin_rate"] >= 0)
        assert all(result["margin_rate"] <= 1)

    def test_sorted_by_revenue_descending(self, sample_retail_data):
        """Test that results are sorted by revenue in descending order."""
        result = summarize_by_category(sample_retail_data)
        assert result["revenue"].is_monotonic_decreasing


class TestSummarizeByProduct:
    """Test summarize_by_product function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = summarize_by_product(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_includes_product_and_category(self, sample_retail_data):
        """Test that product_id and category are in the result."""
        result = summarize_by_product(sample_retail_data)
        assert "product_id" in result.columns
        assert "category" in result.columns

    def test_respects_top_n_parameter(self, sample_retail_data):
        """Test that top_n parameter limits results."""
        result = summarize_by_product(sample_retail_data, top_n=1)
        assert len(result) == 1

    def test_returns_highest_revenue_products(self, sample_retail_data):
        """Test that the highest revenue products are returned."""
        result = summarize_by_product(sample_retail_data, top_n=1)
        # P2 has total revenue of 4500
        assert result["product_id"].iloc[0] == "P2"


class TestSummarizeByStore:
    """Test summarize_by_store function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = summarize_by_store(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_store_and_region(self, sample_retail_data):
        """Test that data is grouped by store_id and region."""
        result = summarize_by_store(sample_retail_data)
        assert "store_id" in result.columns
        assert "region" in result.columns

    def test_sorted_by_revenue_descending(self, sample_retail_data):
        """Test that results are sorted by revenue in descending order."""
        result = summarize_by_store(sample_retail_data)
        assert result["revenue"].is_monotonic_decreasing


class TestSummarizeByRegion:
    """Test summarize_by_region function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = summarize_by_region(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_region(self, sample_retail_data):
        """Test that data is grouped by region."""
        result = summarize_by_region(sample_retail_data)
        assert "region" in result.columns
        assert len(result) == 2  # North and South

    def test_aggregates_revenue_correctly(self, sample_retail_data):
        """Test that revenue is aggregated correctly by region."""
        result = summarize_by_region(sample_retail_data)
        north_revenue = result[result["region"] == "North"]["revenue"].iloc[0]
        south_revenue = result[result["region"] == "South"]["revenue"].iloc[0]
        assert north_revenue == 3000  # 1000 + 2000
        assert south_revenue == 4000  # 1500 + 2500


class TestCreateDiscountBands:
    """Test create_discount_bands function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = create_discount_bands(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_creates_discount_bands(self, sample_retail_data):
        """Test that discount bands are created."""
        result = create_discount_bands(sample_retail_data)
        assert "discount_band" in result.columns

    def test_custom_bins_and_labels(self, sample_retail_data):
        """Test that custom bins and labels work."""
        bins = [0, 0.1, 0.5, 1.0]
        labels = ["Low", "Medium", "High"]
        result = create_discount_bands(
            sample_retail_data, bins=bins, labels=labels
        )
        assert len(result) == len(labels)

    def test_aggregates_units_and_revenue(self, sample_retail_data):
        """Test that units and revenue are aggregated."""
        result = create_discount_bands(sample_retail_data)
        # Should have columns for units_sold and revenue aggregations
        assert "units_sold" in result.columns.get_level_values(0)
        assert "revenue" in result.columns.get_level_values(0)


class TestCalculatePriceDemandCorrelation:
    """Test calculate_price_demand_correlation function."""

    def test_returns_dataframe(self, sample_retail_data):
        """Test that the function returns a DataFrame."""
        result = calculate_price_demand_correlation(sample_retail_data)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_specified_column(self, sample_retail_data):
        """Test that data is grouped by specified column."""
        result = calculate_price_demand_correlation(
            sample_retail_data, group_by="category"
        )
        assert "category" in result.columns

    def test_calculates_correlation(self, sample_retail_data):
        """Test that correlation is calculated."""
        result = calculate_price_demand_correlation(sample_retail_data)
        assert "price_demand_correlation" in result.columns

    def test_includes_observation_count(self, sample_retail_data):
        """Test that observation count is included."""
        result = calculate_price_demand_correlation(sample_retail_data)
        assert "n_observations" in result.columns

    def test_handles_multiple_group_columns(self):
        """Test that multiple group columns work."""
        df = pd.DataFrame(
            {
                "store_id": ["S1", "S1", "S2", "S2"],
                "category": ["A", "A", "B", "B"],
                "selling_price": [10, 12, 11, 13],
                "units_sold": [100, 90, 95, 85],
            }
        )
        result = calculate_price_demand_correlation(
            df, group_by=["store_id", "category"]
        )
        assert "store_id" in result.columns
        assert "category" in result.columns

    def test_skips_groups_with_insufficient_data(self):
        """Test that groups with < 2 observations are skipped."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "B"],
                "selling_price": [10, 12, 11],
                "units_sold": [100, 90, 95],
            }
        )
        result = calculate_price_demand_correlation(df, group_by="category")
        # Category A has only 1 observation, should be skipped
        assert len(result) == 1
        assert result["category"].iloc[0] == "B"
