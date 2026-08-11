"""Unit tests for src/models/elasticity.py"""

import numpy as np
import pandas as pd
import pytest

from src.models.elasticity import (
    calculate_arc_elasticity,
    classify_price_elasticity,
    estimate_category_heterogeneity,
    extract_price_elasticity,
    predict_quantity_from_elasticity,
    prepare_elasticity_data,
    shrink_product_elasticities,
    simulate_price_scenarios,
)


class TestCalculateArcElasticity:
    """Test calculate_arc_elasticity function."""

    def test_normal_elasticity_calculation(self):
        """Test elasticity calculation with normal values."""
        result = calculate_arc_elasticity(
            old_quantity=100, new_quantity=90, old_price=10, new_price=11
        )
        # Price increased 10%, quantity decreased 10.5%, elasticity ≈ -1.05
        assert result < 0  # Negative elasticity
        assert -1.2 < result < -0.9

    def test_price_increase_demand_decrease(self):
        """Test that price increase with demand decrease gives negative elasticity."""
        result = calculate_arc_elasticity(
            old_quantity=100, new_quantity=80, old_price=10, new_price=12
        )
        assert result < 0

    def test_zero_quantity_returns_nan(self):
        """Test that zero quantity returns NaN."""
        result = calculate_arc_elasticity(
            old_quantity=0, new_quantity=90, old_price=10, new_price=11
        )
        assert np.isnan(result)

    def test_zero_price_returns_nan(self):
        """Test that zero price returns NaN."""
        result = calculate_arc_elasticity(
            old_quantity=100, new_quantity=90, old_price=0, new_price=11
        )
        assert np.isnan(result)

    def test_no_price_change_returns_nan(self):
        """Test that no price change returns NaN."""
        result = calculate_arc_elasticity(
            old_quantity=100, new_quantity=90, old_price=10, new_price=10
        )
        assert np.isnan(result)

    def test_infinite_values_return_nan(self):
        """Test that infinite values return NaN."""
        result = calculate_arc_elasticity(
            old_quantity=np.inf, new_quantity=90, old_price=10, new_price=11
        )
        assert np.isnan(result)

    def test_symmetric_calculation(self):
        """Test that elasticity calculation is symmetric."""
        # Forward calculation
        forward = calculate_arc_elasticity(
            old_quantity=100, new_quantity=90, old_price=10, new_price=11
        )
        # Backward calculation
        backward = calculate_arc_elasticity(
            old_quantity=90, new_quantity=100, old_price=11, new_price=10
        )
        assert np.isclose(forward, backward, rtol=0.01)


class TestClassifyPriceElasticity:
    """Test classify_price_elasticity function."""

    def test_elastic_demand(self):
        """Test classification of elastic demand."""
        assert classify_price_elasticity(-1.5) == "elastic"
        assert classify_price_elasticity(-2.0) == "elastic"

    def test_inelastic_demand(self):
        """Test classification of inelastic demand."""
        assert classify_price_elasticity(-0.5) == "inelastic"
        assert classify_price_elasticity(-0.8) == "inelastic"

    def test_unit_elastic_demand(self):
        """Test classification of unit elastic demand."""
        assert classify_price_elasticity(-1.0) == "unit_elastic"
        assert classify_price_elasticity(-0.98) == "unit_elastic"
        assert classify_price_elasticity(-1.02) == "unit_elastic"

    def test_positive_elasticity(self):
        """Test classification of positive elasticity (Giffen goods)."""
        assert classify_price_elasticity(0.5) == "positive"
        assert classify_price_elasticity(1.5) == "positive"

    def test_nan_returns_unknown(self):
        """Test that NaN returns 'unknown'."""
        assert classify_price_elasticity(np.nan) == "unknown"

    def test_none_returns_unknown(self):
        """Test that None returns 'unknown'."""
        assert classify_price_elasticity(None) == "unknown"

    def test_infinity_returns_unknown(self):
        """Test that infinity returns 'unknown'."""
        assert classify_price_elasticity(np.inf) == "unknown"

    def test_custom_tolerance(self):
        """Test that custom tolerance works."""
        # With tolerance of 0.1, -1.05 should be unit_elastic
        assert classify_price_elasticity(-1.05, unitary_tolerance=0.1) == "unit_elastic"
        # With tolerance of 0.01, -1.05 should be elastic
        assert classify_price_elasticity(-1.05, unitary_tolerance=0.01) == "elastic"


class TestPrepareElasticityData:
    """Test prepare_elasticity_data function."""

    def test_basic_preparation(self):
        """Test basic data preparation."""
        df = pd.DataFrame(
            {"quantity": [100, 200, 300], "price": [10, 20, 30]}
        )
        result = prepare_elasticity_data(df)
        assert "log_quantity" in result.columns
        assert "log_price" in result.columns
        assert len(result) == 3

    def test_removes_zero_quantity(self):
        """Test that zero quantity rows are removed."""
        df = pd.DataFrame(
            {"quantity": [0, 200, 300], "price": [10, 20, 30]}
        )
        result = prepare_elasticity_data(df)
        assert len(result) == 2

    def test_removes_negative_price(self):
        """Test that negative price rows are removed."""
        df = pd.DataFrame(
            {"quantity": [100, 200, 300], "price": [-10, 20, 30]}
        )
        result = prepare_elasticity_data(df)
        assert len(result) == 2

    def test_removes_missing_values(self):
        """Test that rows with missing values are removed."""
        df = pd.DataFrame(
            {"quantity": [100, None, 300], "price": [10, 20, 30]}
        )
        result = prepare_elasticity_data(df)
        assert len(result) == 2

    def test_converts_string_numbers(self):
        """Test that string numbers are converted."""
        df = pd.DataFrame(
            {"quantity": ["100", "200", "300"], "price": ["10", "20", "30"]}
        )
        result = prepare_elasticity_data(df)
        assert pd.api.types.is_numeric_dtype(result["log_quantity"])
        assert pd.api.types.is_numeric_dtype(result["log_price"])

    def test_custom_column_names(self):
        """Test that custom column names work."""
        df = pd.DataFrame(
            {"units": [100, 200, 300], "cost": [10, 20, 30]}
        )
        result = prepare_elasticity_data(
            df, quantity_column="units", price_column="cost"
        )
        assert "log_quantity" in result.columns
        assert "log_price" in result.columns


class TestPredictQuantityFromElasticity:
    """Test predict_quantity_from_elasticity function."""

    def test_basic_prediction(self):
        """Test basic quantity prediction."""
        result = predict_quantity_from_elasticity(
            base_quantity=100,
            base_price=10,
            new_price=11,
            elasticity=-1.0,
        )
        # 10% price increase with -1.0 elasticity = 10% quantity decrease
        expected = 100 * (11 / 10) ** -1.0
        assert np.isclose(result, expected)

    def test_price_increase_reduces_quantity(self):
        """Test that price increase reduces quantity with negative elasticity."""
        result = predict_quantity_from_elasticity(
            base_quantity=100,
            base_price=10,
            new_price=12,
            elasticity=-1.5,
        )
        assert result < 100

    def test_price_decrease_increases_quantity(self):
        """Test that price decrease increases quantity with negative elasticity."""
        result = predict_quantity_from_elasticity(
            base_quantity=100,
            base_price=10,
            new_price=8,
            elasticity=-1.5,
        )
        assert result > 100

    def test_negative_base_quantity_raises_error(self):
        """Test that negative base quantity raises ValueError."""
        with pytest.raises(ValueError, match="negative values"):
            predict_quantity_from_elasticity(
                base_quantity=-100,
                base_price=10,
                new_price=11,
                elasticity=-1.0,
            )

    def test_zero_base_price_raises_error(self):
        """Test that zero base price raises ValueError."""
        with pytest.raises(ValueError, match="positive values"):
            predict_quantity_from_elasticity(
                base_quantity=100,
                base_price=0,
                new_price=11,
                elasticity=-1.0,
            )

    def test_zero_new_price_raises_error(self):
        """Test that zero new price raises ValueError."""
        with pytest.raises(ValueError, match="positive values"):
            predict_quantity_from_elasticity(
                base_quantity=100,
                base_price=10,
                new_price=0,
                elasticity=-1.0,
            )

    def test_array_inputs(self):
        """Test that array inputs work correctly."""
        result = predict_quantity_from_elasticity(
            base_quantity=100,
            base_price=10,
            new_price=np.array([9, 10, 11]),
            elasticity=-1.0,
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[1] == 100  # No price change

    def test_zero_elasticity_constant_quantity(self):
        """Test that zero elasticity gives constant quantity."""
        result = predict_quantity_from_elasticity(
            base_quantity=100,
            base_price=10,
            new_price=20,
            elasticity=0.0,
        )
        assert result == 100


class TestSimulatePriceScenarios:
    """Test simulate_price_scenarios function."""

    def test_returns_dataframe(self):
        """Test that the function returns a DataFrame."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
        )
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self):
        """Test that result contains expected columns."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
        )
        expected_cols = [
            "price_change_rate",
            "candidate_price",
            "expected_quantity",
            "expected_revenue",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_includes_profit_when_cost_provided(self):
        """Test that profit columns are included when unit_cost is provided."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert "expected_profit" in result.columns
        assert "unit_margin" in result.columns

    def test_custom_price_change_rates(self):
        """Test that custom price change rates work."""
        custom_rates = [-0.1, 0.0, 0.1]
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
            price_change_rates=custom_rates,
        )
        assert len(result) == 3

    def test_zero_price_change_preserves_quantity(self):
        """Test that zero price change preserves quantity."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
            price_change_rates=[0.0],
        )
        assert np.isclose(result["expected_quantity"].iloc[0], 100)

    def test_negative_base_price_raises_error(self):
        """Test that negative base price raises ValueError."""
        with pytest.raises(ValueError, match="greater than zero"):
            simulate_price_scenarios(
                base_price=-10,
                base_quantity=100,
                elasticity=-1.0,
            )

    def test_negative_base_quantity_raises_error(self):
        """Test that negative base quantity raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            simulate_price_scenarios(
                base_price=10,
                base_quantity=-100,
                elasticity=-1.0,
            )

    def test_negative_unit_cost_raises_error(self):
        """Test that negative unit cost raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            simulate_price_scenarios(
                base_price=10,
                base_quantity=100,
                elasticity=-1.0,
                unit_cost=-5,
            )

    def test_sorted_by_price(self):
        """Test that results are sorted by candidate price."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
            price_change_rates=[0.2, -0.1, 0.0, 0.1, -0.2],
        )
        assert result["candidate_price"].is_monotonic_increasing

    def test_revenue_calculation_correct(self):
        """Test that revenue is calculated correctly."""
        result = simulate_price_scenarios(
            base_price=10,
            base_quantity=100,
            elasticity=-1.0,
            price_change_rates=[0.0],
        )
        expected_revenue = 10 * 100
        assert np.isclose(result["expected_revenue"].iloc[0], expected_revenue)


class TestEstimateCategoryHeterogeneity:
    """Test estimate_category_heterogeneity function."""

    def test_fewer_than_two_products_returns_zero(self):
        """Test that a single product gives no heterogeneity estimate."""
        result = estimate_category_heterogeneity(
            np.array([-0.8]), np.array([0.1]), category_elasticity=-0.7
        )
        assert result == 0.0

    def test_identical_estimates_give_zero_heterogeneity(self):
        """Test that products matching the category exactly show no heterogeneity."""
        result = estimate_category_heterogeneity(
            np.array([-0.7, -0.7, -0.7]),
            np.array([0.1, 0.1, 0.1]),
            category_elasticity=-0.7,
        )
        assert result == 0.0

    def test_detects_real_heterogeneity(self):
        """Test that tight estimates far from the category mean give tau > 0."""
        result = estimate_category_heterogeneity(
            np.array([-2.0, -0.2]),
            np.array([0.05, 0.05]),
            category_elasticity=-0.7,
        )
        assert result > 0

    def test_never_negative(self):
        """Test that the method-of-moments floor at zero holds."""
        result = estimate_category_heterogeneity(
            np.array([-0.71, -0.69]),
            np.array([5.0, 5.0]),
            category_elasticity=-0.7,
        )
        assert result >= 0.0


class TestShrinkProductElasticities:
    """Test shrink_product_elasticities function."""

    def _category_estimates(self):
        return pd.DataFrame(
            [
                {
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -0.70,
                    "standard_error": 0.02,
                },
            ]
        )

    def test_precise_plausible_product_stays_close_to_own_estimate(self):
        """Test that a tight, negative-signed estimate keeps a high weight."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -2.0,
                    "standard_error": 0.05,
                },
                {
                    "product_id": "P002",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -0.2,
                    "standard_error": 0.05,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        row = result.loc[result["product_id"] == "P001"].iloc[0]
        assert row["shrinkage_weight"] > 0.5
        assert abs(row["shrunk_elasticity"] - (-2.0)) < abs(row["shrunk_elasticity"] - (-0.70))

    def test_imprecise_product_pulled_toward_category(self):
        """Test that a noisy estimate (large standard error) is mostly pooled."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -2.0,
                    "standard_error": 0.05,
                },
                {
                    "product_id": "P002",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -3.5,
                    "standard_error": 3.0,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        row = result.loc[result["product_id"] == "P002"].iloc[0]
        assert row["shrinkage_weight"] < 0.5
        assert abs(row["shrunk_elasticity"] - (-0.70)) < abs(row["shrunk_elasticity"] - (-3.5))

    def test_wrong_signed_estimate_falls_back_to_category(self):
        """Test that a positive-signed shrunk result is replaced by the category value."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": 0.9,
                    "standard_error": 0.1,
                },
                {
                    "product_id": "P002",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": 0.6,
                    "standard_error": 0.1,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        assert (result["shrunk_elasticity"] < 0).all()
        assert (result["shrinkage_weight"] == 0.0).all()

    def test_failed_fit_falls_back_to_category(self):
        """Test that a product with no valid own fit is fully pooled."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Beverages",
                    "status": "insufficient_observations",
                    "elasticity": np.nan,
                    "standard_error": np.nan,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        row = result.iloc[0]
        assert row["shrinkage_weight"] == 0.0
        assert row["shrunk_elasticity"] == -0.70

    def test_unknown_category_leaves_shrunk_columns_nan(self):
        """Test that a product whose category has no fit is left unshrunk."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Unmapped",
                    "status": "success",
                    "elasticity": -1.0,
                    "standard_error": 0.1,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        assert np.isnan(result.iloc[0]["shrunk_elasticity"])

    def test_confidence_interval_matches_elasticity_and_standard_error(self):
        """Test that the shrunk CI is a 95% interval around the shrunk elasticity."""
        product_estimates = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -2.0,
                    "standard_error": 0.05,
                },
                {
                    "product_id": "P002",
                    "category": "Beverages",
                    "status": "success",
                    "elasticity": -0.2,
                    "standard_error": 0.05,
                },
            ]
        )
        result = shrink_product_elasticities(
            product_estimates, self._category_estimates(), category_column="category"
        )
        row = result.iloc[0]
        expected_lower = row["shrunk_elasticity"] - 1.96 * row["shrunk_standard_error"]
        expected_upper = row["shrunk_elasticity"] + 1.96 * row["shrunk_standard_error"]
        assert np.isclose(row["shrunk_confidence_interval_lower"], expected_lower)
        assert np.isclose(row["shrunk_confidence_interval_upper"], expected_upper)
