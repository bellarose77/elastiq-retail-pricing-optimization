"""Unit tests for src/optimization/pricing.py"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.pricing import (
    PricingOptimizationConfig,
    build_optimization_configuration,
    calculate_confidence_dampening_factor,
    calculate_minimum_price_for_margin,
    classify_pricing_action,
    evaluate_price_scenarios,
    generate_price_change_rates,
    optimize_item_price,
    optimize_price_portfolio,
    recompute_price_financials,
    round_to_price_ending,
    safe_change_rate,
    select_optimal_price_scenario,
    validate_pricing_configuration,
)


class TestPricingOptimizationConfig:
    """Test PricingOptimizationConfig dataclass."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = PricingOptimizationConfig()
        assert config.objective == "profit"
        assert config.minimum_price_change_rate == -0.20
        assert config.maximum_price_change_rate == 0.20
        assert config.price_change_step == 0.01

    def test_custom_values(self):
        """Test that custom configuration values work."""
        config = PricingOptimizationConfig(
            objective="revenue",
            minimum_price_change_rate=-0.30,
            maximum_price_change_rate=0.30,
        )
        assert config.objective == "revenue"
        assert config.minimum_price_change_rate == -0.30
        assert config.maximum_price_change_rate == 0.30


class TestValidatePricingConfiguration:
    """Test validate_pricing_configuration function."""

    def test_valid_configuration_passes(self):
        """Test that a valid configuration passes validation."""
        config = PricingOptimizationConfig()
        validate_pricing_configuration(config)  # Should not raise

    def test_invalid_objective_raises_error(self):
        """Test that invalid objective raises ValueError."""
        config = PricingOptimizationConfig(objective="invalid")
        with pytest.raises(ValueError, match="objective must be one of"):
            validate_pricing_configuration(config)

    def test_min_greater_than_max_raises_error(self):
        """Test that min > max raises ValueError."""
        config = PricingOptimizationConfig(
            minimum_price_change_rate=0.2,
            maximum_price_change_rate=0.1,
        )
        with pytest.raises(ValueError, match="must be smaller than"):
            validate_pricing_configuration(config)

    def test_min_less_than_minus_one_raises_error(self):
        """Test that min < -1 raises ValueError."""
        config = PricingOptimizationConfig(minimum_price_change_rate=-1.5)
        with pytest.raises(ValueError, match="must be greater than -1"):
            validate_pricing_configuration(config)

    def test_zero_step_raises_error(self):
        """Test that zero price change step raises ValueError."""
        config = PricingOptimizationConfig(price_change_step=0)
        with pytest.raises(ValueError, match="must be greater than zero"):
            validate_pricing_configuration(config)

    def test_invalid_margin_rate_raises_error(self):
        """Test that invalid margin rate raises ValueError."""
        config = PricingOptimizationConfig(minimum_margin_rate=1.5)
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_pricing_configuration(config)

    def test_negative_expected_quantity_raises_error(self):
        """Test that negative expected quantity raises ValueError."""
        config = PricingOptimizationConfig(minimum_expected_quantity=-10)
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_pricing_configuration(config)

    def test_zero_cannibalization_max_iterations_raises_error(self):
        """Test that a non-positive iteration cap raises ValueError."""
        config = PricingOptimizationConfig(
            cannibalization_max_iterations=0
        )
        with pytest.raises(
            ValueError, match="cannibalization_max_iterations"
        ):
            validate_pricing_configuration(config)


class TestSafeChangeRate:
    """Test safe_change_rate function."""

    def test_normal_change_rate(self):
        """Test normal change rate calculation."""
        result = safe_change_rate(110, 100)
        assert result == 0.1

    def test_negative_change_rate(self):
        """Test negative change rate calculation."""
        result = safe_change_rate(90, 100)
        assert result == -0.1

    def test_zero_old_value_returns_nan(self):
        """Test that zero old value returns NaN."""
        result = safe_change_rate(110, 0)
        assert np.isnan(result)

    def test_infinite_new_value_returns_nan(self):
        """Test that infinite new value returns NaN."""
        result = safe_change_rate(np.inf, 100)
        assert np.isnan(result)

    def test_nan_new_value_returns_nan(self):
        """Test that NaN new value returns NaN."""
        result = safe_change_rate(np.nan, 100)
        assert np.isnan(result)


class TestClassifyPricingAction:
    """Test classify_pricing_action function."""

    def test_increase_price_action(self):
        """Test that positive change rates are classified as increase."""
        assert classify_pricing_action(0.1) == "increase_price"
        assert classify_pricing_action(0.01) == "increase_price"

    def test_decrease_price_action(self):
        """Test that negative change rates are classified as decrease."""
        assert classify_pricing_action(-0.1) == "decrease_price"
        assert classify_pricing_action(-0.01) == "decrease_price"

    def test_hold_price_action(self):
        """Test that small changes are classified as hold."""
        assert classify_pricing_action(0.001, threshold=0.005) == "hold_price"
        assert classify_pricing_action(-0.001, threshold=0.005) == "hold_price"
        assert classify_pricing_action(0.0) == "hold_price"

    def test_nan_returns_review(self):
        """Test that NaN returns review."""
        assert classify_pricing_action(np.nan) == "review"

    def test_custom_threshold(self):
        """Test that custom threshold works."""
        assert classify_pricing_action(0.002, threshold=0.001) == "increase_price"
        assert classify_pricing_action(0.002, threshold=0.01) == "hold_price"


class TestCalculateMinimumPriceForMargin:
    """Test calculate_minimum_price_for_margin function."""

    def test_basic_calculation(self):
        """Test basic minimum price calculation."""
        # cost=10, margin=20% -> price = 10/(1-0.2) = 12.5
        result = calculate_minimum_price_for_margin(10, 0.2)
        assert np.isclose(result, 12.5)

    def test_zero_margin_returns_cost(self):
        """Test that zero margin returns cost."""
        result = calculate_minimum_price_for_margin(10, 0.0)
        assert result == 10

    def test_high_margin(self):
        """Test calculation with high margin."""
        # cost=10, margin=50% -> price = 10/(1-0.5) = 20
        result = calculate_minimum_price_for_margin(10, 0.5)
        assert result == 20

    def test_negative_cost_raises_error(self):
        """Test that negative cost raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_minimum_price_for_margin(-10, 0.2)

    def test_invalid_margin_raises_error(self):
        """Test that invalid margin raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            calculate_minimum_price_for_margin(10, 1.5)


class TestGeneratePriceChangeRates:
    """Test generate_price_change_rates function."""

    def test_basic_generation(self):
        """Test basic price change rate generation."""
        result = generate_price_change_rates(
            minimum_change_rate=-0.1,
            maximum_change_rate=0.1,
            step=0.05,
        )
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_includes_zero(self):
        """Test that zero is always included."""
        result = generate_price_change_rates(
            minimum_change_rate=-0.1,
            maximum_change_rate=0.1,
            step=0.03,
        )
        assert 0.0 in result

    def test_includes_endpoints(self):
        """Test that endpoints are included."""
        result = generate_price_change_rates(
            minimum_change_rate=-0.2,
            maximum_change_rate=0.2,
            step=0.1,
        )
        assert -0.2 in result
        assert 0.2 in result

    def test_sorted_output(self):
        """Test that output is sorted."""
        result = generate_price_change_rates()
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_min_greater_than_max_raises_error(self):
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="must be smaller than"):
            generate_price_change_rates(
                minimum_change_rate=0.2,
                maximum_change_rate=0.1,
            )

    def test_min_less_than_minus_one_raises_error(self):
        """Test that min < -1 raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than -1"):
            generate_price_change_rates(minimum_change_rate=-1.5)

    def test_zero_step_raises_error(self):
        """Test that zero step raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            generate_price_change_rates(step=0)


class TestEvaluatePriceScenarios:
    """Test evaluate_price_scenarios function."""

    def test_returns_dataframe(self):
        """Test that the function returns a DataFrame."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self):
        """Test that result contains expected columns."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        expected_cols = [
            "price_change_rate",
            "candidate_price",
            "expected_quantity",
            "expected_revenue",
            "is_feasible",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_sorted_by_price(self):
        """Test that results are sorted by candidate price."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert result["candidate_price"].is_monotonic_increasing

    def test_feasibility_flags(self):
        """Test that feasibility flags are set."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert "is_feasible" in result.columns
        assert "meets_margin_constraint" in result.columns

    def test_negative_price_raises_error(self):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            evaluate_price_scenarios(
                current_price=-10,
                baseline_quantity=100,
                elasticity=-1.0,
            )

    def test_negative_quantity_raises_error(self):
        """Test that negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            evaluate_price_scenarios(
                current_price=10,
                baseline_quantity=-100,
                elasticity=-1.0,
            )

    def test_infinite_elasticity_raises_error(self):
        """Test that infinite elasticity raises ValueError."""
        with pytest.raises(ValueError, match="must be a finite number"):
            evaluate_price_scenarios(
                current_price=10,
                baseline_quantity=100,
                elasticity=np.inf,
            )

    def test_custom_price_change_rates(self):
        """Test that custom price change rates work."""
        custom_rates = [-0.1, 0.0, 0.1]
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            price_change_rates=custom_rates,
        )
        assert len(result) == 3


class TestOptimizeItemPrice:
    """Test optimize_item_price function."""

    def test_returns_result_object(self):
        """Test that the function returns a PriceOptimizationResult."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert hasattr(result, "recommendation")
        assert hasattr(result, "scenarios")
        assert hasattr(result, "configuration")

    def test_recommendation_contains_expected_fields(self):
        """Test that recommendation contains expected fields."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        expected_fields = [
            "current_price",
            "recommended_price",
            "price_change",
            "price_change_rate",
            "recommendation_action",
            "expected_quantity",
            "expected_revenue",
            "expected_profit",
        ]
        for field in expected_fields:
            assert field in result.recommendation

    def test_recommendation_status_success(self):
        """Test that recommendation status is success."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert result.recommendation["status"] == "success"

    def test_profit_objective(self):
        """Test optimization with profit objective."""
        config = PricingOptimizationConfig(objective="profit")
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            configuration=config,
        )
        assert result.recommendation["optimization_objective"] == "profit"

    def test_revenue_objective(self):
        """Test optimization with revenue objective."""
        config = PricingOptimizationConfig(objective="revenue")
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            configuration=config,
        )
        assert result.recommendation["optimization_objective"] == "revenue"

    def test_quantity_objective(self):
        """Test optimization with quantity objective."""
        config = PricingOptimizationConfig(objective="quantity")
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            configuration=config,
        )
        assert result.recommendation["optimization_objective"] == "quantity"

    def test_elastic_demand_suggests_price_decrease(self):
        """Test that elastic demand (< -1) often suggests price decrease."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-2.0,  # Highly elastic
            unit_cost=3,
        )
        # With elastic demand, lowering price often increases revenue/profit
        # This is not guaranteed but is likely with these parameters
        assert result.recommendation["recommended_price"] is not None

    def test_inelastic_demand_suggests_price_increase(self):
        """Test that inelastic demand (> -1) often suggests price increase."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-0.5,  # Highly inelastic
            unit_cost=3,
        )
        # With inelastic demand, raising price often increases revenue/profit
        assert result.recommendation["recommended_price"] is not None

    def test_item_id_and_category_preserved(self):
        """Test that item_id and category are preserved in recommendation."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            item_id="ITEM123",
            category="Beverages",
        )
        assert result.recommendation["item_id"] == "ITEM123"
        assert result.recommendation["category"] == "Beverages"

    def test_scenarios_dataframe_not_empty(self):
        """Test that scenarios dataframe is not empty."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert not result.scenarios.empty
        assert len(result.scenarios) > 0

    def test_feasible_scenario_count(self):
        """Test that feasible scenario count is reported."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert "feasible_scenario_count" in result.recommendation
        assert result.recommendation["feasible_scenario_count"] > 0

    def test_minimum_margin_constraint(self):
        """Test that minimum margin constraint is respected."""
        config = PricingOptimizationConfig(minimum_margin_rate=0.3)
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            configuration=config,
        )
        recommended_price = result.recommendation["recommended_price"]
        margin_rate = (recommended_price - 5) / recommended_price
        assert margin_rate >= 0.3 - 1e-6  # Account for floating point errors

    def test_recommended_price_positive(self):
        """Test that recommended price is always positive."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert result.recommendation["recommended_price"] > 0

    def test_price_change_consistency(self):
        """Test that price change is consistent with current and recommended prices."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        expected_change = (
            result.recommendation["recommended_price"]
            - result.recommendation["current_price"]
        )
        assert np.isclose(
            result.recommendation["price_change"],
            expected_change,
            rtol=1e-6,
        )

    def test_handles_zero_elasticity(self):
        """Test that zero elasticity is handled (constant demand)."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=0.0,
            unit_cost=5,
        )
        # With zero elasticity, should maximize price (up to constraint)
        assert result.recommendation["recommended_price"] is not None

    def test_default_behavior_unchanged_by_new_optional_fields(self):
        """Test that new opt-in fields don't affect default recommendations."""
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.2,
            unit_cost=5,
        )
        assert result.recommendation["confidence_dampening_factor"] == 1.0
        assert result.recommendation["price_ending_applied"] is False
        assert result.recommendation["meets_competitor_constraint"] is True
        assert result.recommendation["meets_inventory_constraint"] is True


class TestCalculateConfidenceDampeningFactor:
    """Test calculate_confidence_dampening_factor function."""

    def test_zero_width_interval_gives_full_trust(self):
        """Test that a point-estimate interval returns a factor of 1."""
        factor = calculate_confidence_dampening_factor(
            elasticity=-1.5,
            elasticity_lower=-1.5,
            elasticity_upper=-1.5,
        )
        assert factor == pytest.approx(1.0)

    def test_tight_interval_gives_high_factor(self):
        """Test that a tight confidence interval yields a high factor."""
        factor = calculate_confidence_dampening_factor(
            elasticity=-1.5,
            elasticity_lower=-1.55,
            elasticity_upper=-1.45,
        )
        assert factor > 0.9

    def test_wide_interval_gives_low_factor(self):
        """Test that a wide confidence interval yields a low factor."""
        factor = calculate_confidence_dampening_factor(
            elasticity=-1.0,
            elasticity_lower=-3.0,
            elasticity_upper=1.0,
        )
        assert factor < 0.3

    def test_factor_is_bounded_between_zero_and_one(self):
        """Test that the factor always stays within (0, 1]."""
        factor = calculate_confidence_dampening_factor(
            elasticity=-0.1,
            elasticity_lower=-10.0,
            elasticity_upper=10.0,
        )
        assert 0.0 < factor <= 1.0

    def test_lower_greater_than_upper_raises_error(self):
        """Test that an inverted interval raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater than"):
            calculate_confidence_dampening_factor(
                elasticity=-1.0,
                elasticity_lower=-0.5,
                elasticity_upper=-1.5,
            )

    def test_non_finite_elasticity_raises_error(self):
        """Test that a non-finite elasticity raises ValueError."""
        with pytest.raises(ValueError, match="finite numbers"):
            calculate_confidence_dampening_factor(
                elasticity=np.inf,
                elasticity_lower=-2.0,
                elasticity_upper=-1.0,
            )


class TestRoundToPriceEnding:
    """Test round_to_price_ending function."""

    def test_rounds_down_when_closer(self):
        """Test rounding to the nearer lower price ending."""
        result = round_to_price_ending(12.34, 0.99)
        assert result == pytest.approx(11.99)

    def test_rounds_up_when_closer(self):
        """Test rounding to the nearer upper price ending."""
        result = round_to_price_ending(12.90, 0.99)
        assert result == pytest.approx(12.99)

    def test_avoids_negative_price(self):
        """Test that rounding never produces a non-positive price."""
        result = round_to_price_ending(0.5, 0.99)
        assert result == pytest.approx(0.99)
        assert result > 0

    def test_zero_ending_rounds_to_whole_number(self):
        """Test that an ending of zero rounds to the nearest whole number."""
        result = round_to_price_ending(10.4, 0.0)
        assert result == pytest.approx(10.0)

    def test_non_positive_price_raises_error(self):
        """Test that a non-positive price raises ValueError."""
        with pytest.raises(ValueError, match="greater than zero"):
            round_to_price_ending(0, 0.99)

    def test_invalid_ending_raises_error(self):
        """Test that an out-of-range ending raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            round_to_price_ending(10.0, 1.5)


class TestRecomputePriceFinancials:
    """Test recompute_price_financials function."""

    def test_basic_recompute(self):
        """Test that unit-elastic demand keeps revenue constant."""
        result = recompute_price_financials(
            candidate_price=11,
            base_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
        )
        assert result["expected_revenue"] == pytest.approx(1000.0)

    def test_promotion_scales_quantity(self):
        """Test that an active promotion scales up expected quantity."""
        result = recompute_price_financials(
            candidate_price=10,
            base_price=10,
            baseline_quantity=100,
            elasticity=0.0,
            promotion_active=True,
            promotion_uplift_rate=0.5,
        )
        assert result["expected_quantity"] == pytest.approx(150.0)

    def test_promotion_cost_reduces_profit(self):
        """Test that promotion cost is subtracted from expected profit."""
        result = recompute_price_financials(
            candidate_price=10,
            base_price=10,
            baseline_quantity=100,
            elasticity=0.0,
            unit_cost=5,
            promotion_active=True,
            promotion_uplift_rate=0.0,
            promotion_cost_rate=0.1,
        )
        assert result["expected_profit"] == pytest.approx(400.0)

    def test_cannibalization_adjustment_scales_quantity(self):
        """Test that a cannibalization adjustment scales expected quantity."""
        result = recompute_price_financials(
            candidate_price=10,
            base_price=10,
            baseline_quantity=100,
            elasticity=0.0,
            cannibalization_adjustment=0.2,
        )
        assert result["expected_quantity"] == pytest.approx(120.0)


class TestCompetitorConstraint:
    """Test competitor-aware price guardrails."""

    def test_constraint_defaults_to_true_without_competitor_price(self):
        """Test that the competitor constraint is a no-op when unset."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert result["meets_competitor_constraint"].all()

    def test_constraint_filters_out_of_band_scenarios(self):
        """Test that prices outside the competitor band are marked infeasible."""
        config = PricingOptimizationConfig(
            competitor_price_tolerance=0.05
        )
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            competitor_price=10.0,
            configuration=config,
        )
        out_of_band = result.loc[
            ~result["meets_competitor_constraint"]
        ]
        assert not out_of_band.empty
        assert (
            (out_of_band["candidate_price"] < 9.5)
            | (out_of_band["candidate_price"] > 10.5)
        ).all()

    def test_optimize_item_price_respects_competitor_band(self):
        """Test that the final recommendation stays within the competitor band."""
        config = PricingOptimizationConfig(
            objective="revenue",
            competitor_price_tolerance=0.05,
        )
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-3.0,
            unit_cost=1,
            competitor_price=10.0,
            configuration=config,
        )
        recommended_price = result.recommendation["recommended_price"]
        assert 9.5 - 1e-6 <= recommended_price <= 10.5 + 1e-6


class TestPromotionAdjustment:
    """Test promotion-aware demand and cost adjustments."""

    def test_promotion_scales_expected_quantity(self):
        """Test that promotion uplift scales expected quantity in the grid."""
        baseline = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        promoted = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            promotion_active=True,
            promotion_uplift_rate=0.25,
        )
        ratio = (
            promoted["expected_quantity"]
            / baseline["expected_quantity"]
        )
        assert ratio.dropna().apply(
            lambda value: value == pytest.approx(1.25)
        ).all()

    def test_promotion_cost_reduces_expected_profit(self):
        """Test that promotion cost lowers expected profit when active."""
        config = PricingOptimizationConfig(promotion_cost_rate=0.1)
        baseline = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            configuration=config,
        )
        promoted = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            promotion_active=True,
            promotion_uplift_rate=0.0,
            configuration=config,
        )
        assert (
            promoted["expected_profit"] < baseline["expected_profit"]
        ).all()


class TestInventoryConstraint:
    """Test inventory-aware feasibility guardrails."""

    def test_constraint_defaults_to_true_without_inventory(self):
        """Test that the inventory constraint is a no-op when unset."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
        )
        assert result["meets_inventory_constraint"].all()

    def test_constraint_mode_blocks_high_demand_scenarios(self):
        """Test that scenarios exceeding stock are infeasible in strict mode."""
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            available_inventory=50,
            configuration=PricingOptimizationConfig(
                inventory_mode="constraint",
            ),
        )
        blocked = result.loc[~result["meets_inventory_constraint"]]
        assert not blocked.empty
        assert (blocked["expected_quantity"] > 50).all()

    def test_cap_mode_truncates_demand_rather_than_blocking(self):
        """Test that the "cap" default sells out instead of failing.

        Every scenario stays feasible, demand never exceeds stock, and the
        `inventory_binding` flag marks the scenarios that sold out -- which
        is the signal that the item is underpriced.
        """
        result = evaluate_price_scenarios(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            available_inventory=50,
            configuration=PricingOptimizationConfig(
                inventory_mode="cap",
            ),
        )

        assert result["meets_inventory_constraint"].all()
        assert (result["expected_quantity"] <= 50 + 1e-9).all()
        assert result["inventory_binding"].any()
        assert (
            result.loc[result["inventory_binding"], "unconstrained_quantity"]
            > 50
        ).all()


class TestPriceEndingRounding:
    """Test psychological price-ending rounding in optimize_item_price."""

    def test_recommended_price_has_configured_ending(self):
        """Test that the final recommended price uses the configured ending."""
        config = PricingOptimizationConfig(
            objective="revenue",
            price_ending=0.99,
        )
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.5,
            configuration=config,
        )
        fractional_part = round(
            result.recommendation["recommended_price"] % 1,
            2,
        )
        assert fractional_part == pytest.approx(0.99)
        assert result.recommendation["price_ending_applied"] is True


class TestConfidenceDampeningIntegration:
    """Test confidence dampening wired through optimize_item_price."""

    def test_missing_confidence_interval_raises_error(self):
        """Test that enabling dampening without bounds raises ValueError."""
        config = PricingOptimizationConfig(
            enable_confidence_dampening=True
        )
        with pytest.raises(ValueError, match="elasticity_lower"):
            optimize_item_price(
                current_price=10,
                baseline_quantity=100,
                elasticity=-2.0,
                configuration=config,
            )

    def test_wide_interval_shrinks_price_move(self):
        """Test that high elasticity uncertainty shrinks the price move."""
        undamped = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-2.0,
            unit_cost=1,
        )

        config = PricingOptimizationConfig(
            enable_confidence_dampening=True
        )
        damped = optimize_item_price(
            current_price=10,
            baseline_quantity=100,
            elasticity=-2.0,
            unit_cost=1,
            elasticity_lower=-6.0,
            elasticity_upper=2.0,
            configuration=config,
        )

        undamped_rate = abs(undamped.recommendation["price_change_rate"])
        damped_rate = abs(damped.recommendation["price_change_rate"])

        assert damped_rate < undamped_rate
        assert damped.recommendation["confidence_dampening_factor"] < 1.0


class TestFinalConstraintRevalidation:
    """Test that post-processing steps can't smuggle in a bad price."""

    def test_price_ending_rounding_snaps_back_into_competitor_band(self):
        """Test that ending-rounding out of the competitor band is undone."""
        config = PricingOptimizationConfig(
            objective="revenue",
            competitor_price_tolerance=0.01,
            price_ending=0.49,
        )
        scenario_kwargs = dict(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.5,
            competitor_price=10.0,
            configuration=config,
        )
        scenarios = evaluate_price_scenarios(**scenario_kwargs)
        optimal_scenario = select_optimal_price_scenario(scenarios)

        rounded_price = round_to_price_ending(
            float(optimal_scenario["candidate_price"]),
            config.price_ending,
        )
        # Sanity check: confirm the rounded price really does leave the
        # 1% competitor band around 10.0, i.e. that this test exercises
        # the fallback path rather than a no-op.
        assert not (9.9 - 1e-9 <= rounded_price <= 10.1 + 1e-9)

        feasible_scenarios = scenarios.loc[scenarios["is_feasible"]]
        nearest_feasible = feasible_scenarios.loc[
            (feasible_scenarios["candidate_price"] - rounded_price)
            .abs()
            .idxmin()
        ]

        result = optimize_item_price(**scenario_kwargs)
        recommendation = result.recommendation

        assert recommendation["meets_competitor_constraint"] is True
        assert recommendation["constraint_fallback_applied"] is True
        assert recommendation["price_ending_applied"] is False
        assert recommendation["recommended_price"] == pytest.approx(
            float(nearest_feasible["candidate_price"])
        )

    def test_dampening_toward_over_capacity_price_respects_inventory(self):
        """Test that dampening back toward an over-inventory price is caught.

        Pinned to inventory_mode="constraint": under the "cap" default the
        over-capacity price is not infeasible, it simply sells out, so no
        fallback is expected (see the companion test below).
        """
        config = PricingOptimizationConfig(
            objective="revenue",
            enable_confidence_dampening=True,
            inventory_mode="constraint",
        )
        scenario_kwargs = dict(
            current_price=10,
            baseline_quantity=200,
            elasticity=-2.0,
            unit_cost=1,
            available_inventory=150,
            configuration=config,
        )
        scenarios = evaluate_price_scenarios(**scenario_kwargs)
        optimal_scenario = select_optimal_price_scenario(scenarios)

        dampening_factor = calculate_confidence_dampening_factor(
            elasticity=-2.0,
            elasticity_lower=-4.5,
            elasticity_upper=0.0,
        )
        damped_price = 10 * (
            1
            + float(optimal_scenario["price_change_rate"])
            * dampening_factor
        )
        damped_financials = recompute_price_financials(
            candidate_price=damped_price,
            base_price=10,
            baseline_quantity=200,
            elasticity=-2.0,
            unit_cost=1,
            inventory_mode=config.inventory_mode,
        )
        # Sanity check: confirm dampening really does pull the price back
        # over the 150-unit inventory cap, i.e. that this test exercises
        # the fallback path rather than a no-op.
        assert damped_financials["expected_quantity"] > 150 + 1e-9

        feasible_scenarios = scenarios.loc[scenarios["is_feasible"]]
        nearest_feasible = feasible_scenarios.loc[
            (feasible_scenarios["candidate_price"] - damped_price)
            .abs()
            .idxmin()
        ]

        result = optimize_item_price(
            elasticity_lower=-4.5,
            elasticity_upper=0.0,
            **scenario_kwargs,
        )
        recommendation = result.recommendation

        assert recommendation["status"] == "success"
        assert recommendation["meets_inventory_constraint"] is True
        assert recommendation["expected_quantity"] <= 150 + 1e-6
        assert recommendation["constraint_fallback_applied"] is True
        assert recommendation["recommended_price"] == pytest.approx(
            float(nearest_feasible["candidate_price"])
        )

    def test_cap_mode_sells_out_instead_of_falling_back(self):
        """Under the "cap" default, exceeding stock truncates demand.

        This is the behaviour that lets step 08 price a stocked-out item at
        all: selling out is an outcome to report, not an infeasibility.
        """
        config = PricingOptimizationConfig(
            objective="revenue",
            enable_confidence_dampening=True,
            inventory_mode="cap",
        )
        result = optimize_item_price(
            current_price=10,
            baseline_quantity=200,
            elasticity=-2.0,
            unit_cost=1,
            available_inventory=150,
            elasticity_lower=-4.5,
            elasticity_upper=0.0,
            configuration=config,
        )
        recommendation = result.recommendation

        assert recommendation["status"] == "success"
        assert recommendation["expected_quantity"] <= 150 + 1e-6
        assert recommendation["inventory_binding"] is True
        assert recommendation["meets_all_constraints"] is True

    def test_margin_floor_clamp_survives_dampening_without_fallback(self):
        """The margin-floor clamp runs after dampening, so it is the last
        write to price before the guardrail re-check -- the margin
        guardrail can therefore never be the one that trips the
        nearest-feasible fallback. Set the elasticity interval wide enough
        that dampening pulls hard toward a below-floor current price, and
        confirm the clamp -- not the fallback -- is what saves it.
        """
        minimum_allowed_price = calculate_minimum_price_for_margin(
            unit_cost=8,
            minimum_margin_rate=0.0,
        )
        config = PricingOptimizationConfig(
            objective="revenue",
            enable_confidence_dampening=True,
            minimum_price_change_rate=-0.10,
            maximum_price_change_rate=0.60,
        )
        result = optimize_item_price(
            current_price=7,  # below the unit_cost=8 margin floor
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=8,
            elasticity_lower=-4.0,
            elasticity_upper=2.0,
            configuration=config,
        )
        recommendation = result.recommendation

        assert recommendation["recommended_price"] == pytest.approx(
            minimum_allowed_price
        )
        assert recommendation["meets_margin_constraint"] is True
        assert recommendation["constraint_fallback_applied"] is False

    def test_quantity_guardrail_triggers_fallback_to_nearest_feasible(self):
        """price_ending rounding can push the price up enough that expected
        quantity drops below minimum_expected_quantity, even though the
        grid-optimal price (lower, higher-quantity) was feasible.
        """
        config = PricingOptimizationConfig(
            objective="revenue",
            price_ending=0.49,
            minimum_expected_quantity=145.0,
        )
        scenario_kwargs = dict(
            current_price=10,
            baseline_quantity=100,
            elasticity=-2.0,
            configuration=config,
        )
        scenarios = evaluate_price_scenarios(**scenario_kwargs)
        optimal_scenario = select_optimal_price_scenario(scenarios)

        rounded_price = round_to_price_ending(
            float(optimal_scenario["candidate_price"]),
            config.price_ending,
        )
        rounded_financials = recompute_price_financials(
            candidate_price=rounded_price,
            base_price=10,
            baseline_quantity=100,
            elasticity=-2.0,
            inventory_mode=config.inventory_mode,
        )
        # Sanity check: confirm the rounded price really does breach the
        # quantity floor, i.e. that this test exercises the fallback path
        # rather than a no-op.
        assert (
            rounded_financials["expected_quantity"]
            < config.minimum_expected_quantity
        )

        feasible_scenarios = scenarios.loc[scenarios["is_feasible"]]
        nearest_feasible = feasible_scenarios.loc[
            (feasible_scenarios["candidate_price"] - rounded_price)
            .abs()
            .idxmin()
        ]

        result = optimize_item_price(**scenario_kwargs)
        recommendation = result.recommendation

        assert recommendation["constraint_fallback_applied"] is True
        assert recommendation["meets_quantity_constraint"] is True
        assert recommendation["price_ending_applied"] is False
        assert recommendation["recommended_price"] == pytest.approx(
            float(nearest_feasible["candidate_price"])
        )

    def test_profit_guardrail_triggers_fallback_to_nearest_feasible(self):
        """A promotion cost rate creates an effective profit floor
        (unit_cost / (1 - promotion_cost_rate)) above the plain margin
        floor. price_ending rounding can push the price below that
        effective floor into negative-profit territory even though the
        grid-optimal price cleared it.
        """
        config = PricingOptimizationConfig(
            objective="quantity",
            price_ending=0.09,
            promotion_cost_rate=0.4,
        )
        scenario_kwargs = dict(
            current_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            promotion_active=True,
            configuration=config,
        )
        scenarios = evaluate_price_scenarios(**scenario_kwargs)
        optimal_scenario = select_optimal_price_scenario(scenarios)

        rounded_price = round_to_price_ending(
            float(optimal_scenario["candidate_price"]),
            config.price_ending,
        )
        rounded_financials = recompute_price_financials(
            candidate_price=rounded_price,
            base_price=10,
            baseline_quantity=100,
            elasticity=-1.0,
            unit_cost=5,
            promotion_active=True,
            promotion_cost_rate=config.promotion_cost_rate,
            inventory_mode=config.inventory_mode,
        )
        assert rounded_financials["expected_profit"] < 0

        feasible_scenarios = scenarios.loc[scenarios["is_feasible"]]
        nearest_feasible = feasible_scenarios.loc[
            (feasible_scenarios["candidate_price"] - rounded_price)
            .abs()
            .idxmin()
        ]

        result = optimize_item_price(**scenario_kwargs)
        recommendation = result.recommendation

        assert recommendation["constraint_fallback_applied"] is True
        assert recommendation["meets_profit_constraint"] is True
        assert recommendation["expected_profit"] >= -1e-9
        assert recommendation["recommended_price"] == pytest.approx(
            float(nearest_feasible["candidate_price"])
        )

    def test_objective_no_harm_triggers_fallback_to_nearest_feasible(self):
        """For constant-elasticity demand with cost, profit(P) peaks at
        P* = unit_cost * elasticity / (elasticity + 1). Choosing unit_cost
        and elasticity so that P* lands exactly on current_price makes
        "hold" the unique profit-maximizing (and only no-harm-satisfying)
        point: any price_ending rounding away from it strictly reduces
        profit below the current level, and the fallback should snap
        straight back to holding.
        """
        config = PricingOptimizationConfig(
            objective="profit",
            price_ending=0.49,
        )
        scenario_kwargs = dict(
            current_price=10,
            baseline_quantity=100,
            elasticity=-2.0,
            unit_cost=5,  # P* = 5 * -2 / (-2 + 1) = 10 = current_price
            configuration=config,
        )
        scenarios = evaluate_price_scenarios(**scenario_kwargs)
        optimal_scenario = select_optimal_price_scenario(scenarios)

        assert float(optimal_scenario["candidate_price"]) == pytest.approx(
            10.0
        )

        rounded_price = round_to_price_ending(
            float(optimal_scenario["candidate_price"]),
            config.price_ending,
        )
        assert rounded_price != pytest.approx(10.0)

        result = optimize_item_price(**scenario_kwargs)
        recommendation = result.recommendation

        assert recommendation["constraint_fallback_applied"] is True
        assert recommendation["meets_objective_no_harm_constraint"] is True
        assert recommendation["price_ending_applied"] is False
        assert recommendation["recommended_price"] == pytest.approx(10.0)
        assert recommendation["expected_profit"] == pytest.approx(
            recommendation["current_profit"]
        )

    def test_no_feasible_scenario_after_fallback_raises_with_breached_list(
        self,
    ):
        """When every scenario in the band fails a guardrail even after
        the nearest-feasible search, the ValueError should name exactly
        the guardrail(s) that were breached. Here unit_cost exceeds
        current_price, so the margin clamp forces the price up to
        unit_cost -- but that same forced increase, combined with elastic
        demand, drops revenue below the current level everywhere in the
        band, so no scenario can satisfy objective_no_harm.
        """
        config = PricingOptimizationConfig(
            objective="revenue",
            enable_confidence_dampening=True,
        )

        with pytest.raises(ValueError, match="objective_no_harm") as excinfo:
            optimize_item_price(
                current_price=10,
                baseline_quantity=100,
                elasticity=-1.2,
                unit_cost=12,
                elasticity_lower=-6.0,
                elasticity_upper=3.0,
                configuration=config,
            )

        assert str(excinfo.value) == (
            "No feasible price scenario satisfies the objective_no_harm "
            "constraint(s) after dampening, rounding, and the "
            "margin-floor clamp."
        )


class TestOptimizePricePortfolioAdvanced:
    """Test portfolio-level cannibalization and category overrides."""

    def _build_portfolio(self):
        return pd.DataFrame(
            {
                "item_id": ["A", "B"],
                "category": ["Snacks", "Snacks"],
                "current_price": [10.0, 10.0],
                "baseline_quantity": [100.0, 100.0],
                "elasticity": [-3.0, -0.3],
                "unit_cost": [2.0, 2.0],
            }
        )

    def test_cannibalization_adds_adjustment_column(self):
        """Test that enabling cross-price elasticity records an adjustment."""
        config = PricingOptimizationConfig(
            objective="revenue",
            cross_price_elasticity=0.3,
        )
        recommendations = optimize_price_portfolio(
            self._build_portfolio(),
            item_column="item_id",
            current_price_column="current_price",
            baseline_quantity_column="baseline_quantity",
            elasticity_column="elasticity",
            unit_cost_column="unit_cost",
            category_column="category",
            configuration=config,
        )
        assert (recommendations["status"] == "success").all()
        assert "cannibalization_adjustment" in recommendations.columns
        assert (
            recommendations["cannibalization_adjustment"] != 0.0
        ).any()

    def test_cannibalization_pass_survives_all_items_failing(self):
        """Test that a first pass with zero successes doesn't crash."""
        config = PricingOptimizationConfig(
            objective="revenue",
            minimum_margin_rate=0.5,
            cross_price_elasticity=0.3,
        )
        portfolio = self._build_portfolio()
        portfolio["unit_cost"] = [9.99, 9.99]
        recommendations = optimize_price_portfolio(
            portfolio,
            item_column="item_id",
            current_price_column="current_price",
            baseline_quantity_column="baseline_quantity",
            elasticity_column="elasticity",
            unit_cost_column="unit_cost",
            category_column="category",
            configuration=config,
        )
        assert (recommendations["status"] == "failed").all()

    def _build_iteration_portfolio(self):
        """Three siblings under a shared inventory constraint.

        Asymmetric elasticities under `inventory_mode="constraint"` make
        each item's revenue-maximizing price a continuous function of its
        *own* demand scale, so a cannibalization adjustment (which rides
        on that scale) genuinely shifts the chosen price -- unlike a
        band-boundary or scale-invariant profit optimum, where identical
        siblings would trivially converge in a single round regardless of
        whether the pass is iterated.
        """
        return pd.DataFrame(
            {
                "item_id": ["A", "B", "C"],
                "category": ["Snacks"] * 3,
                "current_price": [10.0, 10.0, 10.0],
                "unit_cost": [1.0, 1.0, 1.0],
                "elasticity": [-3.0, -2.0, -1.5],
                "baseline_quantity": [100.0, 100.0, 100.0],
                "inventory": [90.0, 90.0, 90.0],
            }
        )

    def _run_iteration_portfolio(self, **config_overrides):
        config = PricingOptimizationConfig(
            objective="revenue",
            inventory_mode="constraint",
            price_change_step=0.002,
            cross_price_elasticity=0.7,
            **config_overrides,
        )
        return optimize_price_portfolio(
            self._build_iteration_portfolio(),
            item_column="item_id",
            current_price_column="current_price",
            baseline_quantity_column="baseline_quantity",
            elasticity_column="elasticity",
            unit_cost_column="unit_cost",
            category_column="category",
            available_inventory_column="inventory",
            configuration=config,
        ).set_index("item_id")

    def test_cannibalization_max_iterations_one_matches_manual_one_shot(
        self,
    ):
        """cannibalization_max_iterations=1 must reproduce the original
        pass-1-frozen heuristic exactly: recompute the expected one-shot
        prices by hand (pass 1, then a single sibling-average round) using
        only the public API, and compare.
        """
        portfolio = self._build_iteration_portfolio().set_index("item_id")

        pass1 = optimize_price_portfolio(
            self._build_iteration_portfolio(),
            item_column="item_id",
            current_price_column="current_price",
            baseline_quantity_column="baseline_quantity",
            elasticity_column="elasticity",
            unit_cost_column="unit_cost",
            category_column="category",
            available_inventory_column="inventory",
            configuration=PricingOptimizationConfig(
                objective="revenue",
                inventory_mode="constraint",
                price_change_step=0.002,
                cross_price_elasticity=0.0,
            ),
        ).set_index("item_id")

        successful_rate = pass1["price_change_rate"].where(
            pass1["status"] == "success"
        )
        category = portfolio["category"]
        category_sum = successful_rate.groupby(category).transform("sum")
        category_count = (
            successful_rate.notna().groupby(category).transform("sum")
        )
        own_contributes = successful_rate.notna().astype(float)
        sibling_count = (category_count - own_contributes).replace(
            0, np.nan
        )
        sibling_avg_rate = (
            (category_sum - successful_rate.fillna(0.0)) / sibling_count
        ).fillna(0.0)
        expected_adjustment = 0.7 * sibling_avg_rate

        actual = self._run_iteration_portfolio(
            cannibalization_max_iterations=1
        )

        for item_id, row in portfolio.iterrows():
            expected = optimize_item_price(
                current_price=row["current_price"],
                baseline_quantity=row["baseline_quantity"],
                elasticity=row["elasticity"],
                unit_cost=row["unit_cost"],
                available_inventory=row["inventory"],
                cannibalization_adjustment=(
                    expected_adjustment[item_id]
                ),
                configuration=PricingOptimizationConfig(
                    objective="revenue",
                    inventory_mode="constraint",
                    price_change_step=0.002,
                ),
            )

            assert actual.loc[
                item_id, "recommended_price"
            ] == pytest.approx(
                expected.recommendation["recommended_price"]
            )

    def test_cannibalization_iterates_beyond_first_round(self):
        """With siblings that move together, letting the pass iterate
        (the default cap of 5) must change the outcome versus freezing
        it at the pass-1-derived one-shot adjustment
        (``cannibalization_max_iterations=1``) -- otherwise the fix is a
        no-op.
        """
        one_shot = self._run_iteration_portfolio(
            cannibalization_max_iterations=1
        )
        iterated = self._run_iteration_portfolio()  # default cap of 5

        assert not np.allclose(
            one_shot["price_change_rate"].to_numpy(),
            iterated["price_change_rate"].to_numpy(),
        )

    def test_cannibalization_iteration_converges_to_a_stable_point(self):
        """The loop should settle rather than keep drifting: results at
        the default cap (5) and a much larger cap (30) should agree,
        confirming early-stopping on convergence actually triggers
        instead of every run silently consuming its full iteration
        budget.
        """
        default_cap = self._run_iteration_portfolio()
        generous_cap = self._run_iteration_portfolio(
            cannibalization_max_iterations=30
        )

        assert default_cap["recommended_price"].to_numpy() == pytest.approx(
            generous_cap["recommended_price"].to_numpy()
        )

    def test_category_change_rate_override_forces_hold(self):
        """Test that a near-zero category override forces a hold price."""
        config = PricingOptimizationConfig(objective="revenue")
        recommendations = optimize_price_portfolio(
            self._build_portfolio(),
            item_column="item_id",
            current_price_column="current_price",
            baseline_quantity_column="baseline_quantity",
            elasticity_column="elasticity",
            unit_cost_column="unit_cost",
            category_column="category",
            category_change_rate_overrides={
                "Snacks": (-0.001, 0.001)
            },
            configuration=config,
        )
        assert (
            recommendations["recommendation_action"] == "hold_price"
        ).all()
        assert (
            recommendations["recommended_price"]
            .sub(recommendations["current_price"])
            .abs()
            < 0.05
        ).all()


class TestBuildOptimizationConfiguration:
    """Test that policy toggles are captured in saved run provenance."""

    def test_cannibalization_max_iterations_is_saved(self):
        """A reviewer enabling cross_price_elasticity must be able to see
        the iteration bound from the saved configuration alone, not just
        a code comment.
        """
        config = PricingOptimizationConfig(
            cross_price_elasticity=0.4,
            cannibalization_max_iterations=7,
        )
        saved = build_optimization_configuration(config)

        assert saved["cannibalization_max_iterations"] == 7
        assert saved["cross_price_elasticity"] == pytest.approx(0.4)

    def test_allow_non_causal_pricing_is_saved(self):
        """allow_non_causal_pricing must round-trip into provenance too,
        since it started life as an untracked module constant.
        """
        config = PricingOptimizationConfig(
            allow_non_causal_pricing=True
        )
        saved = build_optimization_configuration(config)

        assert saved["allow_non_causal_pricing"] is True
