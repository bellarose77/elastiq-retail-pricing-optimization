"""
Regression tests for defects found in the July 2026 review.

Each test names the defect it pins and would have failed before the fix.
These are behavioural tests -- they assert the pipeline produces defensible
numbers, which is the class of test the suite previously lacked entirely.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR, RAW_DIR
from src.data.splitting import create_chronological_split
from src.models.rag import (
    build_rag_corpus,
    prepare_document_chunks,
    retrieve_market_evidence,
)
from src.optimization.pricing import (
    PricingOptimizationConfig,
    generate_price_change_rates,
    optimize_item_price,
    optimize_price_portfolio,
    validate_price_recommendations,
)


class TestTemporalIntegrity:
    """Panel dates and market evidence must never cross time boundaries."""

    def test_panel_split_keeps_each_date_in_one_partition(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        frame = pd.DataFrame(
            {
                "date": np.repeat(dates, 4),
                "item": list(range(40)),
            }
        )

        train, validation, test = create_chronological_split(
            frame,
            "date",
            train_fraction=0.6,
            val_fraction=0.2,
        )

        assert set(train["date"]).isdisjoint(validation["date"])
        assert set(train["date"]).isdisjoint(test["date"])
        assert set(validation["date"]).isdisjoint(test["date"])
        assert train["date"].max() < validation["date"].min()
        assert validation["date"].max() < test["date"].min()

    def test_rag_retrieval_cannot_see_future_or_other_region(self):
        documents = pd.DataFrame(
            {
                "text": ["demand rises", "future demand", "other region"],
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-03-01", "2026-01-01"]
                ),
                "category_key": ["grocery", "grocery", "grocery"],
                "region": ["East", "East", "West"],
            }
        )
        chunks = prepare_document_chunks(
            documents,
            text_column="text",
            metadata_columns=["date", "category_key", "region"],
        )
        corpus = build_rag_corpus(
            chunks,
            metadata_columns=["date", "category_key", "region"],
        )

        result = retrieve_market_evidence(
            "grocery demand",
            corpus,
            top_k=10,
            metadata_filters={
                "category_key": "grocery",
                "region": "East",
            },
            as_of_date="2026-02-01",
        )

        assert len(result) == 1
        assert result.iloc[0]["chunk_text"] == "demand rises"
        assert pd.Timestamp(result.iloc[0]["date"]) <= pd.Timestamp(
            "2026-02-01"
        )


class TestPriceChangeGrid:
    """The no-change scenario must respect the configured band."""

    def test_hold_is_not_injected_into_an_increase_only_band(self):
        grid = generate_price_change_rates(
            minimum_change_rate=0.05,
            maximum_change_rate=0.20,
            step=0.05,
        )

        assert 0.0 not in grid
        assert grid.min() >= 0.05

    def test_hold_is_not_injected_into_a_markdown_only_band(self):
        grid = generate_price_change_rates(
            minimum_change_rate=-0.20,
            maximum_change_rate=-0.05,
            step=0.05,
        )

        assert 0.0 not in grid
        assert grid.max() <= -0.05

    def test_hold_is_still_available_in_a_band_spanning_zero(self):
        grid = generate_price_change_rates(
            minimum_change_rate=-0.10,
            maximum_change_rate=0.10,
            step=0.05,
        )

        assert 0.0 in grid

    def test_mandated_increase_does_not_return_hold(self):
        config = PricingOptimizationConfig(
            minimum_price_change_rate=0.05,
            maximum_price_change_rate=0.20,
            price_change_step=0.05,
        )

        result = optimize_item_price(
            current_price=10.0,
            baseline_quantity=100.0,
            elasticity=-3.0,
            unit_cost=2.0,
            configuration=config,
        )

        assert result.recommendation["recommendation_action"] == (
            "increase_price"
        )
        assert result.recommendation["recommended_price"] > 10.0


class TestPostProcessingRevalidation:
    """Rounding and dampening must not breach guardrails silently."""

    def test_price_ending_cannot_breach_the_quantity_floor(self):
        config = PricingOptimizationConfig(
            minimum_expected_quantity=95.0,
            price_ending=0.99,
            minimum_price_change_rate=-0.20,
            maximum_price_change_rate=0.20,
        )

        recommendation = optimize_item_price(
            current_price=10.0,
            baseline_quantity=100.0,
            elasticity=-1.0,
            unit_cost=1.0,
            configuration=config,
        ).recommendation

        assert recommendation["expected_quantity"] >= 95.0 - 1e-6
        assert recommendation["meets_quantity_constraint"]
        assert recommendation["meets_all_constraints"]

    def test_inventory_cap_is_applied_to_current_and_recommended_profit(self):
        recommendation = optimize_item_price(
            current_price=10.0,
            baseline_quantity=100.0,
            elasticity=-2.0,
            unit_cost=5.0,
            available_inventory=20.0,
            configuration=PricingOptimizationConfig(
                objective="profit",
                inventory_mode="cap",
                price_ending=0.99,
                enable_confidence_dampening=True,
            ),
            elasticity_lower=-2.4,
            elasticity_upper=-1.6,
        ).recommendation

        assert recommendation["current_quantity"] == pytest.approx(20.0)
        assert recommendation["current_profit"] == pytest.approx(100.0)
        assert recommendation["expected_profit"] >= (
            recommendation["current_profit"] - 1e-9
        )
        assert recommendation["meets_objective_no_harm_constraint"]

    def test_every_constraint_flag_is_reported(self):
        recommendation = optimize_item_price(
            current_price=10.0,
            baseline_quantity=100.0,
            elasticity=-1.5,
            unit_cost=4.0,
            configuration=PricingOptimizationConfig(),
        ).recommendation

        for flag in (
            "meets_margin_constraint",
            "meets_quantity_constraint",
            "meets_profit_constraint",
            "meets_competitor_constraint",
            "meets_inventory_constraint",
            "meets_all_constraints",
        ):
            assert flag in recommendation


class TestInventorySemantics:
    """Stock on hand caps demand; it does not make an item unpriceable."""

    ITEM = dict(
        current_price=9.08,
        baseline_quantity=34.0,
        elasticity=-0.0915,
        unit_cost=4.37,
        competitor_price=8.81,
        available_inventory=34.0,
        cannibalization_adjustment=0.021,
    )

    BASE_CONFIG = dict(
        minimum_margin_rate=0.15,
        minimum_price_change_rate=-0.10,
        maximum_price_change_rate=0.20,
        competitor_price_tolerance=0.15,
        promotion_cost_rate=0.02,
    )

    def test_cap_mode_prices_a_stocked_out_item(self):
        config = PricingOptimizationConfig(
            inventory_mode="cap",
            **self.BASE_CONFIG,
        )

        recommendation = optimize_item_price(
            configuration=config,
            **self.ITEM,
        ).recommendation

        assert recommendation["status"] == "success"
        assert recommendation["expected_quantity"] <= 34.0 + 1e-9
        assert recommendation["inventory_binding"]

    def test_constraint_mode_still_available_and_still_strict(self):
        config = PricingOptimizationConfig(
            inventory_mode="constraint",
            **self.BASE_CONFIG,
        )

        with pytest.raises(ValueError):
            optimize_item_price(configuration=config, **self.ITEM)

    def test_invalid_inventory_mode_is_rejected(self):
        with pytest.raises(ValueError, match="inventory_mode"):
            optimize_item_price(
                current_price=10.0,
                baseline_quantity=10.0,
                elasticity=-1.5,
                configuration=PricingOptimizationConfig(
                    inventory_mode="truncate",
                ),
            )


class TestCannibalizationSiblingAverage:
    """A failed item must not inflate its siblings' average price move."""

    def test_failed_rows_do_not_double_the_sibling_average(self):
        frame = pd.DataFrame(
            {
                "item": ["A1", "A2", "A3"],
                "cat": ["A", "A", "A"],
                "price": [10.0, 10.0, 10.0],
                "qty": [100.0, 100.0, 100.0],
                "elas": [-2.0, -2.0, np.nan],
                "cost": [3.0, 3.0, 3.0],
            }
        )

        result = optimize_price_portfolio(
            frame,
            item_column="item",
            current_price_column="price",
            baseline_quantity_column="qty",
            elasticity_column="elas",
            unit_cost_column="cost",
            category_column="cat",
            configuration=PricingOptimizationConfig(
                cross_price_elasticity=0.15,
            ),
        )

        succeeded = result.loc[result["status"] == "success"]

        # Two identical siblings must see each other's move, not double it.
        adjustments = succeeded["cannibalization_adjustment"].unique()

        assert len(adjustments) == 1

        sibling_rate = succeeded["price_change_rate"].iloc[0]

        assert adjustments[0] == pytest.approx(
            0.15 * sibling_rate,
            rel=1e-6,
        )


class TestPortfolioErrorHandling:
    """Programming errors must surface, not become 'review' rows."""

    def test_missing_column_raises_instead_of_failing_silently(self):
        frame = pd.DataFrame(
            {
                "item": ["A1"],
                "price": [10.0],
                "qty": [100.0],
                "elas": [-1.5],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            optimize_price_portfolio(
                frame,
                item_column="item",
                current_price_column="price",
                baseline_quantity_column="qty",
                elasticity_column="does_not_exist",
            )


class TestRecommendationValidation:
    """Validation must audit business guardrails, not just arithmetic."""

    def test_out_of_band_price_is_caught(self):
        recommendations = pd.DataFrame(
            [
                {
                    "item_id": "A1",
                    "status": "success",
                    "current_price": 10.0,
                    # +50% is far outside the configured +/-20% band
                    "recommended_price": 15.0,
                    "price_change_rate": 0.5,
                    "unit_cost": 4.0,
                    "expected_quantity": 50.0,
                    "expected_profit": 550.0,
                }
            ]
        )

        validated = validate_price_recommendations(
            recommendations,
            configuration=PricingOptimizationConfig(),
        )

        assert not bool(validated["price_change_within_band"].iloc[0])
        assert validated["validation_status"].iloc[0] == "failed"

    def test_unscored_items_are_not_counted_as_quality_failures(self):
        recommendations = pd.DataFrame(
            [
                {
                    "item_id": "A1",
                    "status": "held_no_causal_elasticity",
                    "current_price": 10.0,
                    "recommended_price": 10.0,
                    "price_change_rate": 0.0,
                    "unit_cost": 4.0,
                }
            ]
        )

        validated = validate_price_recommendations(
            recommendations,
            configuration=PricingOptimizationConfig(),
        )

        assert validated["validation_status"].iloc[0] == "not_scored"


class TestElasticityGroundTruthRecovery:
    """
    The synthetic generator writes the true elasticity into the data. Any
    estimator claiming to measure price response must recover it.

    This is the test whose absence let step 08 ship recommendations built on
    elasticities with the wrong SIGN for 8 of 10 products.
    """

    TRUTH_FILE = RAW_DIR / "synthetic" / "sample_retail_pricing.csv"
    IV_FILE = PROCESSED_DIR / "iv_product_elasticity_estimates.csv"

    @pytest.fixture(scope="class")
    @classmethod
    def comparison(cls):
        if not (cls.TRUTH_FILE.exists() and cls.IV_FILE.exists()):
            pytest.skip(
                "requires the synthetic dataset and step 04 output; "
                "run the pipeline first"
            )

        truth = (
            pd.read_csv(cls.TRUTH_FILE)
            .groupby("product_id")["true_price_elasticity"]
            .first()
        )

        estimates = pd.read_csv(cls.IV_FILE).set_index("product_id")

        return pd.DataFrame(
            {
                "true": truth,
                "estimate": estimates["iv_elasticity"],
                "lower": estimates["confidence_interval_lower"],
                "upper": estimates["confidence_interval_upper"],
                "reliable": estimates["is_reliable"],
            }
        ).dropna(subset=["estimate"])

    def test_sign_is_always_correct(self, comparison):
        assert (comparison["estimate"] < 0).all()

    def test_point_estimates_are_close_to_truth(self, comparison):
        mean_absolute_error = float(
            (comparison["estimate"] - comparison["true"]).abs().mean()
        )

        # The correlational estimator this replaced scored 1.147.
        assert mean_absolute_error < 0.40

    def test_confidence_intervals_are_calibrated(self, comparison):
        covered = (
            (comparison["lower"] <= comparison["true"])
            & (comparison["true"] <= comparison["upper"])
        )

        assert covered.mean() >= 0.80

    def test_elastic_items_are_classified_as_elastic(self, comparison):
        agrees = (
            comparison["estimate"].abs() > 1
        ) == (comparison["true"].abs() > 1)

        assert agrees.mean() >= 0.80


class TestStepEightProvenance:
    """Every priced item must declare the evidence behind its price."""

    RECOMMENDATIONS = (
        PROCESSED_DIR / "price_optimization_recommendations.csv"
    )

    def test_recommendations_carry_causal_provenance(self):
        if not self.RECOMMENDATIONS.exists():
            pytest.skip("run step 08 first")

        recommendations = pd.read_csv(self.RECOMMENDATIONS)

        assert "elasticity_source" in recommendations.columns
        assert "elasticity_is_causal" in recommendations.columns

        priced = recommendations.loc[
            recommendations["status"] == "success"
        ]

        assert bool(priced["elasticity_is_causal"].all())


class TestParityFixtureIsCurrent:
    """The JS engine is validated against this fixture; it must exist."""

    FIXTURE = (
        PROCESSED_DIR.parent.parent
        / "tests"
        / "fixtures"
        / "engine_parity.json"
    )

    def test_fixture_exists_and_matches_python_defaults(self):
        if not self.FIXTURE.exists():
            pytest.skip(
                "run scripts/generate_parity_fixture.py"
            )

        fixture = json.loads(
            self.FIXTURE.read_text(encoding="utf-8")
        )

        defaults = PricingOptimizationConfig()

        assert fixture["python_defaults"]["maxPriceChangeRate"] == (
            defaults.maximum_price_change_rate
        )
        assert fixture["python_defaults"]["actionThreshold"] == (
            defaults.action_threshold
        )
        assert fixture["python_defaults"]["inventoryMode"] == (
            defaults.inventory_mode
        )
