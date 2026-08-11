# %%
"""Price optimization, scenario analysis, and recommendation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import validate_required_columns
from src.models.elasticity import predict_quantity_from_elasticity


@dataclass(slots=True)
class PricingOptimizationConfig:
    """Configuration settings for item-level price optimization."""

    objective: str = "profit"
    minimum_price_change_rate: float = -0.20
    maximum_price_change_rate: float = 0.20
    price_change_step: float = 0.01
    minimum_margin_rate: float = 0.0
    minimum_expected_quantity: float = 0.0
    action_threshold: float = 0.005
    enforce_nonnegative_profit: bool = True

    # Safety gate: price only on causally identified elasticities (e.g. IV
    # estimates) by default. Correlational (OLS/shrunk) elasticities are
    # confounded by the retailer's own price-setting behaviour and can be
    # biased enough to invert the pricing prescription. Setting this to
    # True opts an entire run in to pricing on correlational estimates as
    # well; because it is a dataclass field it is captured verbatim in
    # every run's saved configuration/provenance, unlike a bare module
    # constant that could be flipped locally with no audit trail.
    allow_non_causal_pricing: bool = False

    # Uncertainty-aware pricing: shrink the recommended price move toward
    # "hold" when the elasticity estimate's confidence interval is wide.
    enable_confidence_dampening: bool = False

    # Competitor-aware guardrail: reject candidate prices that stray more
    # than this fraction away from the supplied competitor price.
    competitor_price_tolerance: float | None = None

    # Promotion-aware cost: fraction of promoted revenue treated as
    # promotion cost and subtracted from expected profit.
    promotion_cost_rate: float = 0.0

    # Cross-item cannibalization: how strongly a category sibling's price
    # move shifts this item's expected demand (0 disables the effect).
    cross_price_elasticity: float = 0.0

    # Cap on the cannibalization fixed-point iteration: each round
    # recomputes every item's sibling-price pressure from the previous
    # round's own results and re-optimizes, stopping early once no item's
    # price-change rate moves by more than a small tolerance. This is a
    # heuristic on a portfolio, not a proven-convergent joint solver, so
    # a hard cap is required regardless of how well-behaved
    # cross_price_elasticity is. 1 reproduces the original one-shot
    # (pass-1-frozen) behaviour. Saved into run provenance so a reviewer
    # enabling cross_price_elasticity can see the iteration bound that
    # applied, rather than discovering it only in a code comment.
    cannibalization_max_iterations: int = 5

    # Psychological pricing: round the final recommended price to this
    # fractional ending (e.g. 0.99). None disables rounding.
    price_ending: float | None = None

    # How ``available_inventory`` enters the problem.
    #
    #   "cap"        -- realized demand is truncated at available stock.
    #                   Selling out is a feasible (indeed common) outcome,
    #                   and because revenue is then price x inventory, the
    #                   optimizer is correctly pushed toward raising price
    #                   on an item that is stocking out.
    #   "constraint" -- any candidate price whose expected demand exceeds
    #                   available stock is rejected as infeasible.
    #
    # "constraint" was the historical behaviour. It makes every item whose
    # observed demand already meets or exceeds its on-hand stock (i.e.
    # every stockout day) unsolvable, because no price in the permitted
    # band can pull demand under the cap for an inelastic item. "cap" is
    # the default because it matches both retail reality and the
    # behaviour of the ELASTIQ JavaScript engine.
    inventory_mode: str = "cap"


@dataclass(slots=True)
class PriceOptimizationResult:
    """Store an optimized price and its evaluated scenario table."""

    recommendation: dict[str, Any]
    scenarios: pd.DataFrame
    configuration: PricingOptimizationConfig
    
# %%
def validate_pricing_configuration(
    configuration: PricingOptimizationConfig,
) -> None:
    """Validate price-optimization configuration values."""

    valid_objectives = {
        "profit",
        "revenue",
        "quantity",
    }

    if configuration.objective not in valid_objectives:
        raise ValueError(
            "objective must be one of: "
            f"{sorted(valid_objectives)}."
        )

    if (
        configuration.minimum_price_change_rate
        >= configuration.maximum_price_change_rate
    ):
        raise ValueError(
            "minimum_price_change_rate must be smaller than "
            "maximum_price_change_rate."
        )

    if configuration.minimum_price_change_rate <= -1:
        raise ValueError(
            "minimum_price_change_rate must be greater than -1."
        )

    if configuration.price_change_step <= 0:
        raise ValueError(
            "price_change_step must be greater than zero."
        )

    if not 0 <= configuration.minimum_margin_rate < 1:
        raise ValueError(
            "minimum_margin_rate must be between 0 and 1."
        )

    if configuration.minimum_expected_quantity < 0:
        raise ValueError(
            "minimum_expected_quantity cannot be negative."
        )

    if configuration.action_threshold < 0:
        raise ValueError(
            "action_threshold cannot be negative."
        )

    if (
        configuration.competitor_price_tolerance is not None
        and configuration.competitor_price_tolerance <= 0
    ):
        raise ValueError(
            "competitor_price_tolerance must be greater than zero when set."
        )

    if not 0 <= configuration.promotion_cost_rate < 1:
        raise ValueError(
            "promotion_cost_rate must be between 0 and 1."
        )

    if not -1 <= configuration.cross_price_elasticity <= 1:
        raise ValueError(
            "cross_price_elasticity must be between -1 and 1."
        )

    if configuration.cannibalization_max_iterations < 1:
        raise ValueError(
            "cannibalization_max_iterations must be at least 1."
        )

    if configuration.price_ending is not None and not (
        0 <= configuration.price_ending < 1
    ):
        raise ValueError(
            "price_ending must be between 0 and 1 when set."
        )

    valid_inventory_modes = {
        "cap",
        "constraint",
    }

    if configuration.inventory_mode not in valid_inventory_modes:
        raise ValueError(
            "inventory_mode must be one of: "
            f"{sorted(valid_inventory_modes)}."
        )


def safe_change_rate(
    new_value: float,
    old_value: float,
) -> float:
    """Calculate a change rate while safely handling a zero baseline."""

    if not np.isfinite(new_value):
        return np.nan

    if not np.isfinite(old_value) or old_value == 0:
        return np.nan

    return float(
        (
            float(new_value)
            - float(old_value)
        )
        / abs(float(old_value))
    )


def classify_pricing_action(
    price_change_rate: float,
    *,
    threshold: float = 0.005,
) -> str:
    """Classify a recommendation as increase, decrease, or hold."""

    if not np.isfinite(price_change_rate):
        return "review"

    if price_change_rate > threshold:
        return "increase_price"

    if price_change_rate < -threshold:
        return "decrease_price"

    return "hold_price"


def calculate_minimum_price_for_margin(
    unit_cost: float,
    minimum_margin_rate: float,
) -> float:
    """Calculate the minimum price required for a target margin."""

    if unit_cost < 0:
        raise ValueError(
            "unit_cost cannot be negative."
        )

    if not 0 <= minimum_margin_rate < 1:
        raise ValueError(
            "minimum_margin_rate must be between 0 and 1."
        )

    if minimum_margin_rate == 0:
        return float(unit_cost)

    return float(
        unit_cost
        / (
            1.0
            - minimum_margin_rate
        )
    )


def calculate_confidence_dampening_factor(
    elasticity: float,
    elasticity_lower: float,
    elasticity_upper: float,
) -> float:
    """
    Calculate how much to shrink a price move given elasticity uncertainty.

    Returns a value in (0, 1]: 1.0 means the confidence interval is a
    point estimate (fully trusted), and the factor shrinks toward 0 as
    the interval widens relative to the elasticity magnitude.
    """

    values = [elasticity, elasticity_lower, elasticity_upper]

    if not all(np.isfinite(value) for value in values):
        raise ValueError(
            "elasticity, elasticity_lower, and elasticity_upper must be "
            "finite numbers."
        )

    if elasticity_lower > elasticity_upper:
        raise ValueError(
            "elasticity_lower cannot be greater than elasticity_upper."
        )

    interval_width = abs(elasticity_upper - elasticity_lower)

    relative_uncertainty = interval_width / max(abs(elasticity), 1e-6)

    return float(1.0 / (1.0 + relative_uncertainty))


def round_to_price_ending(
    price: float,
    ending: float = 0.99,
) -> float:
    """Round a price to the nearest value with the given fractional ending."""

    if not np.isfinite(price) or price <= 0:
        raise ValueError(
            "price must be a finite number greater than zero."
        )

    if not 0 <= ending < 1:
        raise ValueError(
            "ending must be between 0 and 1."
        )

    lower_candidate = np.floor(price - ending) + ending
    upper_candidate = lower_candidate + 1.0

    if lower_candidate <= 0:
        return float(upper_candidate)

    if (price - lower_candidate) <= (upper_candidate - price):
        return float(lower_candidate)

    return float(upper_candidate)


def recompute_price_financials(
    *,
    candidate_price: float,
    base_price: float,
    baseline_quantity: float,
    elasticity: float,
    unit_cost: float = 0.0,
    promotion_active: bool = False,
    promotion_uplift_rate: float = 0.0,
    promotion_cost_rate: float = 0.0,
    cannibalization_adjustment: float = 0.0,
    available_inventory: float | None = None,
    inventory_mode: str = "cap",
) -> dict[str, float]:
    """Recompute demand, revenue, and profit for a single candidate price."""

    expected_quantity = float(
        predict_quantity_from_elasticity(
            base_quantity=float(baseline_quantity),
            base_price=float(base_price),
            new_price=float(candidate_price),
            elasticity=float(elasticity),
        )
    )

    expected_quantity = max(expected_quantity, 0.0)

    if promotion_active:
        expected_quantity *= 1.0 + float(promotion_uplift_rate)

    expected_quantity *= 1.0 + float(cannibalization_adjustment)

    expected_quantity = max(expected_quantity, 0.0)

    unconstrained_quantity = expected_quantity

    inventory_binding = False

    if (
        inventory_mode == "cap"
        and available_inventory is not None
        and np.isfinite(available_inventory)
    ):
        if expected_quantity > float(available_inventory):
            inventory_binding = True

        expected_quantity = min(
            expected_quantity,
            float(available_inventory),
        )

    expected_revenue = float(candidate_price) * expected_quantity

    unit_margin = float(candidate_price) - float(unit_cost)

    expected_profit = unit_margin * expected_quantity

    if promotion_active and promotion_cost_rate:
        expected_profit -= float(promotion_cost_rate) * expected_revenue

    margin_rate = (
        unit_margin / float(candidate_price)
        if candidate_price > 0
        else np.nan
    )

    return {
        "candidate_price": float(candidate_price),
        "expected_quantity": expected_quantity,
        "unconstrained_quantity": unconstrained_quantity,
        "expected_revenue": expected_revenue,
        "expected_profit": expected_profit,
        "margin_rate": margin_rate,
        "inventory_binding": inventory_binding,
    }

# %%
def generate_price_change_rates(
    *,
    minimum_change_rate: float = -0.20,
    maximum_change_rate: float = 0.20,
    step: float = 0.01,
) -> np.ndarray:
    """Generate an inclusive grid of candidate price-change rates."""

    if minimum_change_rate >= maximum_change_rate:
        raise ValueError(
            "minimum_change_rate must be smaller than "
            "maximum_change_rate."
        )

    if minimum_change_rate <= -1:
        raise ValueError(
            "minimum_change_rate must be greater than -1."
        )

    if step <= 0:
        raise ValueError(
            "step must be greater than zero."
        )

    number_of_steps = int(
        np.floor(
            (
                maximum_change_rate
                - minimum_change_rate
            )
            / step
        )
    )

    change_rates = (
        minimum_change_rate
        + np.arange(
            number_of_steps + 1,
            dtype=float,
        )
        * step
    )

    if (
        change_rates[-1]
        < maximum_change_rate - 1e-10
    ):
        change_rates = np.append(
            change_rates,
            maximum_change_rate,
        )

    # Always offer "hold" as a candidate -- but only when holding is
    # actually permitted. Appending 0.0 unconditionally used to smuggle a
    # no-change scenario into bands that explicitly exclude it (e.g. a
    # mandated increase of at least 5%), so the optimizer could return
    # "hold_price" while reporting that it had honoured the floor.
    if minimum_change_rate <= 0.0 <= maximum_change_rate:
        change_rates = np.append(
            change_rates,
            0.0,
        )

    change_rates = np.unique(
        np.round(
            change_rates,
            10,
        )
    )

    return np.sort(
        change_rates
    )

# %%
def evaluate_price_scenarios(
    *,
    current_price: float,
    baseline_quantity: float,
    elasticity: float,
    unit_cost: float = 0.0,
    price_change_rates: Sequence[float] | None = None,
    competitor_price: float | None = None,
    promotion_active: bool = False,
    promotion_uplift_rate: float = 0.0,
    cannibalization_adjustment: float = 0.0,
    available_inventory: float | None = None,
    configuration: PricingOptimizationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate demand, revenue, and profit across price scenarios."""

    config = (
        configuration
        if configuration is not None
        else PricingOptimizationConfig()
    )

    validate_pricing_configuration(
        config
    )

    numeric_values = {
        "current_price": current_price,
        "baseline_quantity": baseline_quantity,
        "elasticity": elasticity,
        "unit_cost": unit_cost,
    }

    for name, value in numeric_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be a finite number."
            )

    if current_price <= 0:
        raise ValueError(
            "current_price must be greater than zero."
        )

    if baseline_quantity < 0:
        raise ValueError(
            "baseline_quantity cannot be negative."
        )

    if unit_cost < 0:
        raise ValueError(
            "unit_cost cannot be negative."
        )

    if competitor_price is not None and (
        not np.isfinite(competitor_price) or competitor_price <= 0
    ):
        raise ValueError(
            "competitor_price must be a finite number greater than zero."
        )

    if not np.isfinite(promotion_uplift_rate) or promotion_uplift_rate <= -1:
        raise ValueError(
            "promotion_uplift_rate must be a finite number greater than -1."
        )

    if (
        not np.isfinite(cannibalization_adjustment)
        or cannibalization_adjustment <= -1
    ):
        raise ValueError(
            "cannibalization_adjustment must be a finite number greater "
            "than -1."
        )

    if available_inventory is not None and (
        not np.isfinite(available_inventory) or available_inventory < 0
    ):
        raise ValueError(
            "available_inventory must be a finite number that is not "
            "negative."
        )

    if price_change_rates is None:
        candidate_change_rates = (
            generate_price_change_rates(
                minimum_change_rate=(
                    config.minimum_price_change_rate
                ),
                maximum_change_rate=(
                    config.maximum_price_change_rate
                ),
                step=config.price_change_step,
            )
        )
    else:
        candidate_change_rates = np.asarray(
            list(price_change_rates),
            dtype=float,
        )

    if len(candidate_change_rates) == 0:
        raise ValueError(
            "At least one price scenario is required."
        )

    if not np.isfinite(
        candidate_change_rates
    ).all():
        raise ValueError(
            "Price-change rates must contain only finite values."
        )

    if (
        candidate_change_rates <= -1
    ).any():
        raise ValueError(
            "Every price-change rate must be greater than -1."
        )

    scenarios = pd.DataFrame(
        {
            "price_change_rate": (
                candidate_change_rates
            )
        }
    )

    scenarios["candidate_price"] = (
        float(current_price)
        * (
            1.0
            + scenarios["price_change_rate"]
        )
    )

    scenarios["expected_quantity"] = (
        predict_quantity_from_elasticity(
            base_quantity=float(
                baseline_quantity
            ),
            base_price=float(
                current_price
            ),
            new_price=scenarios[
                "candidate_price"
            ].to_numpy(),
            elasticity=float(
                elasticity
            ),
        )
    )

    scenarios["expected_quantity"] = (
        scenarios["expected_quantity"]
        .clip(lower=0)
    )

    if promotion_active:
        scenarios["expected_quantity"] = (
            scenarios["expected_quantity"]
            * (1.0 + float(promotion_uplift_rate))
        )

    scenarios["expected_quantity"] = (
        scenarios["expected_quantity"]
        * (1.0 + float(cannibalization_adjustment))
    ).clip(lower=0)

    scenarios["unconstrained_quantity"] = (
        scenarios["expected_quantity"]
    )

    if (
        config.inventory_mode == "cap"
        and available_inventory is not None
    ):
        scenarios["expected_quantity"] = (
            scenarios["expected_quantity"]
            .clip(upper=float(available_inventory))
        )

    scenarios["inventory_binding"] = (
        scenarios["unconstrained_quantity"]
        > scenarios["expected_quantity"] + 1e-9
    )

    scenarios["expected_revenue"] = (
        scenarios["candidate_price"]
        * scenarios["expected_quantity"]
    )

    scenarios["unit_margin"] = (
        scenarios["candidate_price"]
        - float(unit_cost)
    )

    scenarios["margin_rate"] = np.where(
        scenarios["candidate_price"] > 0,
        scenarios["unit_margin"]
        / scenarios["candidate_price"],
        np.nan,
    )

    scenarios["expected_profit"] = (
        scenarios["unit_margin"]
        * scenarios["expected_quantity"]
    )

    if promotion_active and config.promotion_cost_rate:
        scenarios["expected_profit"] = (
            scenarios["expected_profit"]
            - config.promotion_cost_rate
            * scenarios["expected_revenue"]
        )

    # Evaluate the reference state with the same demand, promotion-cost and
    # inventory semantics as every candidate.  The old reference multiplied
    # price/margin by uncapped baseline demand while candidates were capped at
    # available inventory, manufacturing negative "uplift" even when the
    # optimizer improved the feasible plan.
    current_financials = recompute_price_financials(
        candidate_price=float(current_price),
        base_price=float(current_price),
        baseline_quantity=float(baseline_quantity),
        elasticity=float(elasticity),
        unit_cost=float(unit_cost),
        promotion_active=promotion_active,
        promotion_uplift_rate=promotion_uplift_rate,
        promotion_cost_rate=config.promotion_cost_rate,
        cannibalization_adjustment=0.0,
        available_inventory=available_inventory,
        inventory_mode=config.inventory_mode,
    )
    current_quantity = float(current_financials["expected_quantity"])
    current_revenue = float(current_financials["expected_revenue"])
    current_profit = float(current_financials["expected_profit"])
    scenarios["quantity_change"] = (
        scenarios["expected_quantity"]
        - current_quantity
    )

    scenarios["quantity_change_rate"] = (
        scenarios["quantity_change"]
        / max(
            abs(current_quantity),
            1e-12,
        )
    )

    scenarios["revenue_change"] = (
        scenarios["expected_revenue"]
        - current_revenue
    )

    scenarios["revenue_change_rate"] = np.where(
        current_revenue != 0,
        scenarios["revenue_change"]
        / abs(current_revenue),
        np.nan,
    )

    scenarios["profit_change"] = (
        scenarios["expected_profit"]
        - current_profit
    )

    scenarios["profit_change_rate"] = np.where(
        current_profit != 0,
        scenarios["profit_change"]
        / abs(current_profit),
        np.nan,
    )

    minimum_allowed_price = (
        calculate_minimum_price_for_margin(
            unit_cost=float(unit_cost),
            minimum_margin_rate=(
                config.minimum_margin_rate
            ),
        )
    )

    scenarios[
        "meets_margin_constraint"
    ] = (
        scenarios["candidate_price"]
        >= minimum_allowed_price
    )

    scenarios[
        "meets_quantity_constraint"
    ] = (
        scenarios["expected_quantity"]
        >= config.minimum_expected_quantity
    )

    if config.enforce_nonnegative_profit:
        scenarios[
            "meets_profit_constraint"
        ] = (
            scenarios["expected_profit"]
            >= 0
        )
    else:
        scenarios[
            "meets_profit_constraint"
        ] = True

    if (
        competitor_price is not None
        and config.competitor_price_tolerance is not None
    ):
        competitor_lower_bound = float(competitor_price) * (
            1.0 - config.competitor_price_tolerance
        )

        competitor_upper_bound = float(competitor_price) * (
            1.0 + config.competitor_price_tolerance
        )

        scenarios["meets_competitor_constraint"] = (
            scenarios["candidate_price"].between(
                competitor_lower_bound,
                competitor_upper_bound,
            )
        )
    else:
        scenarios["meets_competitor_constraint"] = True

    if (
        available_inventory is not None
        and config.inventory_mode == "constraint"
    ):
        scenarios["meets_inventory_constraint"] = (
            scenarios["expected_quantity"] <= float(available_inventory)
        )
    else:
        # Under "cap" semantics demand has already been truncated at
        # available stock, so every scenario satisfies the stock limit by
        # construction; selling out is an outcome, not an infeasibility.
        scenarios["meets_inventory_constraint"] = True

    scenarios["is_feasible"] = (
        scenarios[
            "meets_margin_constraint"
        ]
        & scenarios[
            "meets_quantity_constraint"
        ]
        & scenarios[
            "meets_profit_constraint"
        ]
        & scenarios["meets_competitor_constraint"]
        & scenarios["meets_inventory_constraint"]
    )

    objective_column_map = {
        "profit": "expected_profit",
        "revenue": "expected_revenue",
        "quantity": "expected_quantity",
    }

    objective_column = (
        objective_column_map[
            config.objective
        ]
    )

    scenarios["objective_name"] = (
        config.objective
    )

    scenarios["objective_value"] = (
        scenarios[objective_column]
    )

    scenarios["current_price"] = float(
        current_price
    )

    scenarios["baseline_quantity"] = float(
        baseline_quantity
    )

    scenarios["elasticity"] = float(
        elasticity
    )

    scenarios["unit_cost"] = float(
        unit_cost
    )

    return (
        scenarios.sort_values(
            "candidate_price"
        )
        .reset_index(drop=True)
    )
    
# %%
def select_optimal_price_scenario(
    scenarios: pd.DataFrame,
) -> pd.Series:
    """Select the feasible scenario with the highest objective value."""

    validate_required_columns(
        scenarios,
        [
            "candidate_price",
            "objective_value",
            "is_feasible",
        ],
        dataframe_name="price scenarios",
    )

    feasible_scenarios = scenarios.loc[
        scenarios["is_feasible"]
    ].copy()

    if feasible_scenarios.empty:
        raise ValueError(
            "No feasible price scenario satisfies the constraints."
        )

    valid_objective_mask = np.isfinite(
        feasible_scenarios[
            "objective_value"
        ]
    )

    feasible_scenarios = feasible_scenarios.loc[
        valid_objective_mask
    ]

    if feasible_scenarios.empty:
        raise ValueError(
            "No feasible scenario contains a valid objective value."
        )

    best_index = feasible_scenarios[
        "objective_value"
    ].idxmax()

    return scenarios.loc[
        best_index
    ].copy()
    
# %%
def optimize_item_price(
    *,
    current_price: float,
    baseline_quantity: float,
    elasticity: float,
    unit_cost: float = 0.0,
    item_id: object | None = None,
    category: object | None = None,
    competitor_price: float | None = None,
    promotion_active: bool = False,
    promotion_uplift_rate: float = 0.0,
    cannibalization_adjustment: float = 0.0,
    available_inventory: float | None = None,
    elasticity_lower: float | None = None,
    elasticity_upper: float | None = None,
    configuration: PricingOptimizationConfig | None = None,
) -> PriceOptimizationResult:
    """Optimize the price of one item using constant-elasticity demand."""

    config = (
        configuration
        if configuration is not None
        else PricingOptimizationConfig()
    )

    scenarios = evaluate_price_scenarios(
        current_price=current_price,
        baseline_quantity=baseline_quantity,
        elasticity=elasticity,
        unit_cost=unit_cost,
        competitor_price=competitor_price,
        promotion_active=promotion_active,
        promotion_uplift_rate=promotion_uplift_rate,
        cannibalization_adjustment=cannibalization_adjustment,
        available_inventory=available_inventory,
        configuration=config,
    )

    optimal_scenario = (
        select_optimal_price_scenario(
            scenarios
        )
    )

    current_financials = recompute_price_financials(
        candidate_price=float(current_price),
        base_price=float(current_price),
        baseline_quantity=float(baseline_quantity),
        elasticity=float(elasticity),
        unit_cost=float(unit_cost),
        promotion_active=promotion_active,
        promotion_uplift_rate=promotion_uplift_rate,
        promotion_cost_rate=config.promotion_cost_rate,
        cannibalization_adjustment=0.0,
        available_inventory=available_inventory,
        inventory_mode=config.inventory_mode,
    )
    current_quantity = float(current_financials["expected_quantity"])
    current_revenue = float(current_financials["expected_revenue"])
    current_profit = float(current_financials["expected_profit"])

    enforce_objective_no_harm = (
        config.minimum_price_change_rate <= 0 <= config.maximum_price_change_rate
        and not (
            config.inventory_mode == "constraint"
            and available_inventory is not None
            and current_quantity > float(available_inventory) + 1e-9
        )
    )

    chosen_price_change_rate = float(
        optimal_scenario["price_change_rate"]
    )

    if config.enable_confidence_dampening:
        if elasticity_lower is None or elasticity_upper is None:
            raise ValueError(
                "elasticity_lower and elasticity_upper are required when "
                "enable_confidence_dampening is True."
            )

        confidence_dampening_factor = (
            calculate_confidence_dampening_factor(
                elasticity=elasticity,
                elasticity_lower=elasticity_lower,
                elasticity_upper=elasticity_upper,
            )
        )
    else:
        confidence_dampening_factor = 1.0

    damped_price_change_rate = (
        chosen_price_change_rate
        * confidence_dampening_factor
    )

    if damped_price_change_rate == chosen_price_change_rate:
        recommended_financials = {
            "candidate_price": float(
                optimal_scenario["candidate_price"]
            ),
            "expected_quantity": float(
                optimal_scenario["expected_quantity"]
            ),
            "expected_revenue": float(
                optimal_scenario["expected_revenue"]
            ),
            "expected_profit": float(
                optimal_scenario["expected_profit"]
            ),
            "margin_rate": float(
                optimal_scenario["margin_rate"]
            ),
            "unconstrained_quantity": float(
                optimal_scenario["unconstrained_quantity"]
            ),
            "inventory_binding": bool(
                optimal_scenario["inventory_binding"]
            ),
        }
    else:
        recommended_financials = recompute_price_financials(
            candidate_price=(
                float(current_price)
                * (1.0 + damped_price_change_rate)
            ),
            base_price=current_price,
            baseline_quantity=baseline_quantity,
            elasticity=elasticity,
            unit_cost=unit_cost,
            promotion_active=promotion_active,
            promotion_uplift_rate=promotion_uplift_rate,
            promotion_cost_rate=config.promotion_cost_rate,
            cannibalization_adjustment=cannibalization_adjustment,
            available_inventory=available_inventory,
            inventory_mode=config.inventory_mode,
        )

    price_ending_applied = False

    if config.price_ending is not None:
        rounded_price = round_to_price_ending(
            recommended_financials["candidate_price"],
            config.price_ending,
        )

        if rounded_price != recommended_financials["candidate_price"]:
            price_ending_applied = True

            recommended_financials = recompute_price_financials(
                candidate_price=rounded_price,
                base_price=current_price,
                baseline_quantity=baseline_quantity,
                elasticity=elasticity,
                unit_cost=unit_cost,
                promotion_active=promotion_active,
                promotion_uplift_rate=promotion_uplift_rate,
                promotion_cost_rate=config.promotion_cost_rate,
                cannibalization_adjustment=cannibalization_adjustment,
                available_inventory=available_inventory,
                inventory_mode=config.inventory_mode,
            )

    minimum_allowed_price = calculate_minimum_price_for_margin(
        unit_cost=float(unit_cost),
        minimum_margin_rate=config.minimum_margin_rate,
    )

    if recommended_financials["candidate_price"] < minimum_allowed_price:
        recommended_financials = recompute_price_financials(
            candidate_price=minimum_allowed_price,
            base_price=current_price,
            baseline_quantity=baseline_quantity,
            elasticity=elasticity,
            unit_cost=unit_cost,
            promotion_active=promotion_active,
            promotion_uplift_rate=promotion_uplift_rate,
            promotion_cost_rate=config.promotion_cost_rate,
            cannibalization_adjustment=cannibalization_adjustment,
            available_inventory=available_inventory,
            inventory_mode=config.inventory_mode,
        )

    # Dampening, rounding, and the margin-floor clamp above can each move
    # the price off the original grid-search pick, which was the only
    # point checked against the constraint set. Re-validate the final
    # price against EVERY guardrail here and, if it now violates any of
    # them, fall back to the nearest still-feasible grid scenario (every
    # "is_feasible" row satisfies margin, quantity, profit, competitor,
    # and inventory constraints together).
    #
    # The previous version of this block re-checked only the competitor
    # band and the inventory cap, so `.99` rounding or confidence
    # dampening could push the recommendation below
    # `minimum_expected_quantity`, or (with a nonzero promotion cost
    # rate) to a negative expected profit, and the recommendation would
    # still be reported as constraint-satisfying.
    def _check_guardrails(
        financials: dict[str, float],
    ) -> dict[str, bool]:
        """Test one set of financials against the full constraint set."""

        price = float(financials["candidate_price"])

        quantity = float(financials["expected_quantity"])

        if (
            competitor_price is not None
            and config.competitor_price_tolerance is not None
        ):
            competitor_ok = (
                float(competitor_price)
                * (1.0 - config.competitor_price_tolerance)
                <= price
                <= float(competitor_price)
                * (1.0 + config.competitor_price_tolerance)
            )
        else:
            competitor_ok = True

        if (
            available_inventory is not None
            and config.inventory_mode == "constraint"
        ):
            inventory_ok = quantity <= float(available_inventory) + 1e-9
        else:
            inventory_ok = True

        if config.enforce_nonnegative_profit:
            profit_ok = (
                float(financials["expected_profit"]) >= -1e-9
            )
        else:
            profit_ok = True

        objective_value = {
            "profit": float(financials["expected_profit"]),
            "revenue": float(financials["expected_revenue"]),
            "quantity": quantity,
        }[config.objective]
        current_objective_value = {
            "profit": current_profit,
            "revenue": current_revenue,
            "quantity": current_quantity,
        }[config.objective]
        objective_ok = (
            not enforce_objective_no_harm
            or objective_value >= current_objective_value - 1e-9
        )

        return {
            "margin": price >= minimum_allowed_price - 1e-9,
            "quantity": (
                quantity
                >= config.minimum_expected_quantity - 1e-9
            ),
            "profit": profit_ok,
            "objective_no_harm": objective_ok,
            "competitor": competitor_ok,
            "inventory": inventory_ok,
        }

    guardrails = _check_guardrails(recommended_financials)

    competitor_satisfied = guardrails["competitor"]

    inventory_satisfied = guardrails["inventory"]

    constraint_fallback_applied = False

    if not all(guardrails.values()):
        feasible_scenarios = scenarios.loc[
            scenarios["is_feasible"]
        ]

        current_objective_value = {
            "profit": current_profit,
            "revenue": current_revenue,
            "quantity": current_quantity,
        }[config.objective]
        if enforce_objective_no_harm:
            feasible_scenarios = feasible_scenarios.loc[
                feasible_scenarios["objective_value"]
                >= current_objective_value - 1e-9
            ]

        if feasible_scenarios.empty:
            breached = sorted(
                name
                for name, satisfied in guardrails.items()
                if not satisfied
            )

            raise ValueError(
                "No feasible price scenario satisfies the "
                f"{', '.join(breached)} constraint(s) after dampening, "
                "rounding, and the margin-floor clamp."
            )

        nearest_index = (
            feasible_scenarios["candidate_price"]
            - recommended_financials["candidate_price"]
        ).abs().idxmin()

        nearest_scenario = scenarios.loc[nearest_index]

        recommended_financials = {
            "candidate_price": float(
                nearest_scenario["candidate_price"]
            ),
            "expected_quantity": float(
                nearest_scenario["expected_quantity"]
            ),
            "expected_revenue": float(
                nearest_scenario["expected_revenue"]
            ),
            "expected_profit": float(
                nearest_scenario["expected_profit"]
            ),
            "margin_rate": float(
                nearest_scenario["margin_rate"]
            ),
            "unconstrained_quantity": float(
                nearest_scenario["unconstrained_quantity"]
            ),
            "inventory_binding": bool(
                nearest_scenario["inventory_binding"]
            ),
        }

        price_ending_applied = False
        constraint_fallback_applied = True

        guardrails = _check_guardrails(recommended_financials)

        competitor_satisfied = guardrails["competitor"]
        inventory_satisfied = guardrails["inventory"]

    recommended_price = float(
        recommended_financials["candidate_price"]
    )

    recommended_quantity = float(
        recommended_financials["expected_quantity"]
    )

    recommended_revenue = float(
        recommended_financials["expected_revenue"]
    )

    recommended_profit = float(
        recommended_financials["expected_profit"]
    )

    price_change = (
        recommended_price
        - float(current_price)
    )

    price_change_rate = (
        price_change
        / float(current_price)
    )

    recommendation: dict[
        str,
        Any,
    ] = {
        "item_id": item_id,
        "category": category,
        "status": "success",
        "optimization_objective": (
            config.objective
        ),
        "current_price": float(
            current_price
        ),
        "recommended_price": (
            recommended_price
        ),
        "price_change": float(
            price_change
        ),
        "price_change_rate": float(
            price_change_rate
        ),
        "recommendation_action": (
            classify_pricing_action(
                price_change_rate,
                threshold=(
                    config.action_threshold
                ),
            )
        ),
        "baseline_quantity": float(
            baseline_quantity
        ),
        "current_quantity": current_quantity,
        "expected_quantity": (
            recommended_quantity
        ),
        "quantity_change": float(
            recommended_quantity
            - current_quantity
        ),
        "quantity_change_rate": (
            safe_change_rate(
                recommended_quantity,
                current_quantity,
            )
        ),
        "current_revenue": float(
            current_revenue
        ),
        "expected_revenue": (
            recommended_revenue
        ),
        "revenue_change": float(
            recommended_revenue
            - current_revenue
        ),
        "revenue_change_rate": (
            safe_change_rate(
                recommended_revenue,
                current_revenue,
            )
        ),
        "unit_cost": float(
            unit_cost
        ),
        "current_unit_margin": float(
            current_price
            - unit_cost
        ),
        "recommended_unit_margin": float(
            recommended_price
            - unit_cost
        ),
        "recommended_margin_rate": float(
            recommended_financials["margin_rate"]
        ),
        "current_profit": float(
            current_profit
        ),
        "expected_profit": (
            recommended_profit
        ),
        "profit_change": float(
            recommended_profit
            - current_profit
        ),
        "profit_change_rate": (
            safe_change_rate(
                recommended_profit,
                current_profit,
            )
        ),
        "elasticity": float(
            elasticity
        ),
        "objective_value": float(
            optimal_scenario[
                "objective_value"
            ]
        ),
        "feasible_scenario_count": int(
            scenarios[
                "is_feasible"
            ].sum()
        ),
        "evaluated_scenario_count": int(
            len(scenarios)
        ),
        "confidence_dampening_factor": float(
            confidence_dampening_factor
        ),
        "price_ending_applied": bool(
            price_ending_applied
        ),
        "competitor_price": (
            float(competitor_price)
            if competitor_price is not None
            else np.nan
        ),
        "meets_competitor_constraint": bool(
            competitor_satisfied
        ),
        "promotion_active": bool(
            promotion_active
        ),
        "promotion_uplift_rate": float(
            promotion_uplift_rate
        ),
        "cannibalization_adjustment": float(
            cannibalization_adjustment
        ),
        "available_inventory": (
            float(available_inventory)
            if available_inventory is not None
            else np.nan
        ),
        "meets_inventory_constraint": bool(
            inventory_satisfied
        ),
        "meets_margin_constraint": bool(
            guardrails["margin"]
        ),
        "meets_quantity_constraint": bool(
            guardrails["quantity"]
        ),
        "meets_profit_constraint": bool(
            guardrails["profit"]
        ),
        "meets_objective_no_harm_constraint": bool(
            guardrails["objective_no_harm"]
        ),
        "meets_all_constraints": bool(
            all(guardrails.values())
        ),
        "minimum_allowed_price": float(
            minimum_allowed_price
        ),
        "inventory_mode": config.inventory_mode,
        "inventory_binding": bool(
            recommended_financials.get(
                "inventory_binding",
                False,
            )
        ),
        "unconstrained_quantity": float(
            recommended_financials.get(
                "unconstrained_quantity",
                recommended_quantity,
            )
        ),
        "constraint_fallback_applied": bool(
            constraint_fallback_applied
        ),
    }

    return PriceOptimizationResult(
        recommendation=recommendation,
        scenarios=scenarios,
        configuration=config,
    )
    
# %%
def _coerce_optional_numeric(
    row: pd.Series,
    column: str | None,
    *,
    default: float | None,
) -> float | None:
    """Read an optional numeric column, falling back to a default."""

    if column is None:
        return default

    value = pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return default

    return float(value)


# A round of the cannibalization fixed-point loop is treated as converged
# once every item's price-change rate has moved by less than this amount
# from the previous round -- tight enough that no candidate price shifts
# by a meaningfully different grid step, loose enough to stop promptly
# once the portfolio has settled.
_CANNIBALIZATION_RATE_TOLERANCE = 1e-6


def _extract_price_change_rate(
    pass_rows: list[dict[str, Any]],
    index: pd.Index,
) -> pd.Series:
    """Pull each item's price-change rate out of one portfolio pass.

    Returns NaN for any item whose optimization failed, so callers
    computing a sibling average exclude it exactly as a successful-only
    pass would.
    """

    pass_frame = pd.DataFrame(
        pass_rows,
        index=index,
    )

    if "price_change_rate" not in pass_frame.columns:
        return pd.Series(
            np.nan,
            index=index,
        )

    return pass_frame["price_change_rate"].where(
        pass_frame["status"] == "success"
    )


def _compute_sibling_avg_rate(
    successful_rate: pd.Series,
    category_series: pd.Series,
) -> pd.Series:
    """Average each item's category siblings' price-change rate."""

    category_sum = successful_rate.groupby(
        category_series
    ).transform("sum")

    category_count = successful_rate.notna().groupby(
        category_series
    ).transform("sum")

    # Exclude the item itself from its own sibling average -- but only
    # subtract it from the denominator when it actually contributed to
    # the numerator. Unconditionally subtracting 1 inflated the average
    # by a factor of n/(n-1) for every item whose own rate was NaN (i.e.
    # every item that failed), because such an item never entered
    # `category_count` to begin with.
    own_contributes = successful_rate.notna().astype(float)

    sibling_count = (category_count - own_contributes).replace(
        0,
        np.nan,
    )

    return (
        (category_sum - successful_rate.fillna(0.0))
        / sibling_count
    ).fillna(0.0)


def _run_portfolio_pass(
    dataframe: pd.DataFrame,
    *,
    item_column: str,
    current_price_column: str,
    baseline_quantity_column: str,
    elasticity_column: str,
    unit_cost_column: str | None,
    category_column: str | None,
    competitor_price_column: str | None,
    promotion_active_column: str | None,
    promotion_uplift_rate_column: str | None,
    available_inventory_column: str | None,
    elasticity_lower_column: str | None,
    elasticity_upper_column: str | None,
    cannibalization_adjustments: pd.Series | None,
    config: PricingOptimizationConfig,
    category_change_rate_overrides: (
        dict[str, tuple[float, float]] | None
    ),
) -> list[dict[str, Any]]:
    """Run one optimization pass over every row of a portfolio dataset."""

    recommendation_rows: list[dict[str, Any]] = []

    for row_index, row in dataframe.iterrows():
        item_id = row[item_column]

        category = (
            row[category_column]
            if category_column is not None
            else None
        )

        current_price = _coerce_optional_numeric(
            row, current_price_column, default=np.nan
        )

        baseline_quantity = _coerce_optional_numeric(
            row, baseline_quantity_column, default=np.nan
        )

        elasticity = _coerce_optional_numeric(
            row, elasticity_column, default=np.nan
        )

        unit_cost = _coerce_optional_numeric(
            row, unit_cost_column, default=0.0
        )

        competitor_price = _coerce_optional_numeric(
            row, competitor_price_column, default=None
        )

        promotion_uplift_rate = _coerce_optional_numeric(
            row, promotion_uplift_rate_column, default=0.0
        )

        available_inventory = _coerce_optional_numeric(
            row, available_inventory_column, default=None
        )

        elasticity_lower = _coerce_optional_numeric(
            row, elasticity_lower_column, default=None
        )

        elasticity_upper = _coerce_optional_numeric(
            row, elasticity_upper_column, default=None
        )

        if (
            promotion_active_column is not None
            and pd.notna(row[promotion_active_column])
        ):
            promotion_active = bool(
                row[promotion_active_column]
            )
        else:
            promotion_active = False

        cannibalization_adjustment = (
            float(cannibalization_adjustments.loc[row_index])
            if cannibalization_adjustments is not None
            else 0.0
        )

        row_config = config

        if (
            category_change_rate_overrides
            and category in category_change_rate_overrides
        ):
            override_minimum, override_maximum = (
                category_change_rate_overrides[category]
            )

            row_config = replace(
                config,
                minimum_price_change_rate=override_minimum,
                maximum_price_change_rate=override_maximum,
            )

            validate_pricing_configuration(row_config)

        try:
            result = optimize_item_price(
                item_id=item_id,
                category=category,
                current_price=float(current_price),
                baseline_quantity=float(baseline_quantity),
                elasticity=float(elasticity),
                unit_cost=float(unit_cost),
                competitor_price=competitor_price,
                promotion_active=promotion_active,
                promotion_uplift_rate=promotion_uplift_rate,
                cannibalization_adjustment=(
                    cannibalization_adjustment
                ),
                available_inventory=available_inventory,
                elasticity_lower=elasticity_lower,
                elasticity_upper=elasticity_upper,
                configuration=row_config,
            )

            recommendation_rows.append(
                result.recommendation
            )

        # Only genuine, per-item modelling failures are absorbed into a
        # "failed" row: a ValueError means this item's inputs or
        # constraints admit no solution, which is a legitimate business
        # outcome to report. Everything else (KeyError from a mistyped
        # column, TypeError from a bad dtype, AttributeError from an API
        # change) is a defect in the caller or the library and must
        # propagate -- the previous blanket `except Exception` turned such
        # defects into a portfolio of silent "review" rows.
        except ValueError as error:
            recommendation_rows.append(
                {
                    "item_id": item_id,
                    "category": category,
                    "status": "failed",
                    "error_message": str(error),
                    "optimization_objective": (
                        config.objective
                    ),
                    "current_price": current_price,
                    "recommended_price": np.nan,
                    "baseline_quantity": baseline_quantity,
                    "expected_quantity": np.nan,
                    "elasticity": elasticity,
                    "unit_cost": unit_cost,
                    "recommendation_action": "review",
                }
            )

    return recommendation_rows


def optimize_price_portfolio(
    dataframe: pd.DataFrame,
    *,
    item_column: str,
    current_price_column: str,
    baseline_quantity_column: str,
    elasticity_column: str,
    unit_cost_column: str | None = None,
    category_column: str | None = None,
    competitor_price_column: str | None = None,
    promotion_active_column: str | None = None,
    promotion_uplift_rate_column: str | None = None,
    available_inventory_column: str | None = None,
    elasticity_lower_column: str | None = None,
    elasticity_upper_column: str | None = None,
    category_change_rate_overrides: (
        dict[str, tuple[float, float]] | None
    ) = None,
    configuration: PricingOptimizationConfig | None = None,
) -> pd.DataFrame:
    """Optimize prices for every item in a portfolio dataset.

    When ``configuration.cross_price_elasticity`` is nonzero and
    ``category_column`` is supplied, optimization runs an initial
    independent pass followed by up to
    ``configuration.cannibalization_max_iterations`` further passes, each
    nudging every item's expected demand based on how much its category
    siblings' prices moved in the *previous* pass and stopping once
    those moves stop changing (a fixed-point heuristic, not a jointly
    solved demand system).
    """

    optional_columns = {
        "unit_cost_column": unit_cost_column,
        "category_column": category_column,
        "competitor_price_column": competitor_price_column,
        "promotion_active_column": promotion_active_column,
        "promotion_uplift_rate_column": (
            promotion_uplift_rate_column
        ),
        "available_inventory_column": (
            available_inventory_column
        ),
        "elasticity_lower_column": elasticity_lower_column,
        "elasticity_upper_column": elasticity_upper_column,
    }

    required_columns = [
        item_column,
        current_price_column,
        baseline_quantity_column,
        elasticity_column,
        *[
            column
            for column in optional_columns.values()
            if column is not None
        ],
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="price optimization data",
    )

    config = (
        configuration
        if configuration is not None
        else PricingOptimizationConfig()
    )

    validate_pricing_configuration(
        config
    )

    pass_kwargs = dict(
        item_column=item_column,
        current_price_column=current_price_column,
        baseline_quantity_column=baseline_quantity_column,
        elasticity_column=elasticity_column,
        unit_cost_column=unit_cost_column,
        category_column=category_column,
        competitor_price_column=competitor_price_column,
        promotion_active_column=promotion_active_column,
        promotion_uplift_rate_column=(
            promotion_uplift_rate_column
        ),
        available_inventory_column=(
            available_inventory_column
        ),
        elasticity_lower_column=elasticity_lower_column,
        elasticity_upper_column=elasticity_upper_column,
        category_change_rate_overrides=(
            category_change_rate_overrides
        ),
    )

    first_pass_rows = _run_portfolio_pass(
        dataframe,
        cannibalization_adjustments=None,
        config=config,
        **pass_kwargs,
    )

    # Cannibalization is a fixed-point iteration on a heuristic, not a
    # jointly solved demand system (e.g. multinomial logit / AIDS): each
    # round's sibling-price pressure is recomputed from the *previous*
    # round's own results and re-optimized, rather than frozen at pass 1,
    # so the adjustment can propagate when several siblings move
    # together. It stops as soon as no item's price-change rate moves
    # meaningfully between rounds, bounded by
    # `cannibalization_max_iterations` (a hard cap because nothing here
    # proves the loop must converge). `cannibalization_max_iterations=1`
    # reproduces the original one-shot (pass-1-frozen) behaviour exactly.
    run_cannibalization_pass = (
        config.cross_price_elasticity != 0
        and category_column is not None
    )

    if run_cannibalization_pass:
        category_series = dataframe[category_column]

        previous_rate = _extract_price_change_rate(
            first_pass_rows,
            dataframe.index,
        )

        recommendation_rows = first_pass_rows

        for _ in range(config.cannibalization_max_iterations):
            sibling_avg_rate = _compute_sibling_avg_rate(
                previous_rate,
                category_series,
            )

            cannibalization_adjustments = (
                config.cross_price_elasticity * sibling_avg_rate
            )

            recommendation_rows = _run_portfolio_pass(
                dataframe,
                cannibalization_adjustments=(
                    cannibalization_adjustments
                ),
                config=config,
                **pass_kwargs,
            )

            latest_rate = _extract_price_change_rate(
                recommendation_rows,
                dataframe.index,
            )

            converged = (
                (latest_rate.fillna(0.0) - previous_rate.fillna(0.0))
                .abs()
                .max()
                <= _CANNIBALIZATION_RATE_TOLERANCE
            )

            previous_rate = latest_rate

            if converged:
                break
    else:
        recommendation_rows = first_pass_rows

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    if "item_id" in recommendations.columns:
        recommendations = (
            recommendations.sort_values(
                [
                    "status",
                    "item_id",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return recommendations

# %%
# %%
def summarize_optimization_portfolio(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Create a one-row executive summary of portfolio recommendations."""

    validate_required_columns(
        recommendations,
        [
            "status",
            "recommendation_action",
        ],
        dataframe_name="price recommendations",
    )

    successful = recommendations.loc[
        recommendations["status"] == "success"
    ].copy()

    total_items = int(
        len(recommendations)
    )

    successful_items = int(
        len(successful)
    )

    failed_items = int(
        total_items - successful_items
    )

    summary: dict[str, Any] = {
        "total_items": total_items,
        "successful_items": successful_items,
        "failed_items": failed_items,
        "success_rate": (
            successful_items / total_items
            if total_items > 0
            else np.nan
        ),
        "price_increase_count": int(
            (
                successful["recommendation_action"]
                == "increase_price"
            ).sum()
        ),
        "price_decrease_count": int(
            (
                successful["recommendation_action"]
                == "decrease_price"
            ).sum()
        ),
        "hold_price_count": int(
            (
                successful["recommendation_action"]
                == "hold_price"
            ).sum()
        ),
    }

    numeric_totals = [
        "current_revenue",
        "expected_revenue",
        "revenue_change",
        "current_profit",
        "expected_profit",
        "profit_change",
        "baseline_quantity",
        "expected_quantity",
    ]

    for column in numeric_totals:
        if column in successful.columns:
            summary[f"total_{column}"] = float(
                pd.to_numeric(
                    successful[column],
                    errors="coerce",
                ).sum()
            )

    if {
        "current_revenue",
        "expected_revenue",
    }.issubset(successful.columns):
        summary[
            "portfolio_revenue_change_rate"
        ] = safe_change_rate(
            summary["total_expected_revenue"],
            summary["total_current_revenue"],
        )

    if {
        "current_profit",
        "expected_profit",
    }.issubset(successful.columns):
        summary[
            "portfolio_profit_change_rate"
        ] = safe_change_rate(
            summary["total_expected_profit"],
            summary["total_current_profit"],
        )

    if "price_change_rate" in successful.columns:
        summary[
            "average_price_change_rate"
        ] = float(
            pd.to_numeric(
                successful["price_change_rate"],
                errors="coerce",
            ).mean()
        )

        summary[
            "median_price_change_rate"
        ] = float(
            pd.to_numeric(
                successful["price_change_rate"],
                errors="coerce",
            ).median()
        )

    return pd.DataFrame(
        [summary]
    )
# %%
def build_optimization_action_summary(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize optimization outcomes by recommended action."""

    validate_required_columns(
        recommendations,
        [
            "status",
            "recommendation_action",
        ],
        dataframe_name="price recommendations",
    )

    successful = recommendations.loc[
        recommendations["status"] == "success"
    ].copy()

    if successful.empty:
        return pd.DataFrame(
            columns=[
                "recommendation_action",
                "item_count",
            ]
        )

    aggregation_map: dict[
        str,
        tuple[str, str],
    ] = {
        "item_count": (
            "recommendation_action",
            "size",
        ),
    }

    optional_aggregations = {
        "average_price_change_rate": (
            "price_change_rate",
            "mean",
        ),
        "total_revenue_change": (
            "revenue_change",
            "sum",
        ),
        "average_revenue_change_rate": (
            "revenue_change_rate",
            "mean",
        ),
        "total_profit_change": (
            "profit_change",
            "sum",
        ),
        "average_profit_change_rate": (
            "profit_change_rate",
            "mean",
        ),
        "total_quantity_change": (
            "quantity_change",
            "sum",
        ),
    }

    for output_column, aggregation in (
        optional_aggregations.items()
    ):
        source_column = aggregation[0]

        if source_column in successful.columns:
            aggregation_map[
                output_column
            ] = aggregation

    action_summary = (
        successful.groupby(
            "recommendation_action",
            dropna=False,
        )
        .agg(
            **aggregation_map
        )
        .reset_index()
        .sort_values(
            "item_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return action_summary

# %%
def build_optimization_category_summary(
    recommendations: pd.DataFrame,
    *,
    category_column: str = "category",
) -> pd.DataFrame:
    """Summarize optimization results by product category."""

    validate_required_columns(
        recommendations,
        [
            "status",
            category_column,
        ],
        dataframe_name="price recommendations",
    )

    successful = recommendations.loc[
        recommendations["status"] == "success"
    ].copy()

    if successful.empty:
        return pd.DataFrame(
            columns=[
                category_column,
                "item_count",
            ]
        )

    aggregation_map: dict[
        str,
        tuple[str, str],
    ] = {
        "item_count": (
            category_column,
            "size",
        ),
    }

    optional_aggregations = {
        "average_current_price": (
            "current_price",
            "mean",
        ),
        "average_recommended_price": (
            "recommended_price",
            "mean",
        ),
        "average_price_change_rate": (
            "price_change_rate",
            "mean",
        ),
        "current_revenue": (
            "current_revenue",
            "sum",
        ),
        "expected_revenue": (
            "expected_revenue",
            "sum",
        ),
        "revenue_change": (
            "revenue_change",
            "sum",
        ),
        "current_profit": (
            "current_profit",
            "sum",
        ),
        "expected_profit": (
            "expected_profit",
            "sum",
        ),
        "profit_change": (
            "profit_change",
            "sum",
        ),
    }

    for output_column, aggregation in (
        optional_aggregations.items()
    ):
        source_column = aggregation[0]

        if source_column in successful.columns:
            aggregation_map[
                output_column
            ] = aggregation

    category_summary = (
        successful.groupby(
            category_column,
            dropna=False,
        )
        .agg(
            **aggregation_map
        )
        .reset_index()
    )

    if {
        "current_revenue",
        "expected_revenue",
        "revenue_change",
    }.issubset(
        category_summary.columns
    ):
        category_summary[
            "revenue_change_rate"
        ] = np.where(
            category_summary[
                "current_revenue"
            ] != 0,
            category_summary[
                "revenue_change"
            ]
            / category_summary[
                "current_revenue"
            ].abs(),
            np.nan,
        )

    if {
        "current_profit",
        "expected_profit",
        "profit_change",
    }.issubset(
        category_summary.columns
    ):
        category_summary[
            "profit_change_rate"
        ] = np.where(
            category_summary[
                "current_profit"
            ] != 0,
            category_summary[
                "profit_change"
            ]
            / category_summary[
                "current_profit"
            ].abs(),
            np.nan,
        )

    sort_column = (
        "profit_change"
        if "profit_change"
        in category_summary.columns
        else "item_count"
    )

    category_summary = (
        category_summary.sort_values(
            sort_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return category_summary

# %%
def validate_price_recommendations(
    recommendations: pd.DataFrame,
    *,
    configuration: PricingOptimizationConfig | None = None,
) -> pd.DataFrame:
    """
    Create row-level validation checks for price recommendations.

    When ``configuration`` is supplied this independently re-derives every
    business guardrail from the recommendation table -- the permitted
    price-change band, the minimum margin rate, the minimum expected
    quantity, non-negative expected profit, and the competitor band -- so
    that validation is a genuine audit of the output rather than an
    arithmetic self-consistency check. Passing ``None`` keeps the older,
    weaker behaviour for callers that do not know the configuration.
    """

    validate_required_columns(
        recommendations,
        [
            "item_id",
            "status",
            "current_price",
            "recommended_price",
        ],
        dataframe_name="price recommendations",
    )

    validation_data = recommendations.copy()

    current_price = pd.to_numeric(
        validation_data["current_price"],
        errors="coerce",
    )

    recommended_price = pd.to_numeric(
        validation_data["recommended_price"],
        errors="coerce",
    )

    validation_data[
        "recommended_price_is_positive"
    ] = (
        recommended_price > 0
    )

    if "unit_cost" in validation_data.columns:
        unit_cost = pd.to_numeric(
            validation_data["unit_cost"],
            errors="coerce",
        )

        validation_data[
            "recommended_price_covers_cost"
        ] = (
            recommended_price >= unit_cost
        )

    else:
        validation_data[
            "recommended_price_covers_cost"
        ] = True

    if "price_change_rate" in validation_data.columns:
        recorded_change_rate = pd.to_numeric(
            validation_data["price_change_rate"],
            errors="coerce",
        )

        calculated_change_rate = (
            recommended_price
            - current_price
        ) / current_price

        validation_data[
            "price_change_rate_is_consistent"
        ] = np.isclose(
            calculated_change_rate,
            recorded_change_rate,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=False,
        )

    else:
        validation_data[
            "price_change_rate_is_consistent"
        ] = False

    validation_columns = [
        "recommended_price_is_positive",
        "recommended_price_covers_cost",
        "price_change_rate_is_consistent",
    ]

    # ---------------------------------------------------------
    # Independent audit of the configured business guardrails
    # ---------------------------------------------------------

    if configuration is not None:
        config = configuration

        change_rate = (
            recommended_price - current_price
        ) / current_price

        validation_data[
            "price_change_within_band"
        ] = (
            change_rate
            >= config.minimum_price_change_rate - 1e-6
        ) & (
            change_rate
            <= config.maximum_price_change_rate + 1e-6
        )

        validation_columns.append(
            "price_change_within_band"
        )

        if "unit_cost" in validation_data.columns:
            minimum_price = unit_cost / (
                1.0 - config.minimum_margin_rate
            )

            validation_data[
                "margin_rate_meets_floor"
            ] = (
                recommended_price >= minimum_price - 1e-6
            )

            validation_columns.append(
                "margin_rate_meets_floor"
            )

        if "expected_quantity" in validation_data.columns:
            validation_data[
                "expected_quantity_meets_floor"
            ] = (
                pd.to_numeric(
                    validation_data["expected_quantity"],
                    errors="coerce",
                )
                >= config.minimum_expected_quantity - 1e-6
            )

            validation_columns.append(
                "expected_quantity_meets_floor"
            )

        if (
            config.enforce_nonnegative_profit
            and "expected_profit" in validation_data.columns
        ):
            validation_data[
                "expected_profit_is_nonnegative"
            ] = (
                pd.to_numeric(
                    validation_data["expected_profit"],
                    errors="coerce",
                )
                >= -1e-6
            )

            validation_columns.append(
                "expected_profit_is_nonnegative"
            )

        objective_columns = {
            "profit": ("current_profit", "expected_profit"),
            "revenue": ("current_revenue", "expected_revenue"),
            "quantity": ("current_quantity", "expected_quantity"),
        }
        current_objective_column, expected_objective_column = (
            objective_columns[config.objective]
        )

        if {
            current_objective_column,
            expected_objective_column,
        }.issubset(validation_data.columns):
            validation_data["objective_does_not_harm_baseline"] = (
                pd.to_numeric(
                    validation_data[expected_objective_column],
                    errors="coerce",
                )
                >= pd.to_numeric(
                    validation_data[current_objective_column],
                    errors="coerce",
                )
                - 1e-6
            )
            validation_columns.append("objective_does_not_harm_baseline")

        if (
            config.competitor_price_tolerance is not None
            and "competitor_price" in validation_data.columns
        ):
            competitor_price = pd.to_numeric(
                validation_data["competitor_price"],
                errors="coerce",
            )

            within_band = (
                recommended_price
                >= competitor_price
                * (1.0 - config.competitor_price_tolerance)
                - 1e-6
            ) & (
                recommended_price
                <= competitor_price
                * (1.0 + config.competitor_price_tolerance)
                + 1e-6
            )

            # A missing competitor price means the guardrail does not
            # apply to that item, not that the item failed it.
            validation_data[
                "within_competitor_band"
            ] = within_band | competitor_price.isna()

            validation_columns.append(
                "within_competitor_band"
            )

    # Items that never produced a recommendation are reported under
    # `status`, not as validation failures; mixing the two conflated
    # "we declined to price this" with "we priced it badly".
    is_scored = validation_data["status"].astype(str).eq("success")

    validation_data[
        "validation_passed"
    ] = validation_data[
        validation_columns
    ].all(axis=1) | ~is_scored

    validation_data[
        "validation_status"
    ] = np.select(
        [
            ~is_scored,
            validation_data["validation_passed"],
        ],
        [
            "not_scored",
            "passed",
        ],
        default="failed",
    )

    return validation_data[
        [
            "item_id",
            "status",
            *validation_columns,
            "validation_passed",
            "validation_status",
        ]
    ]
    
# %%
def summarize_recommendation_validation(
    validation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recommendation-validation pass rates."""

    validate_required_columns(
        validation_data,
        [
            "validation_passed",
            "validation_status",
        ],
        dataframe_name="recommendation validation",
    )

    total_recommendations = int(
        len(validation_data)
    )

    status = validation_data["validation_status"].astype(str)

    not_scored_recommendations = int(
        status.eq("not_scored").sum()
    )

    scored_recommendations = int(
        total_recommendations
        - not_scored_recommendations
    )

    passed_recommendations = int(
        status.eq("passed").sum()
    )

    failed_recommendations = int(
        status.eq("failed").sum()
    )

    # The pass rate is over items that were actually priced. Including
    # items the optimizer declined to price made the headline number look
    # like a quality score when it was really a coverage score.
    validation_pass_rate = (
        passed_recommendations
        / scored_recommendations
        if scored_recommendations > 0
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "total_recommendations": (
                    total_recommendations
                ),
                "scored_recommendations": (
                    scored_recommendations
                ),
                "not_scored_recommendations": (
                    not_scored_recommendations
                ),
                "passed_recommendations": (
                    passed_recommendations
                ),
                "failed_recommendations": (
                    failed_recommendations
                ),
                "validation_pass_rate": (
                    validation_pass_rate
                ),
            }
        ]
    )

    return summary

# %%
def run_price_sensitivity_analysis(
    dataframe: pd.DataFrame,
    *,
    item_column: str,
    current_price_column: str,
    baseline_quantity_column: str,
    elasticity_column: str,
    unit_cost_column: str | None = None,
    category_column: str | None = None,
    elasticity_multipliers: Sequence[float] = (
        0.80,
        1.00,
        1.20,
    ),
    cost_multipliers: Sequence[float] = (
        0.90,
        1.00,
        1.10,
    ),
    configuration: PricingOptimizationConfig | None = None,
) -> pd.DataFrame:
    """Re-optimize prices under alternative elasticity and cost assumptions."""

    required_columns = [
        item_column,
        current_price_column,
        baseline_quantity_column,
        elasticity_column,
    ]

    if unit_cost_column is not None:
        required_columns.append(
            unit_cost_column
        )

    if category_column is not None:
        required_columns.append(
            category_column
        )

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="sensitivity data",
    )

    config = (
        configuration
        if configuration is not None
        else PricingOptimizationConfig()
    )

    validate_pricing_configuration(
        config
    )

    sensitivity_results: list[
        pd.DataFrame
    ] = []

    for elasticity_multiplier in (
        elasticity_multipliers
    ):
        if (
            not np.isfinite(
                elasticity_multiplier
            )
            or elasticity_multiplier <= 0
        ):
            raise ValueError(
                "Elasticity multipliers must be positive."
            )

        for cost_multiplier in (
            cost_multipliers
        ):
            if (
                not np.isfinite(
                    cost_multiplier
                )
                or cost_multiplier < 0
            ):
                raise ValueError(
                    "Cost multipliers cannot be negative."
                )

            scenario_data = dataframe.copy()

            scenario_data[
                "_sensitivity_elasticity"
            ] = (
                pd.to_numeric(
                    scenario_data[
                        elasticity_column
                    ],
                    errors="coerce",
                )
                * float(
                    elasticity_multiplier
                )
            )

            if unit_cost_column is not None:
                scenario_data[
                    "_sensitivity_unit_cost"
                ] = (
                    pd.to_numeric(
                        scenario_data[
                            unit_cost_column
                        ],
                        errors="coerce",
                    )
                    * float(
                        cost_multiplier
                    )
                )

            else:
                scenario_data[
                    "_sensitivity_unit_cost"
                ] = 0.0

            scenario_recommendations = (
                optimize_price_portfolio(
                    scenario_data,
                    item_column=item_column,
                    current_price_column=(
                        current_price_column
                    ),
                    baseline_quantity_column=(
                        baseline_quantity_column
                    ),
                    elasticity_column=(
                        "_sensitivity_elasticity"
                    ),
                    unit_cost_column=(
                        "_sensitivity_unit_cost"
                    ),
                    category_column=(
                        category_column
                    ),
                    configuration=config,
                )
            )

            scenario_recommendations[
                "elasticity_multiplier"
            ] = float(
                elasticity_multiplier
            )

            scenario_recommendations[
                "cost_multiplier"
            ] = float(
                cost_multiplier
            )

            scenario_recommendations[
                "sensitivity_scenario"
            ] = (
                "elasticity_"
                + str(
                    elasticity_multiplier
                )
                + "_cost_"
                + str(
                    cost_multiplier
                )
            )

            sensitivity_results.append(
                scenario_recommendations
            )

    if not sensitivity_results:
        return pd.DataFrame()

    sensitivity_analysis = pd.concat(
        sensitivity_results,
        ignore_index=True,
    )

    return sensitivity_analysis


# %%
def build_optimization_configuration(
    configuration: PricingOptimizationConfig,
) -> dict[str, Any]:
    """Convert the optimization configuration into a JSON-ready dictionary."""

    validate_pricing_configuration(
        configuration
    )

    configuration_data = {
        "module": "src.optimization.pricing",
        "method": "constant_elasticity_grid_search",
        **asdict(configuration),
    }

    return configuration_data        
