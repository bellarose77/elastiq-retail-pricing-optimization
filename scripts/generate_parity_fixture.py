"""
Generate a cross-language parity fixture for the pricing engine.

`src/optimization/pricing.py` (pipeline) and `app/frontend/src/lib/engine.js`
(ELASTIQ app) are two independent implementations of the same demand model.
Nothing previously asserted that they agree, and they had silently diverged on
defaults and on inventory semantics.

This writes a fixture of inputs plus the Python engine's answers.
`app/frontend/scripts/validate-engine.mjs` replays it through the JavaScript
engine and fails the build on any disagreement beyond tolerance.

Run after changing either engine:

    PYTHONPATH=. python scripts/generate_parity_fixture.py
"""

from __future__ import annotations

import json

from src.config import PROJECT_ROOT
from src.optimization.pricing import (
    PricingOptimizationConfig,
    generate_price_change_rates,
    recompute_price_financials,
    round_to_price_ending,
)

FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "engine_parity.json"
)

CASES = [
    {
        "label": "elastic, no promo, ample stock",
        "current_price": 10.0,
        "baseline_quantity": 100.0,
        "elasticity": -1.8,
        "unit_cost": 4.0,
        "candidate_price": 11.0,
        "promotion_active": False,
        "promotion_uplift_rate": 0.0,
        "promotion_cost_rate": 0.02,
        "available_inventory": None,
    },
    {
        "label": "inelastic, promo on, promo cost applies",
        "current_price": 8.5,
        "baseline_quantity": 60.0,
        "elasticity": -0.6,
        "unit_cost": 3.2,
        "candidate_price": 9.35,
        "promotion_active": True,
        "promotion_uplift_rate": 0.15,
        "promotion_cost_rate": 0.02,
        "available_inventory": None,
    },
    {
        "label": "stock binds: demand truncated at inventory",
        "current_price": 9.08,
        "baseline_quantity": 34.0,
        "elasticity": -0.09,
        "unit_cost": 4.37,
        "candidate_price": 8.5,
        "promotion_active": False,
        "promotion_uplift_rate": 0.0,
        "promotion_cost_rate": 0.0,
        "available_inventory": 34.0,
    },
    {
        "label": "price cut, elastic, stock ample",
        "current_price": 12.0,
        "baseline_quantity": 200.0,
        "elasticity": -2.4,
        "unit_cost": 5.0,
        "candidate_price": 10.8,
        "promotion_active": False,
        "promotion_uplift_rate": 0.0,
        "promotion_cost_rate": 0.0,
        "available_inventory": 500.0,
    },
]

GRID_CASES = [
    {"minimum": -0.20, "maximum": 0.20, "step": 0.05},
    {"minimum": 0.05, "maximum": 0.20, "step": 0.05},
    {"minimum": -0.20, "maximum": -0.05, "step": 0.05},
]

ROUNDING_CASES = [
    {"price": 12.34, "ending": 0.99},
    {"price": 12.90, "ending": 0.99},
    {"price": 0.50, "ending": 0.99},
    {"price": 7.05, "ending": 0.95},
]


def main() -> None:
    """Write the parity fixture."""

    candidates = []

    for case in CASES:
        financials = recompute_price_financials(
            candidate_price=case["candidate_price"],
            base_price=case["current_price"],
            baseline_quantity=case["baseline_quantity"],
            elasticity=case["elasticity"],
            unit_cost=case["unit_cost"],
            promotion_active=case["promotion_active"],
            promotion_uplift_rate=case["promotion_uplift_rate"],
            promotion_cost_rate=case["promotion_cost_rate"],
            available_inventory=case["available_inventory"],
            inventory_mode="cap",
        )

        candidates.append(
            {
                "input": case,
                "expected": {
                    "demand": financials["expected_quantity"],
                    "revenue": financials["expected_revenue"],
                    "profit": financials["expected_profit"],
                    "marginRate": financials["margin_rate"],
                    "inventoryBinding": financials["inventory_binding"],
                },
            }
        )

    grids = [
        {
            "input": grid_case,
            "expected": generate_price_change_rates(
                minimum_change_rate=grid_case["minimum"],
                maximum_change_rate=grid_case["maximum"],
                step=grid_case["step"],
            ).tolist(),
        }
        for grid_case in GRID_CASES
    ]

    roundings = [
        {
            "input": rounding_case,
            "expected": round_to_price_ending(
                rounding_case["price"],
                rounding_case["ending"],
            ),
        }
        for rounding_case in ROUNDING_CASES
    ]

    fixture = {
        "generated_by": "scripts/generate_parity_fixture.py",
        "python_defaults": {
            "objective": PricingOptimizationConfig().objective,
            "minPriceChangeRate": (
                PricingOptimizationConfig().minimum_price_change_rate
            ),
            "maxPriceChangeRate": (
                PricingOptimizationConfig().maximum_price_change_rate
            ),
            "priceChangeStep": (
                PricingOptimizationConfig().price_change_step
            ),
            "actionThreshold": (
                PricingOptimizationConfig().action_threshold
            ),
            "inventoryMode": PricingOptimizationConfig().inventory_mode,
        },
        "candidates": candidates,
        "grids": grids,
        "roundings": roundings,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    FIXTURE_PATH.write_text(
        json.dumps(fixture, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {FIXTURE_PATH.relative_to(PROJECT_ROOT)}")
    print(
        f"  {len(candidates)} candidate cases, {len(grids)} grid cases, "
        f"{len(roundings)} rounding cases"
    )


if __name__ == "__main__":
    main()
