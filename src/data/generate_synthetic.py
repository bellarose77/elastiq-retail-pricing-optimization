"""Generate reproducible synthetic retail-pricing and market-note datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"


PRODUCTS = pd.DataFrame(
    [
        ["P001", "Beverages", 4.49, 2.10, 38, -1.8],
        ["P002", "Beverages", 6.99, 3.20, 30, -2.1],
        ["P003", "Beverages", 9.49, 4.60, 22, -2.4],
        ["P004", "Grocery", 3.99, 1.75, 45, -1.4],
        ["P005", "Grocery", 5.49, 2.55, 36, -1.7],
        ["P006", "Grocery", 8.99, 4.30, 25, -2.0],
        ["P007", "Household", 7.49, 3.80, 20, -1.2],
        ["P008", "Household", 12.99, 6.50, 15, -1.5],
        ["P009", "Personal Care", 6.49, 3.00, 24, -1.3],
        ["P010", "Personal Care", 10.99, 5.25, 17, -1.6],
    ],
    columns=[
        "product_id",
        "category",
        "base_price",
        "base_cost",
        "base_daily_demand",
        "true_price_elasticity",
    ],
)


STORES = pd.DataFrame(
    [
        ["S001", "West", 1.10],
        ["S002", "West", 0.95],
        ["S003", "Central", 1.20],
        ["S004", "East", 1.00],
        ["S005", "East", 0.90],
    ],
    columns=["store_id", "region", "demand_factor"],
)


def generate_retail_data() -> pd.DataFrame:
    """Generate daily store-product pricing and sales observations."""

    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    holidays = {
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-19"),
        pd.Timestamp("2024-03-29"),
        pd.Timestamp("2024-05-20"),
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-09-02"),
        pd.Timestamp("2024-10-14"),
        pd.Timestamp("2024-12-25"),
        pd.Timestamp("2024-12-26"),
    }

    rows: list[dict] = []

    for date in dates:
        day_number = (date - dates[0]).days
        day_of_year = date.dayofyear

        weekend_flag = int(date.dayofweek >= 5)
        holiday_flag = int(date.normalize() in holidays)

        annual_seasonality = np.sin(
            2 * np.pi * day_of_year / 365.25
        )

        supplier_cost_index = (
            1.0
            + 0.035 * np.sin(2 * np.pi * day_number / 120)
            + rng.normal(0, 0.008)
        )

        shipping_cost_index = (
            1.0
            + 0.025 * np.cos(2 * np.pi * day_number / 90)
            + rng.normal(0, 0.006)
        )

        weather_index = np.clip(
            0.50
            + 0.35 * annual_seasonality
            + rng.normal(0, 0.08),
            0,
            1,
        )

        for store in STORES.itertuples(index=False):
            customer_traffic = max(
                250,
                round(
                    850
                    * store.demand_factor
                    * (1 + 0.12 * weekend_flag)
                    * (1 + 0.20 * holiday_flag)
                    * (1 + rng.normal(0, 0.08))
                ),
            )

            for product in PRODUCTS.itertuples(index=False):
                
                
                latent_demand_shock = rng.normal(0, 0.18)
                cost_pass_through = (
                0.80 * (supplier_cost_index - 1)
                + 0.50 * (shipping_cost_index - 1)
                )
                regular_price = product.base_price * (
                1
                + 0.015 * annual_seasonality
                + cost_pass_through
                + 0.12 * latent_demand_shock
                + rng.normal(0, 0.010)
                )

                promotion_propensity = np.clip(
                    0.13
                    + 0.06 * weekend_flag
                    + 0.07 * holiday_flag
                    + 0.03 * int(product.category == "Beverages"),
                    0.05,
                    0.45,
                )

                promotion_flag = int(
                    rng.random() < promotion_propensity
                )

                if promotion_flag:
                    promotion_type = rng.choice(
                        [
                            "percentage_discount",
                            "feature_display",
                            "bundle_offer",
                        ],
                        p=[0.55, 0.25, 0.20],
                    )

                    if promotion_type == "percentage_discount":
                        discount_pct = rng.choice(
                            [0.10, 0.15, 0.20, 0.25]
                        )
                    elif promotion_type == "bundle_offer":
                        discount_pct = rng.choice(
                            [0.08, 0.12, 0.15]
                        )
                    else:
                        discount_pct = rng.choice(
                            [0.05, 0.08, 0.10]
                        )
                else:
                    promotion_type = "none"
                    discount_pct = 0.0

                selling_price = regular_price * (1 - discount_pct)

                competitor_price = regular_price * (
                    1
                    + rng.normal(0.015, 0.055)
                    - 0.03 * promotion_flag
                )

                unit_cost = product.base_cost * (
                    0.70 * supplier_cost_index
                    + 0.30 * shipping_cost_index
                )

                advertising_spend = max(
                    0,
                    rng.normal(
                        18 + 45 * promotion_flag,
                        8 + 6 * promotion_flag,
                    ),
                )

                promotion_uplift = (
                    1
                    + 0.18 * promotion_flag
                    + 0.08
                    * int(promotion_type == "feature_display")
                )

                price_effect = (
                    selling_price / product.base_price
                ) ** product.true_price_elasticity

                competitor_effect = (
                    competitor_price / selling_price
                ) ** 0.30

                traffic_effect = (
                    customer_traffic / 850
                ) ** 0.45

                seasonal_effect = (
                    1
                    + 0.08 * annual_seasonality
                    + 0.12 * weekend_flag
                    + 0.22 * holiday_flag
                )

                advertising_effect = (
                    1 + 0.0025 * advertising_spend
                )

                expected_demand = (
                product.base_daily_demand
                * store.demand_factor
                * price_effect
                * competitor_effect
                * traffic_effect
                * seasonal_effect
                * promotion_uplift
                * advertising_effect
                * np.exp(latent_demand_shock)
                )

                expected_demand = max(expected_demand, 0.10)

                unconstrained_sales = rng.poisson(expected_demand)

                inventory_level = max(
                    0,
                    round(
                        expected_demand
                        * rng.uniform(0.85, 1.55)
                    ),
                )

                units_sold = min(
                    unconstrained_sales,
                    inventory_level,
                )

                stockout_flag = int(
                    unconstrained_sales > inventory_level
                )

                revenue = units_sold * selling_price
                gross_profit = units_sold * (
                    selling_price - unit_cost
                )

                rows.append(
                    {
                        "date": date,
                        "store_id": store.store_id,
                        "region": store.region,
                        "product_id": product.product_id,
                        "category": product.category,
                        "year": date.year,
                        "month": date.month,
                        "day_of_week": date.day_name(),
                        "weekend_flag": weekend_flag,
                        "holiday_flag": holiday_flag,
                        "regular_price": round(regular_price, 2),
                        "selling_price": round(selling_price, 2),
                        "competitor_price": round(
                            competitor_price,
                            2,
                        ),
                        "unit_cost": round(unit_cost, 2),
                        "discount_pct": round(
                            discount_pct,
                            2,
                        ),
                        "promotion_flag": promotion_flag,
                        "promotion_type": promotion_type,
                        "promotion_propensity": round(
                            promotion_propensity,
                            4,
                        ),
                        "advertising_spend": round(
                            advertising_spend,
                            2,
                        ),
                        "supplier_cost_index": round(
                            supplier_cost_index,
                            4,
                        ),
                        "shipping_cost_index": round(
                            shipping_cost_index,
                            4,
                        ),
                        "weather_index": round(
                            weather_index,
                            4,
                        ),
                        "customer_traffic": customer_traffic,
                        "inventory_level": inventory_level,
                        "stockout_flag": stockout_flag,
                        "units_sold": units_sold,
                        "revenue": round(revenue, 2),
                        "gross_profit": round(
                            gross_profit,
                            2,
                        ),
                        "true_price_elasticity": (
                            product.true_price_elasticity
                        ),
                    }
                )

    return pd.DataFrame(rows)


def generate_market_notes() -> pd.DataFrame:
    """Generate synthetic text notes for the future LLM-RAG workflow."""

    rng = np.random.default_rng(SEED + 1)
    dates = pd.date_range("2024-01-01", "2024-12-31")

    event_templates = [
        {
            "event_type": "competitor_discount",
            "sentiment": "negative",
            "expected_demand_effect": "decrease",
            "impact_score": -0.70,
            "text": (
                "A competing retailer introduced a significant "
                "discount on comparable {category} products."
            ),
        },
        {
            "event_type": "local_event",
            "sentiment": "positive",
            "expected_demand_effect": "increase",
            "impact_score": 0.55,
            "text": (
                "A major community event is expected to increase "
                "store traffic and demand for {category} products."
            ),
        },
        {
            "event_type": "supplier_disruption",
            "sentiment": "negative",
            "expected_demand_effect": "uncertain",
            "impact_score": -0.45,
            "text": (
                "A supplier disruption may reduce availability "
                "and increase procurement costs for {category}."
            ),
        },
        {
            "event_type": "seasonal_demand",
            "sentiment": "positive",
            "expected_demand_effect": "increase",
            "impact_score": 0.65,
            "text": (
                "Seasonal demand for {category} is strengthening "
                "ahead of the upcoming holiday period."
            ),
        },
        {
            "event_type": "consumer_trend",
            "sentiment": "positive",
            "expected_demand_effect": "increase",
            "impact_score": 0.40,
            "text": (
                "Recent customer feedback indicates growing "
                "interest in products within the {category} category."
            ),
        },
        {
            "event_type": "weather_event",
            "sentiment": "neutral",
            "expected_demand_effect": "mixed",
            "impact_score": 0.10,
            "text": (
                "Forecast weather conditions may change customer "
                "traffic and short-term demand for {category}."
            ),
        },
    ]

    sources = [
        "Store Manager",
        "Market Intelligence",
        "Supplier Update",
        "Competitor Review",
        "Regional Sales Team",
    ]

    rows: list[dict] = []

    for note_number in range(1, 101):
        product = PRODUCTS.iloc[
            rng.integers(0, len(PRODUCTS))
        ]

        event = event_templates[
            rng.integers(0, len(event_templates))
        ]

        note_date = dates[
            rng.integers(0, len(dates))
        ]

        region = rng.choice(
            ["West", "Central", "East"]
        )

        rows.append(
            {
                "note_id": f"N{note_number:03d}",
                "date": note_date,
                "product_id": product["product_id"],
                "category": product["category"],
                "region": region,
                "source": rng.choice(sources),
                "market_note": event["text"].format(
                    category=product["category"]
                ),
                "sentiment": event["sentiment"],
                "event_type": event["event_type"],
                "expected_demand_effect": (
                    event["expected_demand_effect"]
                ),
                "impact_score": event["impact_score"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save both synthetic datasets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retail_data = generate_retail_data()
    market_notes = generate_market_notes()

    retail_path = (
        OUTPUT_DIR / "sample_retail_pricing.csv"
    )
    notes_path = OUTPUT_DIR / "market_notes.csv"

    retail_data.to_csv(retail_path, index=False)
    market_notes.to_csv(notes_path, index=False)

    print(f"Retail data saved: {retail_path}")
    print(f"Retail data shape: {retail_data.shape}")

    print(f"Market notes saved: {notes_path}")
    print(f"Market notes shape: {market_notes.shape}")


if __name__ == "__main__":
   main()