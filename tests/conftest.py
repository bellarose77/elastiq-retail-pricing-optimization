"""Pytest configuration and shared fixtures."""

import matplotlib

# Tests only need Figure/Axes objects, never an on-screen window, and the
# default interactive backend (e.g. TkAgg) depends on a working system Tk
# installation that isn't guaranteed to be present. Force the
# non-interactive Agg backend before matplotlib.pyplot is imported
# anywhere (including transitively via src.visualization.plots) so the
# suite doesn't depend on the local machine's GUI toolkit at all.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_retail_dataframe():
    """
    Create a sample retail dataset for testing.

    Returns
    -------
    pd.DataFrame
        Sample retail data with typical columns
    """
    np.random.seed(42)

    n_rows = 100
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    products = [f"PROD_{i%10}" for i in range(n_rows)]
    categories = ["Beverage", "Grocery", "Household"] * (n_rows // 3) + ["Beverage"]

    return pd.DataFrame(
        {
            "date": dates,
            "store_id": [f"STORE_{i%5}" for i in range(n_rows)],
            "region": ["North", "South", "East", "West"] * (n_rows // 4),
            "product_id": products,
            "category": categories,
            "regular_price": np.random.uniform(5, 20, n_rows),
            "selling_price": np.random.uniform(4, 19, n_rows),
            "competitor_price": np.random.uniform(4, 21, n_rows),
            "unit_cost": np.random.uniform(2, 10, n_rows),
            "discount_pct": np.random.uniform(0, 0.3, n_rows),
            "promotion_flag": np.random.choice([0, 1], n_rows),
            "promotion_type": np.random.choice(["None", "BOGO", "Discount"], n_rows),
            "inventory_level": np.random.randint(0, 1000, n_rows),
            "stockout_flag": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
            "units_sold": np.random.randint(10, 200, n_rows),
            "revenue": np.random.uniform(50, 2000, n_rows),
            "gross_profit": np.random.uniform(10, 800, n_rows),
        }
    )


@pytest.fixture
def sample_time_series_data():
    """
    Create sample time series data for testing.

    Returns
    -------
    pd.DataFrame
        Time series data with date, product, price, and quantity
    """
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    products = ["PROD_A", "PROD_B"] * (len(dates) // 2) + ["PROD_A"]

    return pd.DataFrame(
        {
            "date": dates,
            "product_id": products,
            "price": np.random.uniform(8, 12, len(dates)),
            "quantity": np.random.randint(50, 150, len(dates)),
        }
    )


@pytest.fixture
def sample_elasticity_data():
    """
    Create sample elasticity estimation data.

    Returns
    -------
    pd.DataFrame
        Data suitable for elasticity estimation
    """
    np.random.seed(42)

    n_rows = 200
    base_price = 10
    base_quantity = 100
    elasticity = -1.5

    # Generate prices with some variation
    prices = base_price + np.random.uniform(-2, 2, n_rows)

    # Generate quantities based on price elasticity
    quantities = base_quantity * (prices / base_price) ** elasticity
    quantities += np.random.normal(0, 10, n_rows)  # Add noise
    quantities = np.maximum(quantities, 1)  # Ensure positive

    return pd.DataFrame(
        {
            "product_id": [f"PROD_{i%10}" for i in range(n_rows)],
            "category": ["Beverage", "Grocery"] * (n_rows // 2),
            "price": prices,
            "quantity": quantities,
            "promotion": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def sample_optimization_data():
    """
    Create sample data for price optimization.

    Returns
    -------
    pd.DataFrame
        Data with current prices, quantities, elasticities, and costs
    """
    np.random.seed(42)

    n_items = 50

    return pd.DataFrame(
        {
            "item_id": [f"ITEM_{i:03d}" for i in range(n_items)],
            "category": np.random.choice(
                ["Beverage", "Grocery", "Household"],
                n_items,
            ),
            "current_price": np.random.uniform(5, 25, n_items),
            "baseline_quantity": np.random.randint(50, 500, n_items),
            "elasticity": np.random.uniform(-2.5, -0.5, n_items),
            "unit_cost": np.random.uniform(2, 15, n_items),
        }
    )


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )
