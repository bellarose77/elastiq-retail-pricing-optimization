"""Central project paths and configuration settings."""

from pathlib import Path


# Project root:
# retail-pricing-promotion-optimization/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"
TESTS_DIR = PROJECT_ROOT / "tests"


def create_project_directories() -> None:
    """Create project output directories when they do not already exist."""

    directories = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MODEL_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        APP_DIR,
        TESTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


create_project_directories()


# ---------------------------------------------------------
# Data schema constants
# ---------------------------------------------------------

RETAIL_REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "region",
    "product_id",
    "category",
    "regular_price",
    "selling_price",
    "competitor_price",
    "unit_cost",
    "discount_pct",
    "promotion_flag",
    "promotion_type",
    "inventory_level",
    "stockout_flag",
    "units_sold",
    "revenue",
    "gross_profit",
]

RETAIL_NUMERIC_COLUMNS = [
    "regular_price",
    "selling_price",
    "competitor_price",
    "unit_cost",
    "discount_rate",
    "advertising_spend",
    "inventory_level",
    "units_sold",
    "revenue",
    "gross_profit",
    "margin_rate",
    "stockout_flag",
    "weather_index",
    "customer_traffic",
]


# ---------------------------------------------------------
# Elasticity analysis constants
# ---------------------------------------------------------

ELASTICITY_COLUMN_ALIASES = {
    "units_sold": "quantity",
    "selling_price": "price",
    "promotion_flag": "promotion",
    "holiday_flag": "holiday",
    "weekend_flag": "weekend",
    "inventory_level": "inventory",
    "true_price_elasticity": "true_elasticity",
    "advertising_spend": "marketing_spend",
}


# ---------------------------------------------------------
# Forecasting configuration
# ---------------------------------------------------------

FORECAST_LAG_PERIODS = [1, 2, 3, 7, 14]
FORECAST_ROLLING_WINDOWS = [7, 14, 30]


# ---------------------------------------------------------
# RAG configuration
# ---------------------------------------------------------

RAG_CHUNK_SIZE = 120
RAG_CHUNK_OVERLAP = 20
RAG_TOP_K = 5
RAG_MINIMUM_SIMILARITY = 0.0


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15