"""Exploratory data analysis and reporting functions."""

from src.analysis.eda import (
    calculate_kpi_summary,
    calculate_price_demand_correlation,
    create_discount_bands,
    summarize_by_category,
    summarize_by_product,
    summarize_by_region,
    summarize_by_store,
)

__all__ = [
    "calculate_kpi_summary",
    "calculate_price_demand_correlation",
    "create_discount_bands",
    "summarize_by_category",
    "summarize_by_product",
    "summarize_by_region",
    "summarize_by_store",
]
