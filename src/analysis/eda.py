"""
Exploratory Data Analysis Functions

This module provides reusable functions for exploratory data analysis
of retail pricing and promotion data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import safe_divide


def calculate_kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate high-level KPIs for the retail dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with required columns: store_id, product_id, category,
        units_sold, revenue, gross_profit, selling_price, is_promotion,
        discount_rate, stockout_flag

    Returns
    -------
    pd.DataFrame
        Summary with columns 'metric' and 'value'
    """
    promotion_mask = df["is_promotion"].eq(1)

    total_revenue = float(df["revenue"].sum())
    total_gross_profit = float(df["gross_profit"].sum())

    overall_gross_margin_rate = (
        total_gross_profit / total_revenue
        if total_revenue != 0
        else np.nan
    )

    return pd.DataFrame(
        {
            "metric": [
                "Number of observations",
                "Number of stores",
                "Number of products",
                "Number of categories",
                "Total units sold",
                "Total revenue",
                "Total gross profit",
                "Average selling price",
                "Average units per observation",
                "Promotion rate",
                "Average promotional discount rate",
                "Stockout rate",
                "Overall gross margin rate",
            ],
            "value": [
                len(df),
                df["store_id"].nunique(),
                df["product_id"].nunique(),
                df["category"].nunique(),
                df["units_sold"].sum(),
                total_revenue,
                total_gross_profit,
                df["selling_price"].mean(),
                df["units_sold"].mean(),
                df["is_promotion"].mean(),
                df.loc[promotion_mask, "discount_rate"].mean(),
                df["stockout_flag"].mean(),
                overall_gross_margin_rate,
            ],
        }
    )


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by product category.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with columns: category, revenue, gross_profit,
        units_sold, is_promotion

    Returns
    -------
    pd.DataFrame
        Category-level aggregated metrics
    """
    category_summary = (
        df.groupby("category", as_index=False)
        .agg(
            {
                "revenue": "sum",
                "gross_profit": "sum",
                "units_sold": "sum",
                "is_promotion": "mean",
            }
        )
        .rename(
            columns={
                "is_promotion": "promotion_rate",
            }
        )
    )

    category_summary["margin_rate"] = safe_divide(
        category_summary["gross_profit"],
        category_summary["revenue"],
    )

    return category_summary.sort_values(
        "revenue",
        ascending=False,
    )


def summarize_by_product(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Aggregate metrics by product and return top N by revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with columns: product_id, revenue, gross_profit,
        units_sold, category
    top_n : int, default=20
        Number of top products to return

    Returns
    -------
    pd.DataFrame
        Product-level aggregated metrics for top N products
    """
    product_summary = (
        df.groupby(
            ["product_id", "category"],
            as_index=False,
        )
        .agg(
            {
                "revenue": "sum",
                "gross_profit": "sum",
                "units_sold": "sum",
            }
        )
    )

    product_summary["margin_rate"] = safe_divide(
        product_summary["gross_profit"],
        product_summary["revenue"],
    )

    return product_summary.nlargest(
        top_n,
        "revenue",
    )


def summarize_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by store.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with columns: store_id, region, revenue,
        gross_profit, units_sold

    Returns
    -------
    pd.DataFrame
        Store-level aggregated metrics
    """
    store_summary = (
        df.groupby(
            ["store_id", "region"],
            as_index=False,
        )
        .agg(
            {
                "revenue": "sum",
                "gross_profit": "sum",
                "units_sold": "sum",
            }
        )
    )

    store_summary["margin_rate"] = safe_divide(
        store_summary["gross_profit"],
        store_summary["revenue"],
    )

    return store_summary.sort_values(
        "revenue",
        ascending=False,
    )


def summarize_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by region.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with columns: region, revenue, gross_profit,
        units_sold

    Returns
    -------
    pd.DataFrame
        Region-level aggregated metrics
    """
    region_summary = (
        df.groupby("region", as_index=False)
        .agg(
            {
                "revenue": "sum",
                "gross_profit": "sum",
                "units_sold": "sum",
            }
        )
    )

    region_summary["margin_rate"] = safe_divide(
        region_summary["gross_profit"],
        region_summary["revenue"],
    )

    return region_summary.sort_values(
        "revenue",
        ascending=False,
    )


def create_discount_bands(
    df: pd.DataFrame,
    discount_column: str = "discount_rate",
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create discount bands and summarize demand response.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with discount and demand columns
    discount_column : str, default='discount_rate'
        Name of the discount percentage column
    bins : list[float] | None
        Bin edges for discount bands. Default: [0, 0.1, 0.2, 0.3, 1.0]
    labels : list[str] | None
        Labels for discount bands. Default: ['0-10%', '10-20%', '20-30%', '30%+']

    Returns
    -------
    pd.DataFrame
        Discount band summary with average units and revenue
    """
    if bins is None:
        bins = [0, 0.1, 0.2, 0.3, 1.0]

    if labels is None:
        labels = ["0-10%", "10-20%", "20-30%", "30%+"]

    df_copy = df.copy()
    df_copy["discount_band"] = pd.cut(
        df_copy[discount_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return (
        df_copy.groupby("discount_band", as_index=False, observed=False)
        .agg(
            {
                "units_sold": ["mean", "sum"],
                "revenue": ["mean", "sum"],
            }
        )
    )


def calculate_price_demand_correlation(
    df: pd.DataFrame,
    group_by: str | list[str] = "category",
) -> pd.DataFrame:
    """
    Calculate correlation between price and demand by group.

    Parameters
    ----------
    df : pd.DataFrame
        Retail data with columns: selling_price, units_sold
    group_by : str | list[str], default='category'
        Column(s) to group by for correlation calculation

    Returns
    -------
    pd.DataFrame
        Correlation coefficients by group
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    correlations = []

    for group_values, group_df in df.groupby(group_by):
        if len(group_df) < 2:
            continue

        corr = group_df[["selling_price", "units_sold"]].corr().iloc[0, 1]

        # Grouping by a list always yields tuple keys in pandas, even for
        # a single column, so normalize before zipping with group_by.
        group_values_tuple = (
            group_values
            if isinstance(group_values, tuple)
            else (group_values,)
        )

        group_dict = dict(zip(group_by, group_values_tuple))

        group_dict["price_demand_correlation"] = corr
        group_dict["n_observations"] = len(group_df)

        correlations.append(group_dict)

    return pd.DataFrame(correlations)
