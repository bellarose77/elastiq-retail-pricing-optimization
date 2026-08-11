"""
Text Processing Functions

This module provides functions for text normalization and processing,
particularly for category labels and product descriptions.
"""

from __future__ import annotations

import re

import pandas as pd


def normalize_category_key(value: object) -> str:
    """
    Convert category labels into stable matching keys.

    This function normalizes category names by:
    1. Converting to lowercase
    2. Removing special characters
    3. Applying common aliases (e.g., "beverages" -> "beverage")

    Parameters
    ----------
    value : object
        Category value to normalize (str, or missing value)

    Returns
    -------
    str
        Normalized category key

    Examples
    --------
    >>> normalize_category_key("Beverages")
    'beverage'
    >>> normalize_category_key("Personal Care Products")
    'personal_care'
    >>> normalize_category_key(None)
    'unknown'
    """
    if value is None or pd.isna(value):
        return "unknown"

    normalized_value = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")

    aliases = {
        "beverages": "beverage",
        "beverage": "beverage",
        "drinks": "beverage",
        "groceries": "grocery",
        "grocery": "grocery",
        "foods": "grocery",
        "food": "grocery",
        "households": "household",
        "household_products": "household",
        "personal_care_products": "personal_care",
        "personalcare": "personal_care",
    }

    return aliases.get(
        normalized_value,
        normalized_value,
    )
