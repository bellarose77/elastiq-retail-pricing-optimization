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

    if normalized_value in aliases:
        return aliases[normalized_value]

    # No explicit alias: fall back to stripping an ordinary plural so
    # unlisted categories ("Hobbies", "Categories") still normalize
    # consistently instead of staying pluralized.
    if normalized_value.endswith("ies") and len(normalized_value) > 4:
        return normalized_value[:-3] + "y"

    if (
        normalized_value.endswith("s")
        and not normalized_value.endswith("ss")
        and len(normalized_value) > 3
    ):
        return normalized_value[:-1]

    return normalized_value
