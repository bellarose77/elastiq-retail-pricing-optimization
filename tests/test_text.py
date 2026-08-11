"""Unit tests for src/features/text.py"""

import pandas as pd
import pytest

from src.features.text import normalize_category_key


class TestNormalizeCategoryKey:
    """Test normalize_category_key function."""

    def test_lowercase_conversion(self):
        """Test that category names are converted to lowercase."""
        assert normalize_category_key("BEVERAGES") == "beverage"
        assert normalize_category_key("GROCERY") == "grocery"

    def test_special_character_removal(self):
        """Test that special characters are replaced with underscores."""
        assert normalize_category_key("Personal Care Products") == "personal_care"
        assert normalize_category_key("Food & Beverage") == "food_beverage"

    def test_none_returns_unknown(self):
        """Test that None values return 'unknown'."""
        assert normalize_category_key(None) == "unknown"

    def test_nan_returns_unknown(self):
        """Test that NaN values return 'unknown'."""
        assert normalize_category_key(float("nan")) == "unknown"
        assert normalize_category_key(pd.NA) == "unknown"

    def test_alias_mapping_beverages(self):
        """Test that 'beverages' maps to 'beverage'."""
        assert normalize_category_key("Beverages") == "beverage"
        assert normalize_category_key("beverages") == "beverage"
        assert normalize_category_key("Drinks") == "beverage"

    def test_alias_mapping_grocery(self):
        """Test that grocery variants map to 'grocery'."""
        assert normalize_category_key("Groceries") == "grocery"
        assert normalize_category_key("Grocery") == "grocery"
        assert normalize_category_key("Foods") == "grocery"
        assert normalize_category_key("Food") == "grocery"

    def test_alias_mapping_household(self):
        """Test that household variants map to 'household'."""
        assert normalize_category_key("Households") == "household"
        assert normalize_category_key("Household Products") == "household"

    def test_alias_mapping_personal_care(self):
        """Test that personal care variants map to 'personal_care'."""
        assert normalize_category_key("Personal Care Products") == "personal_care"
        assert normalize_category_key("PersonalCare") == "personal_care"

    def test_unlisted_plural_stems_to_singular(self):
        """Categories with no explicit alias still normalize via plural stemming."""
        assert normalize_category_key("hobbies") == "hobby"
        assert normalize_category_key("categories") == "category"
        assert normalize_category_key("Snacks") == "snack"

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        assert normalize_category_key("  Beverages  ") == "beverage"
        assert normalize_category_key("\tGrocery\n") == "grocery"

    def test_numeric_values_converted_to_string(self):
        """Test that numeric values are converted to string."""
        result = normalize_category_key(123)
        assert isinstance(result, str)
        assert result == "123"

    def test_empty_string_handling(self):
        """Test that empty strings are handled properly."""
        result = normalize_category_key("")
        assert result == "unknown" or result == ""

    def test_idempotent_for_normalized_values(self):
        """Test that already normalized values remain unchanged."""
        normalized = "beverage"
        assert normalize_category_key(normalized) == normalized

    def test_multiple_underscores_collapsed(self):
        """Test that multiple consecutive underscores are handled."""
        result = normalize_category_key("Category  --  Name")
        assert "__" not in result  # No double underscores

    def test_returns_string_type(self):
        """Test that the function always returns a string."""
        test_inputs = ["Beverages", None, 123, pd.NA]
        for input_val in test_inputs:
            result = normalize_category_key(input_val)
            assert isinstance(result, str)
