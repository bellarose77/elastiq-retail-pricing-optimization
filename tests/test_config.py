"""Unit tests for src/config.py"""

from pathlib import Path

import pytest

from src import config


class TestProjectPaths:
    """Test project path configurations."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is a valid path."""
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()

    def test_project_root_contains_src(self):
        """Test that PROJECT_ROOT contains the src directory."""
        src_dir = config.PROJECT_ROOT / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_data_directories_defined(self):
        """Test that all data directories are defined."""
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.RAW_DIR, Path)
        assert isinstance(config.INTERIM_DIR, Path)
        assert isinstance(config.PROCESSED_DIR, Path)

    def test_output_directories_defined(self):
        """Test that output directories are defined."""
        assert isinstance(config.MODEL_DIR, Path)
        assert isinstance(config.REPORTS_DIR, Path)
        assert isinstance(config.FIGURES_DIR, Path)

    def test_directory_hierarchy(self):
        """Test that directories follow expected hierarchy."""
        assert config.RAW_DIR.parent == config.DATA_DIR
        assert config.INTERIM_DIR.parent == config.DATA_DIR
        assert config.PROCESSED_DIR.parent == config.DATA_DIR
        assert config.FIGURES_DIR.parent == config.REPORTS_DIR


class TestCreateProjectDirectories:
    """Test project directory creation function."""

    def test_create_directories_creates_all_paths(self, tmp_path, monkeypatch):
        """Test that create_project_directories creates all required directories."""
        # Create temporary directory structure
        test_root = tmp_path / "test_project"
        test_root.mkdir()

        # Monkeypatch the config paths
        monkeypatch.setattr(config, "PROJECT_ROOT", test_root)
        monkeypatch.setattr(config, "DATA_DIR", test_root / "data")
        monkeypatch.setattr(config, "RAW_DIR", test_root / "data" / "raw")
        monkeypatch.setattr(config, "INTERIM_DIR", test_root / "data" / "interim")
        monkeypatch.setattr(config, "PROCESSED_DIR", test_root / "data" / "processed")
        monkeypatch.setattr(config, "MODEL_DIR", test_root / "models")
        monkeypatch.setattr(config, "REPORTS_DIR", test_root / "reports")
        monkeypatch.setattr(config, "FIGURES_DIR", test_root / "reports" / "figures")
        monkeypatch.setattr(config, "APP_DIR", test_root / "app")
        monkeypatch.setattr(config, "TESTS_DIR", test_root / "tests")

        # Call the function
        config.create_project_directories()

        # Verify all directories were created
        assert (test_root / "data" / "raw").exists()
        assert (test_root / "data" / "interim").exists()
        assert (test_root / "data" / "processed").exists()
        assert (test_root / "models").exists()
        assert (test_root / "reports").exists()
        assert (test_root / "reports" / "figures").exists()
        assert (test_root / "app").exists()
        assert (test_root / "tests").exists()

    def test_create_directories_idempotent(self, tmp_path, monkeypatch):
        """Test that create_project_directories can be called multiple times."""
        test_root = tmp_path / "test_project"
        test_root.mkdir()

        monkeypatch.setattr(config, "RAW_DIR", test_root / "data" / "raw")
        monkeypatch.setattr(config, "INTERIM_DIR", test_root / "data" / "interim")
        monkeypatch.setattr(config, "PROCESSED_DIR", test_root / "data" / "processed")
        monkeypatch.setattr(config, "MODEL_DIR", test_root / "models")
        monkeypatch.setattr(config, "REPORTS_DIR", test_root / "reports")
        monkeypatch.setattr(config, "FIGURES_DIR", test_root / "reports" / "figures")
        monkeypatch.setattr(config, "APP_DIR", test_root / "app")
        monkeypatch.setattr(config, "TESTS_DIR", test_root / "tests")

        # Call multiple times
        config.create_project_directories()
        config.create_project_directories()

        # Should still work without errors
        assert (test_root / "models").exists()


class TestDataSchemaConstants:
    """Test data schema constant definitions."""

    def test_required_columns_defined(self):
        """Test that RETAIL_REQUIRED_COLUMNS is defined and non-empty."""
        assert isinstance(config.RETAIL_REQUIRED_COLUMNS, list)
        assert len(config.RETAIL_REQUIRED_COLUMNS) > 0

    def test_required_columns_contains_expected_fields(self):
        """Test that required columns contain key retail fields."""
        expected_fields = [
            "date",
            "product_id",
            "category",
            "selling_price",
            "units_sold",
            "revenue",
        ]
        for field in expected_fields:
            assert field in config.RETAIL_REQUIRED_COLUMNS

    def test_numeric_columns_defined(self):
        """Test that RETAIL_NUMERIC_COLUMNS is defined."""
        assert isinstance(config.RETAIL_NUMERIC_COLUMNS, list)
        assert len(config.RETAIL_NUMERIC_COLUMNS) > 0


class TestElasticityConstants:
    """Test elasticity analysis constants."""

    def test_elasticity_column_aliases_defined(self):
        """Test that ELASTICITY_COLUMN_ALIASES is defined."""
        assert isinstance(config.ELASTICITY_COLUMN_ALIASES, dict)
        assert len(config.ELASTICITY_COLUMN_ALIASES) > 0

    def test_elasticity_aliases_are_strings(self):
        """Test that all keys and values are strings."""
        for key, value in config.ELASTICITY_COLUMN_ALIASES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestForecastingConfiguration:
    """Test forecasting configuration constants."""

    def test_forecast_lag_periods_defined(self):
        """Test that FORECAST_LAG_PERIODS is defined."""
        assert isinstance(config.FORECAST_LAG_PERIODS, list)
        assert len(config.FORECAST_LAG_PERIODS) > 0

    def test_forecast_lag_periods_are_positive(self):
        """Test that all lag periods are positive integers."""
        for lag in config.FORECAST_LAG_PERIODS:
            assert isinstance(lag, int)
            assert lag > 0

    def test_forecast_rolling_windows_defined(self):
        """Test that FORECAST_ROLLING_WINDOWS is defined."""
        assert isinstance(config.FORECAST_ROLLING_WINDOWS, list)
        assert len(config.FORECAST_ROLLING_WINDOWS) > 0

    def test_forecast_rolling_windows_are_positive(self):
        """Test that all rolling windows are positive integers."""
        for window in config.FORECAST_ROLLING_WINDOWS:
            assert isinstance(window, int)
            assert window > 0


class TestRAGConfiguration:
    """Test RAG configuration constants."""

    def test_rag_chunk_size_is_positive(self):
        """Test that RAG_CHUNK_SIZE is a positive integer."""
        assert isinstance(config.RAG_CHUNK_SIZE, int)
        assert config.RAG_CHUNK_SIZE > 0

    def test_rag_chunk_overlap_is_non_negative(self):
        """Test that RAG_CHUNK_OVERLAP is non-negative."""
        assert isinstance(config.RAG_CHUNK_OVERLAP, int)
        assert config.RAG_CHUNK_OVERLAP >= 0

    def test_rag_chunk_overlap_less_than_size(self):
        """Test that chunk overlap is less than chunk size."""
        assert config.RAG_CHUNK_OVERLAP < config.RAG_CHUNK_SIZE

    def test_rag_top_k_is_positive(self):
        """Test that RAG_TOP_K is a positive integer."""
        assert isinstance(config.RAG_TOP_K, int)
        assert config.RAG_TOP_K > 0

    def test_rag_minimum_similarity_is_valid(self):
        """Test that RAG_MINIMUM_SIMILARITY is a valid float."""
        assert isinstance(config.RAG_MINIMUM_SIMILARITY, (int, float))
        assert 0.0 <= config.RAG_MINIMUM_SIMILARITY <= 1.0


class TestTrainingConfiguration:
    """Test training configuration constants."""

    def test_random_state_defined(self):
        """Test that RANDOM_STATE is defined."""
        assert isinstance(config.RANDOM_STATE, int)
        assert config.RANDOM_STATE >= 0

    def test_train_fraction_is_valid(self):
        """Test that TRAIN_FRACTION is between 0 and 1."""
        assert isinstance(config.TRAIN_FRACTION, (int, float))
        assert 0.0 < config.TRAIN_FRACTION < 1.0

    def test_val_fraction_is_valid(self):
        """Test that VAL_FRACTION is between 0 and 1."""
        assert isinstance(config.VAL_FRACTION, (int, float))
        assert 0.0 < config.VAL_FRACTION < 1.0

    def test_train_and_val_sum_less_than_one(self):
        """Test that train + val fractions leave room for test set."""
        total = config.TRAIN_FRACTION + config.VAL_FRACTION
        assert total < 1.0
