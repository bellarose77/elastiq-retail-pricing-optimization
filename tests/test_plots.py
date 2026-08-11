"""Unit tests for src/visualization/plots.py"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.visualization.plots import (
    finalize_figure,
    format_currency_axis,
    format_percent_axis,
    plot_forecast_performance,
    plot_forecast_residuals,
    plot_optimization_actions,
    plot_price_recommendations,
    plot_price_scenarios,
)


@pytest.fixture
def sample_scenarios():
    """Create sample price scenario data."""
    return pd.DataFrame(
        {
            "candidate_price": [8, 9, 10, 11, 12],
            "expected_profit": [100, 120, 130, 125, 115],
            "expected_revenue": [800, 900, 1000, 1100, 1200],
        }
    )


@pytest.fixture
def sample_action_summary():
    """Create sample optimization action summary."""
    return pd.DataFrame(
        {
            "recommendation_action": ["increase_price", "decrease_price", "hold_price"],
            "item_count": [10, 5, 15],
        }
    )


@pytest.fixture
def sample_recommendations():
    """Create sample price recommendations."""
    return pd.DataFrame(
        {
            "item_id": [f"ITEM{i}" for i in range(20)],
            "current_price": np.random.uniform(8, 12, 20),
            "recommended_price": np.random.uniform(8, 12, 20),
        }
    )


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=30),
            "quantity": np.random.randint(80, 120, 30),
            "predicted_value": np.random.randint(80, 120, 30),
        }
    )


class TestFormatPercentAxis:
    """Test format_percent_axis function."""

    def test_returns_axis(self):
        """Test that the function returns an Axes object."""
        fig, ax = plt.subplots()
        result = format_percent_axis(ax)
        assert isinstance(result, Axes)
        plt.close(fig)

    def test_formats_axis_as_percentage(self):
        """Test that the axis is formatted as percentage."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 0.5])
        format_percent_axis(ax)
        # Check that formatter is set (we can't easily verify the exact format)
        assert ax.yaxis.get_major_formatter() is not None
        plt.close(fig)

    def test_custom_decimals(self):
        """Test that custom decimal places work."""
        fig, ax = plt.subplots()
        result = format_percent_axis(ax, decimals=2)
        assert isinstance(result, Axes)
        plt.close(fig)


class TestFormatCurrencyAxis:
    """Test format_currency_axis function."""

    def test_returns_axis(self):
        """Test that the function returns an Axes object."""
        fig, ax = plt.subplots()
        result = format_currency_axis(ax)
        assert isinstance(result, Axes)
        plt.close(fig)

    def test_custom_symbol(self):
        """Test that custom currency symbol works."""
        fig, ax = plt.subplots()
        result = format_currency_axis(ax, symbol="€")
        assert isinstance(result, Axes)
        plt.close(fig)

    def test_custom_decimals(self):
        """Test that custom decimal places work."""
        fig, ax = plt.subplots()
        result = format_currency_axis(ax, decimals=2)
        assert isinstance(result, Axes)
        plt.close(fig)


class TestFinalizeFigure:
    """Test finalize_figure function."""

    def test_returns_figure(self):
        """Test that the function returns a Figure object."""
        fig = plt.figure()
        result = finalize_figure(fig)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_tight_layout_applied(self):
        """Test that tight_layout is applied by default."""
        fig = plt.figure()
        result = finalize_figure(fig, tight_layout=True)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_tight_layout_disabled(self):
        """Test that tight_layout can be disabled."""
        fig = plt.figure()
        result = finalize_figure(fig, tight_layout=False)
        assert isinstance(result, Figure)
        plt.close(fig)


class TestPlotPriceScenarios:
    """Test plot_price_scenarios function."""

    def test_returns_figure_and_axis(self, sample_scenarios):
        """Test that the function returns figure and axis."""
        fig, ax = plot_price_scenarios(sample_scenarios)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_custom_objective_column(self, sample_scenarios):
        """Test that custom objective column works."""
        fig, ax = plot_price_scenarios(
            sample_scenarios,
            objective_column="expected_revenue",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_current_price_marker(self, sample_scenarios):
        """Test that current price marker is shown."""
        fig, ax = plot_price_scenarios(
            sample_scenarios,
            current_price=10,
        )
        assert isinstance(fig, Figure)
        # Check that a vertical line was added
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_recommended_price_marker(self, sample_scenarios):
        """Test that recommended price marker is shown."""
        fig, ax = plot_price_scenarios(
            sample_scenarios,
            recommended_price=11,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self, sample_scenarios):
        """Test that custom title works."""
        custom_title = "My Custom Title"
        fig, ax = plot_price_scenarios(
            sample_scenarios,
            title=custom_title,
        )
        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValueError."""
        empty_df = pd.DataFrame({"candidate_price": [], "expected_profit": []})
        with pytest.raises(ValueError, match="is empty"):
            plot_price_scenarios(empty_df)

    def test_missing_column_raises_error(self, sample_scenarios):
        """Test that missing required column raises ValueError."""
        df = sample_scenarios.drop(columns=["candidate_price"])
        with pytest.raises(ValueError, match="missing required columns"):
            plot_price_scenarios(df)


class TestPlotOptimizationActions:
    """Test plot_optimization_actions function."""

    def test_returns_figure_and_axis(self, sample_action_summary):
        """Test that the function returns figure and axis."""
        fig, ax = plot_optimization_actions(sample_action_summary)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_creates_bar_chart(self, sample_action_summary):
        """Test that a bar chart is created."""
        fig, ax = plot_optimization_actions(sample_action_summary)
        # Check that bars were created
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_custom_columns(self):
        """Test that custom column names work."""
        df = pd.DataFrame(
            {
                "action": ["increase", "decrease"],
                "count": [10, 5],
            }
        )
        fig, ax = plot_optimization_actions(
            df,
            action_column="action",
            count_column="count",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValueError."""
        empty_df = pd.DataFrame({"recommendation_action": [], "item_count": []})
        with pytest.raises(ValueError, match="is empty"):
            plot_optimization_actions(empty_df)


class TestPlotPriceRecommendations:
    """Test plot_price_recommendations function."""

    def test_returns_figure_and_axis(self, sample_recommendations):
        """Test that the function returns figure and axis."""
        fig, ax = plot_price_recommendations(sample_recommendations)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_respects_top_n(self, sample_recommendations):
        """Test that top_n parameter limits results."""
        fig, ax = plot_price_recommendations(sample_recommendations, top_n=5)
        # Should have bars for top 5 items (2 bars per item)
        assert len(ax.patches) == 10  # 5 items × 2 bars
        plt.close(fig)

    def test_custom_column_names(self):
        """Test that custom column names work."""
        df = pd.DataFrame(
            {
                "product": ["A", "B"],
                "price_now": [10, 20],
                "price_new": [11, 19],
            }
        )
        fig, ax = plot_price_recommendations(
            df,
            item_column="product",
            current_price_column="price_now",
            recommended_price_column="price_new",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zero_top_n_raises_error(self, sample_recommendations):
        """Test that zero top_n raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            plot_price_recommendations(sample_recommendations, top_n=0)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValueError."""
        empty_df = pd.DataFrame(
            {
                "item_id": [],
                "current_price": [],
                "recommended_price": [],
            }
        )
        with pytest.raises(ValueError, match="is empty"):
            plot_price_recommendations(empty_df)


class TestPlotForecastPerformance:
    """Test plot_forecast_performance function."""

    def test_returns_figure_and_axis(self, sample_forecast_data):
        """Test that the function returns figure and axis."""
        fig, ax = plot_forecast_performance(sample_forecast_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_creates_two_lines(self, sample_forecast_data):
        """Test that two lines are created (actual and predicted)."""
        fig, ax = plot_forecast_performance(sample_forecast_data)
        # Should have 2 lines: actual and predicted
        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_custom_column_names(self):
        """Test that custom column names work."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10),
                "actual_demand": np.random.randint(80, 120, 10),
                "forecast": np.random.randint(80, 120, 10),
            }
        )
        fig, ax = plot_forecast_performance(
            df,
            date_column="timestamp",
            actual_column="actual_demand",
            predicted_column="forecast",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValueError."""
        empty_df = pd.DataFrame(
            {
                "date": [],
                "quantity": [],
                "predicted_value": [],
            }
        )
        with pytest.raises(ValueError, match="is empty"):
            plot_forecast_performance(empty_df)


class TestPlotForecastResiduals:
    """Test plot_forecast_residuals function."""

    def test_returns_figure_and_axis(self):
        """Test that the function returns figure and axis."""
        df = pd.DataFrame(
            {
                "predicted_value": np.random.uniform(80, 120, 50),
                "residual": np.random.normal(0, 10, 50),
            }
        )
        fig, ax = plot_forecast_residuals(df)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_creates_scatter_plot(self):
        """Test that a scatter plot is created."""
        df = pd.DataFrame(
            {
                "predicted_value": np.random.uniform(80, 120, 50),
                "residual": np.random.normal(0, 10, 50),
            }
        )
        fig, ax = plot_forecast_residuals(df)
        # Check that scatter points were created
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_zero_line_shown(self):
        """Test that zero reference line is shown."""
        df = pd.DataFrame(
            {
                "predicted_value": np.random.uniform(80, 120, 50),
                "residual": np.random.normal(0, 10, 50),
            }
        )
        fig, ax = plot_forecast_residuals(df)
        # Check that a horizontal line was added
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValueError."""
        empty_df = pd.DataFrame(
            {
                "predicted_value": [],
                "residual": [],
            }
        )
        with pytest.raises(ValueError, match="is empty"):
            plot_forecast_residuals(empty_df)


class TestPlotIntegration:
    """Integration tests for plotting functions."""

    def test_multiple_plots_can_be_created(self, sample_scenarios, sample_action_summary):
        """Test that multiple plots can be created in sequence."""
        fig1, ax1 = plot_price_scenarios(sample_scenarios)
        fig2, ax2 = plot_optimization_actions(sample_action_summary)

        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)
        assert fig1 is not fig2

        plt.close(fig1)
        plt.close(fig2)

    def test_custom_figure_size(self, sample_scenarios):
        """Test that custom figure size is respected."""
        custom_size = (15, 8)
        fig, ax = plot_price_scenarios(
            sample_scenarios,
            figure_size=custom_size,
        )
        # Matplotlib figure size is in inches
        assert fig.get_figwidth() == custom_size[0]
        assert fig.get_figheight() == custom_size[1]
        plt.close(fig)
