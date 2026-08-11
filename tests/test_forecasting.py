"""Unit tests for src/models/forecasting.py"""

import sys

import numpy as np
import pandas as pd
import pytest

from src.models.forecasting import (
    ForecastModelBundle,
    PreparedForecastData,
    calculate_forecast_metrics,
    compare_forecast_with_baseline,
    create_xgboost_regressor,
    fit_xgboost_forecast,
    get_feature_importance,
    predict_future_demand,
    prepare_forecasting_data,
    prepare_future_features,
    time_based_train_validation_split,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _build_demand_panel(n_days=160, products=("A", "B", "C"), seed=7):
    """Build a multi-product daily demand panel with trend + weekly seasonality.

    Rows are grouped by product (not interleaved chronologically), so the
    panel also exercises `prepare_forecasting_data`'s chronological sort.
    The target ("quantity") is strictly positive and driven by a linear
    trend, a weekly seasonal wave, a price effect, and a promo bump, so
    XGBoost has real signal to learn and forecast-quality assertions are
    meaningful rather than arbitrary.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    rows = []
    for product_index, product in enumerate(products):
        base_price = 8.0 + 2.0 * product_index
        for day_index, date in enumerate(dates):
            day_of_week = date.dayofweek
            promo_flag = int(day_index % 11 == 0)
            price = base_price + rng.normal(0, 0.2)

            trend = 100 + 0.4 * day_index
            seasonal = 15 * np.sin(2 * np.pi * day_of_week / 7)
            promo_effect = 25 if promo_flag else 0
            price_effect = -3 * (price - base_price)
            noise = rng.normal(0, 3)

            quantity = max(
                5.0,
                trend + seasonal + promo_effect + price_effect + noise,
            )

            rows.append(
                {
                    "date": date,
                    "product_id": product,
                    "day_index": day_index,
                    "day_of_week": day_of_week,
                    "price": price,
                    "promo_flag": promo_flag,
                    "quantity": quantity,
                }
            )

    return pd.DataFrame(rows)


def _prepared_from_frame(frame, *, date_column="date", target_column="target", feature_columns=("x",)):
    """Build a PreparedForecastData directly, bypassing prepare_forecasting_data."""
    return PreparedForecastData(
        dataframe=frame.reset_index(drop=True),
        target_column=target_column,
        date_column=date_column,
        feature_columns=list(feature_columns),
        numeric_feature_columns=list(feature_columns),
        categorical_columns=[],
    )


class _StubModel:
    def __init__(self, importances):
        self.feature_importances_ = np.asarray(importances, dtype=float)


def _make_importance_bundle(feature_columns, importances=None, model=None):
    prepared = PreparedForecastData(
        dataframe=pd.DataFrame(),
        target_column="target",
        date_column="date",
        feature_columns=list(feature_columns),
        numeric_feature_columns=list(feature_columns),
        categorical_columns=[],
    )
    if model is None:
        model = _StubModel(importances)
    return ForecastModelBundle(
        model=model,
        prepared_data=prepared,
        training_data=pd.DataFrame(),
        validation_data=pd.DataFrame(),
        validation_predictions=pd.DataFrame(),
        validation_metrics={},
        baseline_metrics={},
        residual_lower_quantile=0.0,
        residual_upper_quantile=0.0,
    )


def _make_metrics_bundle(validation_metrics, baseline_metrics):
    prepared = PreparedForecastData(
        dataframe=pd.DataFrame(),
        target_column="target",
        date_column="date",
        feature_columns=["x"],
        numeric_feature_columns=["x"],
        categorical_columns=[],
    )
    return ForecastModelBundle(
        model=None,
        prepared_data=prepared,
        training_data=pd.DataFrame(),
        validation_data=pd.DataFrame(),
        validation_predictions=pd.DataFrame(),
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        residual_lower_quantile=0.0,
        residual_upper_quantile=0.0,
    )


FEATURE_COLUMNS = ["day_index", "day_of_week", "price", "promo_flag"]


@pytest.fixture(scope="module")
def demand_panel():
    return _build_demand_panel()


@pytest.fixture(scope="module")
def prepared_panel(demand_panel):
    return prepare_forecasting_data(
        demand_panel,
        target_column="quantity",
        date_column="date",
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=["product_id"],
    )


@pytest.fixture(scope="module")
def fitted_bundle(demand_panel):
    return fit_xgboost_forecast(
        demand_panel,
        target_column="quantity",
        date_column="date",
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=["product_id"],
        number_of_estimators=40,
        learning_rate=0.2,
        maximum_depth=3,
    )


# ---------------------------------------------------------------------------
# prepare_forecasting_data
# ---------------------------------------------------------------------------


class TestPrepareForecastingData:
    """Test prepare_forecasting_data function."""

    def test_sorts_chronologically(self, demand_panel):
        """Rows are grouped by product in the source data, so this exercises the sort."""
        result = prepare_forecasting_data(
            demand_panel,
            target_column="quantity",
            date_column="date",
            feature_columns=FEATURE_COLUMNS,
            categorical_columns=["product_id"],
        )
        assert result.dataframe["date"].is_monotonic_increasing

    def test_categorical_dummy_columns_created(self, prepared_panel):
        """Test that categorical columns become drop-first dummy feature columns."""
        assert "product_id_B" in prepared_panel.feature_columns
        assert "product_id_C" in prepared_panel.feature_columns
        assert "product_id_A" not in prepared_panel.feature_columns
        assert "product_id" not in prepared_panel.feature_columns
        assert prepared_panel.categorical_columns == ["product_id"]

    def test_numeric_feature_columns_preserved(self, prepared_panel):
        """Test that the requested numeric feature columns are recorded."""
        assert prepared_panel.numeric_feature_columns == FEATURE_COLUMNS
        for column in FEATURE_COLUMNS:
            assert column in prepared_panel.feature_columns

    def test_no_features_raises(self):
        """Test that omitting both numeric and categorical features raises."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "y": range(5)})
        with pytest.raises(ValueError, match="At least one forecasting feature"):
            prepare_forecasting_data(
                df, target_column="y", date_column="date", feature_columns=[]
            )

    def test_overlapping_columns_raises(self):
        """Test that a column listed as both numeric and categorical raises."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "y": range(5),
                "flag": [0, 1, 0, 1, 0],
            }
        )
        with pytest.raises(ValueError, match="both numeric and"):
            prepare_forecasting_data(
                df,
                target_column="y",
                date_column="date",
                feature_columns=["flag"],
                categorical_columns=["flag"],
            )

    def test_missing_required_column_raises(self):
        """Test that a missing feature column raises ValueError."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "y": range(5)})
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_forecasting_data(
                df, target_column="y", date_column="date", feature_columns=["nonexistent"]
            )

    def test_drops_rows_with_unparseable_date_or_target(self):
        """Test that rows with an invalid date or target are dropped."""
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "not-a-date", "2023-01-04", "2023-01-05"],
                "y": [100, 101, 102, "oops", 104],
                "x": [1, 2, 3, 4, 5],
            }
        )
        result = prepare_forecasting_data(
            df, target_column="y", date_column="date", feature_columns=["x"]
        )
        assert len(result.dataframe) == 3

    def test_drops_constant_feature_column(self):
        """Test that a numeric feature with no variation is excluded."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=6),
                "y": [10, 11, 12, 13, 14, 15],
                "x": [1, 2, 3, 4, 5, 6],
                "constant": [7, 7, 7, 7, 7, 7],
            }
        )
        result = prepare_forecasting_data(
            df, target_column="y", date_column="date", feature_columns=["x", "constant"]
        )
        assert "constant" not in result.feature_columns
        assert "constant" not in result.dataframe.columns
        assert "x" in result.feature_columns

    def test_all_constant_features_raises(self):
        """Test that no remaining usable features raises ValueError."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=6),
                "y": [10, 11, 12, 13, 14, 15],
                "constant": [7, 7, 7, 7, 7, 7],
            }
        )
        with pytest.raises(ValueError, match="No usable forecasting features"):
            prepare_forecasting_data(
                df, target_column="y", date_column="date", feature_columns=["constant"]
            )

    def test_infinite_feature_values_become_missing(self):
        """Test that infinite values are treated as missing but the row is kept."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "y": [10, 11, 12, 13, 14],
                "x": [1.0, 2.0, np.inf, 4.0, 5.0],
            }
        )
        result = prepare_forecasting_data(
            df, target_column="y", date_column="date", feature_columns=["x"]
        )
        assert len(result.dataframe) == 5
        assert result.dataframe["x"].isna().sum() == 1


# ---------------------------------------------------------------------------
# time_based_train_validation_split
# ---------------------------------------------------------------------------


class TestTimeBasedTrainValidationSplit:
    """Test time_based_train_validation_split function."""

    def test_default_fraction_is_chronological(self, prepared_panel):
        """Test that training dates strictly precede validation dates."""
        training_data, validation_data = time_based_train_validation_split(prepared_panel)
        assert training_data["date"].max() < validation_data["date"].min()
        assert len(training_data) + len(validation_data) == len(prepared_panel.dataframe)

    def test_split_uses_date_values_not_row_order(self):
        """Regression test: shuffled row order must not leak future rows into training."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        frame = pd.DataFrame({"date": dates, "target": range(100), "x": range(100)})
        shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)

        prepared = _prepared_from_frame(shuffled, feature_columns=["x"])
        training_data, validation_data = time_based_train_validation_split(prepared)

        assert training_data["date"].max() < validation_data["date"].min()
        assert len(training_data) + len(validation_data) == 100
        # No row in training should have a date >= any row in validation.
        assert training_data["date"].max() < validation_data["date"].min()

    def test_explicit_validation_start_date_boundary(self):
        """Test that a row exactly on the cutoff date goes to validation."""
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        frame = pd.DataFrame({"date": dates, "target": range(90), "x": range(90)})
        prepared = _prepared_from_frame(frame, feature_columns=["x"])

        cutoff = dates[70]
        training_data, validation_data = time_based_train_validation_split(
            prepared, validation_start_date=cutoff
        )

        assert training_data["date"].max() == dates[69]
        assert validation_data["date"].min() == cutoff
        assert len(training_data) == 70
        assert len(validation_data) == 20

    def test_invalid_validation_fraction_raises(self, prepared_panel):
        """Test that a validation fraction outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="validation_fraction must be between"):
            time_based_train_validation_split(prepared_panel, validation_fraction=0)
        with pytest.raises(ValueError, match="validation_fraction must be between"):
            time_based_train_validation_split(prepared_panel, validation_fraction=1.5)

    def test_single_unique_date_raises(self):
        """Test that fewer than two distinct dates raises ValueError."""
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-01-01")] * 60,
                "target": range(60),
                "x": range(60),
            }
        )
        prepared = _prepared_from_frame(frame, feature_columns=["x"])
        with pytest.raises(ValueError, match="At least two distinct dates"):
            time_based_train_validation_split(prepared)

    def test_insufficient_training_observations_raises(self):
        """Test that too few training rows raises ValueError."""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        frame = pd.DataFrame({"date": dates, "target": range(60), "x": range(60)})
        prepared = _prepared_from_frame(frame, feature_columns=["x"])
        with pytest.raises(ValueError, match="Insufficient training observations"):
            time_based_train_validation_split(prepared, validation_start_date=dates[5])

    def test_insufficient_validation_observations_raises(self):
        """Test that too few validation rows raises ValueError."""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        frame = pd.DataFrame({"date": dates, "target": range(60), "x": range(60)})
        prepared = _prepared_from_frame(frame, feature_columns=["x"])
        with pytest.raises(ValueError, match="Insufficient validation observations"):
            time_based_train_validation_split(prepared, validation_start_date=dates[59])

    def test_indices_are_reset(self, prepared_panel):
        """Test that returned partitions have a fresh RangeIndex."""
        training_data, validation_data = time_based_train_validation_split(prepared_panel)
        assert list(training_data.index) == list(range(len(training_data)))
        assert list(validation_data.index) == list(range(len(validation_data)))


# ---------------------------------------------------------------------------
# calculate_forecast_metrics
# ---------------------------------------------------------------------------


class TestCalculateForecastMetrics:
    """Test calculate_forecast_metrics function."""

    def test_perfect_predictions(self):
        """Test that identical actual and predicted values give zero error."""
        actual = [10.0, 20.0, 30.0, 40.0]
        result = calculate_forecast_metrics(actual, actual)
        assert result["mae"] == 0
        assert result["rmse"] == 0
        assert result["mape_percent"] == 0
        assert result["wape_percent"] == 0
        assert result["r_squared"] == pytest.approx(1.0)
        assert result["observations"] == 4

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            calculate_forecast_metrics([1, 2, 3], [1, 2])

    def test_ignores_nan_and_inf_values(self):
        """Test that non-finite entries are excluded from evaluation."""
        actual = [1.0, 2.0, np.nan, 4.0]
        predicted = [1.0, 2.0, 3.0, np.inf]
        result = calculate_forecast_metrics(actual, predicted)
        assert result["observations"] == 2
        assert result["mae"] == 0
        assert result["r_squared"] == pytest.approx(1.0)

    def test_all_invalid_raises(self):
        """Test that no finite observations raises ValueError."""
        with pytest.raises(ValueError, match="No valid observations"):
            calculate_forecast_metrics([np.nan, np.inf], [np.inf, np.nan])

    def test_zero_actual_gives_nan_wape_and_mape(self):
        """Test that an all-zero actual series makes WAPE/MAPE undefined (NaN)."""
        result = calculate_forecast_metrics([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert np.isnan(result["wape_percent"])
        assert np.isnan(result["mape_percent"])
        assert result["mae"] == pytest.approx(2.0)
        assert result["smape_percent"] == pytest.approx(200.0)

    def test_single_observation_r_squared_is_nan(self):
        """Test that R^2 is undefined (NaN) for a single observation."""
        result = calculate_forecast_metrics([5.0], [4.0])
        assert np.isnan(result["r_squared"])
        assert result["mae"] == pytest.approx(1.0)

    def test_zero_mean_actual_normalized_mae_is_nan(self):
        """Test that normalized MAE is NaN when the mean actual value is zero."""
        result = calculate_forecast_metrics([-5.0, 5.0], [-5.0, 5.0])
        assert result["mae"] == 0
        assert np.isnan(result["normalized_mae"])

    def test_metrics_within_sane_range_for_noisy_signal(self):
        """Test error metrics stay in a sane range for a mildly noisy linear signal."""
        rng = np.random.default_rng(0)
        actual = np.linspace(100, 200, 50)
        predicted = actual + rng.normal(0, 5, 50)
        result = calculate_forecast_metrics(actual, predicted)
        assert result["observations"] == 50
        assert 0 <= result["mae"] < 15
        assert result["rmse"] >= result["mae"]
        assert result["r_squared"] > 0.8


# ---------------------------------------------------------------------------
# create_xgboost_regressor
# ---------------------------------------------------------------------------


class TestCreateXgboostRegressor:
    """Test create_xgboost_regressor function."""

    def test_returns_configured_regressor(self):
        """Test that hyperparameters are passed through to the regressor."""
        model = create_xgboost_regressor(
            number_of_estimators=50, learning_rate=0.1, maximum_depth=4, random_state=7
        )
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 4
        assert model.random_state == 7
        assert model.objective == "reg:squarederror"

    def test_non_positive_estimators_raises(self):
        """Test that a non-positive estimator count raises ValueError."""
        with pytest.raises(ValueError, match="number_of_estimators"):
            create_xgboost_regressor(number_of_estimators=0)

    def test_non_positive_learning_rate_raises(self):
        """Test that a non-positive learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            create_xgboost_regressor(learning_rate=0)

    def test_non_positive_max_depth_raises(self):
        """Test that a non-positive max depth raises ValueError."""
        with pytest.raises(ValueError, match="maximum_depth"):
            create_xgboost_regressor(maximum_depth=0)

    def test_missing_xgboost_raises_import_error(self, monkeypatch):
        """Test that an unavailable xgboost package raises a clear ImportError."""
        monkeypatch.setitem(sys.modules, "xgboost", None)
        with pytest.raises(ImportError, match="xgboost"):
            create_xgboost_regressor()


# ---------------------------------------------------------------------------
# fit_xgboost_forecast
# ---------------------------------------------------------------------------


class TestFitXgboostForecast:
    """Test fit_xgboost_forecast function."""

    def test_returns_expected_bundle_structure(self, fitted_bundle):
        """Test the fitted bundle exposes a usable model and matching feature columns."""
        assert hasattr(fitted_bundle.model, "predict")
        assert fitted_bundle.prepared_data.feature_columns
        assert list(fitted_bundle.validation_predictions.columns) == [
            "date",
            "quantity",
            "predicted_value",
            "residual",
            "absolute_error",
            "prediction_interval_lower",
            "prediction_interval_upper",
        ]

    def test_no_chronological_leakage(self, fitted_bundle):
        """Test that no validation-period date leaks into the training partition."""
        assert (
            fitted_bundle.training_data["date"].max()
            < fitted_bundle.validation_data["date"].min()
        )

    def test_validation_predictions_are_internally_consistent(self, fitted_bundle):
        """Test that residual and absolute_error are derived correctly."""
        predictions = fitted_bundle.validation_predictions
        expected_residual = predictions["quantity"] - predictions["predicted_value"]
        assert np.allclose(predictions["residual"], expected_residual)
        assert np.allclose(predictions["absolute_error"], expected_residual.abs())

    def test_metrics_are_finite(self, fitted_bundle):
        """Test that validation and baseline metrics are finite and non-negative."""
        for metrics in (fitted_bundle.validation_metrics, fitted_bundle.baseline_metrics):
            assert np.isfinite(metrics["mae"])
            assert np.isfinite(metrics["rmse"])
            assert metrics["mae"] >= 0
            assert metrics["rmse"] >= 0

    def test_beats_naive_mean_baseline(self, fitted_bundle):
        """Test that the trend-aware model outperforms a constant-mean baseline."""
        assert fitted_bundle.validation_metrics["mae"] < fitted_bundle.baseline_metrics["mae"]
        assert fitted_bundle.validation_metrics["rmse"] < fitted_bundle.baseline_metrics["rmse"]

    def test_custom_model_instance_is_used_and_fitted(self, demand_panel):
        """Test that a caller-provided model instance is fit and returned as-is."""
        custom_model = create_xgboost_regressor(number_of_estimators=10, maximum_depth=2)
        bundle = fit_xgboost_forecast(
            demand_panel,
            target_column="quantity",
            date_column="date",
            feature_columns=FEATURE_COLUMNS,
            categorical_columns=["product_id"],
            model=custom_model,
        )
        assert bundle.model is custom_model
        assert hasattr(bundle.model, "feature_importances_")

    def test_explicit_validation_start_date(self, demand_panel):
        """Test that an explicit validation_start_date is respected end-to-end."""
        cutoff = pd.Timestamp("2023-05-01")
        bundle = fit_xgboost_forecast(
            demand_panel,
            target_column="quantity",
            date_column="date",
            feature_columns=FEATURE_COLUMNS,
            categorical_columns=["product_id"],
            validation_start_date=cutoff,
            number_of_estimators=15,
            maximum_depth=2,
        )
        assert (bundle.validation_data["date"] >= cutoff).all()
        assert (bundle.training_data["date"] < cutoff).all()

    def test_insufficient_rows_raises(self):
        """Test that too little data raises rather than silently fitting a bad model."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tiny_df = pd.DataFrame(
            {
                "date": dates,
                "quantity": np.arange(30, dtype=float) + 10,
                "x": np.arange(30, dtype=float),
            }
        )
        with pytest.raises(ValueError, match="Insufficient training observations"):
            fit_xgboost_forecast(
                tiny_df, target_column="quantity", date_column="date", feature_columns=["x"]
            )

    def test_categorical_columns_recorded_on_bundle(self, fitted_bundle):
        """Test that categorical column metadata is retained on the prepared data."""
        assert fitted_bundle.prepared_data.categorical_columns == ["product_id"]
        assert any(
            column.startswith("product_id_")
            for column in fitted_bundle.prepared_data.feature_columns
        )


# ---------------------------------------------------------------------------
# prepare_future_features
# ---------------------------------------------------------------------------


class TestPrepareFutureFeatures:
    """Test prepare_future_features function."""

    def _future_rows(self, demand_panel, product="A", n=3):
        max_day_index = int(demand_panel["day_index"].max())
        rows = []
        for offset in range(1, n + 1):
            day_index = max_day_index + offset
            date = demand_panel["date"].max() + pd.Timedelta(days=offset)
            rows.append(
                {
                    "date": date,
                    "product_id": product,
                    "day_index": day_index,
                    "day_of_week": date.dayofweek,
                    "price": 8.0,
                    "promo_flag": 0,
                }
            )
        return pd.DataFrame(rows)

    def test_output_matches_training_feature_columns(self, demand_panel, fitted_bundle):
        """Test that the output columns exactly match the fitted feature columns."""
        future_df = self._future_rows(demand_panel)
        result = prepare_future_features(future_df, fitted_bundle)
        assert list(result.columns) == fitted_bundle.prepared_data.feature_columns

    def test_output_is_float_and_finite(self, demand_panel, fitted_bundle):
        """Test that all prepared future feature values are finite floats."""
        future_df = self._future_rows(demand_panel)
        result = prepare_future_features(future_df, fitted_bundle)
        assert (result.dtypes == float).all()
        assert np.isfinite(result.to_numpy()).all()

    def test_missing_required_column_raises(self, demand_panel, fitted_bundle):
        """Test that a missing feature column raises ValueError."""
        future_df = self._future_rows(demand_panel).drop(columns=["price"])
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_future_features(future_df, fitted_bundle)

    def test_unseen_categorical_value_defaults_to_zero_dummies(self, demand_panel, fitted_bundle):
        """Test that a never-seen category produces all-zero dummy columns."""
        future_df = self._future_rows(demand_panel, product="UNSEEN_PRODUCT")
        result = prepare_future_features(future_df, fitted_bundle)
        dummy_columns = [
            column
            for column in fitted_bundle.prepared_data.feature_columns
            if column.startswith("product_id_")
        ]
        assert (result[dummy_columns] == 0.0).all().all()

    def test_extra_columns_are_ignored(self, demand_panel, fitted_bundle):
        """Test that columns outside the trained feature set are dropped, not errored on."""
        future_df = self._future_rows(demand_panel)
        future_df["unused_extra_column"] = "irrelevant"
        result = prepare_future_features(future_df, fitted_bundle)
        assert "unused_extra_column" not in result.columns


# ---------------------------------------------------------------------------
# predict_future_demand
# ---------------------------------------------------------------------------


class TestPredictFutureDemand:
    """Test predict_future_demand function."""

    def _future_rows(self, demand_panel, product="A", n=3):
        max_day_index = int(demand_panel["day_index"].max())
        rows = []
        for offset in range(1, n + 1):
            day_index = max_day_index + offset
            date = demand_panel["date"].max() + pd.Timedelta(days=offset)
            rows.append(
                {
                    "date": date,
                    "product_id": product,
                    "day_index": day_index,
                    "day_of_week": date.dayofweek,
                    "price": 8.0,
                    "promo_flag": 0,
                }
            )
        return pd.DataFrame(rows)

    def test_adds_finite_predicted_value_column(self, demand_panel, fitted_bundle):
        """Test that predictions are added and are finite."""
        future_df = self._future_rows(demand_panel)
        result = predict_future_demand(future_df, fitted_bundle)
        assert "predicted_value" in result.columns
        assert np.isfinite(result["predicted_value"]).all()

    def test_predictions_positive_for_positive_demand_target(self, demand_panel, fitted_bundle):
        """Test that predictions stay positive for a strictly-positive demand target."""
        future_df = self._future_rows(demand_panel)
        result = predict_future_demand(future_df, fitted_bundle)
        assert (result["predicted_value"] > 0).all()

    def test_prediction_interval_columns_bracket_point_estimate(self, demand_panel, fitted_bundle):
        """Test that the upper interval bound is never below the lower bound."""
        future_df = self._future_rows(demand_panel)
        result = predict_future_demand(future_df, fitted_bundle, include_prediction_interval=True)
        assert "prediction_interval_lower" in result.columns
        assert "prediction_interval_upper" in result.columns
        assert (result["prediction_interval_upper"] >= result["prediction_interval_lower"]).all()

    def test_no_interval_columns_when_disabled(self, demand_panel, fitted_bundle):
        """Test that interval columns are omitted when not requested."""
        future_df = self._future_rows(demand_panel)
        result = predict_future_demand(future_df, fitted_bundle, include_prediction_interval=False)
        assert "prediction_interval_lower" not in result.columns
        assert "prediction_interval_upper" not in result.columns

    def test_preserves_original_columns(self, demand_panel, fitted_bundle):
        """Test that the input dataframe's own columns are retained in the result."""
        future_df = self._future_rows(demand_panel)
        result = predict_future_demand(future_df, fitted_bundle)
        for column in future_df.columns:
            assert column in result.columns
        assert list(result["product_id"]) == list(future_df["product_id"])


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    """Test get_feature_importance function."""

    def test_sorted_descending_with_sequential_ranks(self, fitted_bundle):
        """Test that importance is sorted descending with 1-based sequential ranks."""
        result = get_feature_importance(fitted_bundle)
        assert result["importance"].is_monotonic_decreasing
        assert list(result["importance_rank"]) == list(range(1, len(result) + 1))

    def test_importance_percent_sums_to_100(self):
        """Test that importance percentages sum to 100 and preserve rank order."""
        bundle = _make_importance_bundle(["a", "b", "c"], importances=[1, 2, 3])
        result = get_feature_importance(bundle)
        assert list(result["feature"]) == ["c", "b", "a"]
        assert result["importance_percent"].sum() == pytest.approx(100.0)
        assert result["importance_percent"].iloc[0] == pytest.approx(50.0)

    def test_zero_total_importance_gives_zero_percent(self):
        """Test that an all-zero importance model reports 0% rather than NaN."""
        bundle = _make_importance_bundle(["a", "b"], importances=[0.0, 0.0])
        result = get_feature_importance(bundle)
        assert (result["importance_percent"] == 0.0).all()

    def test_top_n_filters_rows(self, fitted_bundle):
        """Test that top_n limits the number of returned rows."""
        full_result = get_feature_importance(fitted_bundle)
        top_result = get_feature_importance(fitted_bundle, top_n=2)
        assert len(top_result) == 2
        assert list(top_result["feature"]) == list(full_result["feature"].iloc[:2])

    def test_non_positive_top_n_raises(self, fitted_bundle):
        """Test that top_n <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="top_n"):
            get_feature_importance(fitted_bundle, top_n=0)

    def test_model_without_feature_importances_raises_type_error(self):
        """Test that a model lacking feature_importances_ raises TypeError."""
        bundle = _make_importance_bundle(["a", "b"], model=object())
        with pytest.raises(TypeError, match="feature_importances_"):
            get_feature_importance(bundle)

    def test_mismatched_importance_length_raises_value_error(self):
        """Test that an importance array of the wrong length raises ValueError."""
        bundle = _make_importance_bundle(["a", "b", "c"], importances=[0.5, 0.5])
        with pytest.raises(ValueError, match="does not match"):
            get_feature_importance(bundle)


# ---------------------------------------------------------------------------
# compare_forecast_with_baseline
# ---------------------------------------------------------------------------


class TestCompareForecastWithBaseline:
    """Test compare_forecast_with_baseline function."""

    def test_returns_baseline_and_model_rows(self, fitted_bundle):
        """Test that the comparison has exactly a baseline row and an XGBoost row."""
        comparison = compare_forecast_with_baseline(fitted_bundle)
        assert list(comparison["model"]) == ["Mean Baseline", "XGBoost"]
        assert len(comparison) == 2

    def test_improvement_columns_match_manual_computation(self, fitted_bundle):
        """Test that improvement columns equal baseline metric minus each row's metric."""
        comparison = compare_forecast_with_baseline(fitted_bundle)
        baseline_mae = fitted_bundle.baseline_metrics["mae"]
        xgboost_row = comparison.loc[comparison["model"] == "XGBoost"].iloc[0]

        expected_improvement = baseline_mae - fitted_bundle.validation_metrics["mae"]
        assert xgboost_row["mae_improvement_vs_baseline"] == pytest.approx(expected_improvement)

        expected_percent = expected_improvement / baseline_mae * 100
        assert xgboost_row["mae_improvement_percent"] == pytest.approx(expected_percent)

    def test_zero_baseline_mae_gives_nan_improvement_percent(self):
        """Test that a zero baseline MAE makes the improvement percentage NaN."""
        bundle = _make_metrics_bundle(
            validation_metrics={"mae": 3.0, "rmse": 4.0},
            baseline_metrics={"mae": 0.0, "rmse": 0.0},
        )
        comparison = compare_forecast_with_baseline(bundle)
        assert comparison["mae_improvement_percent"].isna().all()
