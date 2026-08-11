"""Unit tests for src/models/promotion.py

Covers the promotion-uplift core: propensity-score / IPW estimation and
T-learner causal uplift estimation, plus the guardrails this module relies
on to gate whether promotion evidence is allowed to influence pricing
(common-support trimming, leakage-free feature preparation, and ranking
quality of the resulting uplift predictions).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.models.promotion import (
    PreparedPromotionData,
    PromotionUpliftBundle,
    calculate_ipw_weights,
    calculate_propensity_scores,
    estimate_ipw_promotion_effect,
    estimate_naive_promotion_uplift,
    evaluate_uplift_by_decile,
    fit_promotion_t_learner,
    fit_propensity_model,
    normalize_promotion_indicator,
    predict_promotion_uplift,
    prepare_promotion_data,
    rank_promotion_opportunities,
    summarize_promotion_predictions,
    trim_common_support,
)


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _make_confounded_promotion_dataframe(
    n=400,
    true_uplift=6.0,
    noise_sd=1.5,
    random_state=7,
    heterogeneous=False,
):
    """Build data where a confounder `x` drives both promotion assignment
    and the outcome, with a known additive promotion effect.

    outcome = 20 + 3*x + effect*treatment + noise
    treatment ~ Bernoulli(sigmoid(0.5*(x - 5)))   (higher x -> more likely promoted)

    Because x also raises the outcome directly, a naive treated-vs-control
    mean comparison is biased upward; propensity/IPW correction should pull
    the estimate back toward the true effect.
    """
    rng = np.random.default_rng(random_state)

    x = rng.uniform(0, 10, n)
    logit = 0.5 * (x - 5)
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity)

    if heterogeneous:
        effect = 1.0 + 0.8 * x
    else:
        effect = true_uplift

    noise = rng.normal(0, noise_sd, n)
    outcome = 20 + 3 * x + effect * treatment + noise

    return pd.DataFrame({"x": x, "treatment": treatment, "outcome": outcome})


def _prepared(df, feature_columns=("x",), outcome_column="outcome", treatment_column="treatment"):
    return prepare_promotion_data(
        df,
        outcome_column=outcome_column,
        treatment_column=treatment_column,
        feature_columns=list(feature_columns),
    )


class _StubModelMissingTreatedClass:
    """Propensity-model stand-in that never saw class 1 during fitting."""

    classes_ = np.array([0])

    def predict_proba(self, predictors):
        return np.ones((len(predictors), 1))


# ---------------------------------------------------------------------------
# normalize_promotion_indicator
# ---------------------------------------------------------------------------


class TestNormalizePromotionIndicator:
    """Test normalize_promotion_indicator function."""

    def test_boolean_series(self):
        result = normalize_promotion_indicator(pd.Series([True, False, True]))
        assert result.tolist() == [1, 0, 1]
        assert result.dtype == "int8"

    def test_numeric_series_positive_is_one(self):
        result = normalize_promotion_indicator(pd.Series([0, 1, 2, -1]))
        assert result.tolist() == [0, 1, 1, 0]

    def test_numeric_series_with_nan_treated_as_zero(self):
        result = normalize_promotion_indicator(pd.Series([1.0, np.nan, 0.0]))
        assert result.tolist() == [1, 0, 0]

    def test_string_positive_keywords(self):
        values = pd.Series(["Yes", "PROMO", " promoted ", "Treatment", "no"])
        result = normalize_promotion_indicator(values)
        assert result.tolist() == [1, 1, 1, 1, 0]

    def test_string_unrecognized_values_are_zero(self):
        values = pd.Series(["banana", "maybe", ""])
        result = normalize_promotion_indicator(values)
        assert result.tolist() == [0, 0, 0]

    def test_string_series_with_missing_values(self):
        values = pd.Series(["yes", None, "active"])
        result = normalize_promotion_indicator(values)
        assert result.tolist() == [1, 0, 1]

    def test_output_is_int8(self):
        result = normalize_promotion_indicator(pd.Series(["yes", "no"]))
        assert result.dtype == "int8"


# ---------------------------------------------------------------------------
# prepare_promotion_data
# ---------------------------------------------------------------------------


class TestPreparePromotionData:
    """Test prepare_promotion_data function."""

    def _basic_dataframe(self):
        return pd.DataFrame(
            {
                "outcome": [10.0, 12.0, 9.0, 15.0, 11.0, 14.0],
                "treatment": [1, 0, 0, 1, 0, 1],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

    def test_basic_preparation(self):
        result = _prepared(self._basic_dataframe())
        assert isinstance(result, PreparedPromotionData)
        assert result.feature_columns == ["x"]
        assert len(result.dataframe) == 6
        assert set(result.dataframe[result.treatment_column].unique()) == {0, 1}

    def test_requires_at_least_one_feature(self):
        with pytest.raises(ValueError, match="At least one"):
            prepare_promotion_data(
                self._basic_dataframe(),
                outcome_column="outcome",
                treatment_column="treatment",
                feature_columns=[],
            )

    def test_overlapping_numeric_and_categorical_columns_raise(self):
        with pytest.raises(ValueError, match="both numeric and"):
            prepare_promotion_data(
                self._basic_dataframe(),
                outcome_column="outcome",
                treatment_column="treatment",
                feature_columns=["x"],
                categorical_columns=["x"],
            )

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            prepare_promotion_data(
                self._basic_dataframe(),
                outcome_column="outcome",
                treatment_column="treatment",
                feature_columns=["does_not_exist"],
            )

    def test_requires_both_treatment_groups(self):
        df = self._basic_dataframe()
        df["treatment"] = 1
        with pytest.raises(ValueError, match="both treated and untreated"):
            _prepared(df)

    def test_constant_feature_column_is_dropped(self):
        df = self._basic_dataframe()
        df["constant_feature"] = 5.0
        result = _prepared(df, feature_columns=("x", "constant_feature"))
        assert "constant_feature" not in result.feature_columns
        assert "x" in result.feature_columns

    def test_all_features_constant_raises(self):
        df = self._basic_dataframe()
        df["x"] = 1.0
        with pytest.raises(ValueError, match="No usable modelling features"):
            _prepared(df)

    def test_no_complete_observations_raises(self):
        df = self._basic_dataframe()
        df["x"] = np.nan
        with pytest.raises(ValueError, match="No complete observations"):
            _prepared(df)

    def test_categorical_columns_generate_dummies(self):
        df = self._basic_dataframe()
        df["region"] = ["North", "South", "North", "South", "North", "South"]
        result = prepare_promotion_data(
            df,
            outcome_column="outcome",
            treatment_column="treatment",
            feature_columns=["x"],
            categorical_columns=["region"],
        )
        dummy_columns = [c for c in result.feature_columns if c.startswith("region_")]
        assert len(dummy_columns) == 1  # drop_first=True with 2 categories
        assert "x" in result.feature_columns

    def test_leakage_columns_not_required(self):
        """Post-treatment fields (price, inventory, stockout) are not part
        of the module's contract -- callers choose feature_columns, and a
        minimal confounder-only feature set must work standalone."""
        df = self._basic_dataframe()
        assert "selling_price" not in df.columns
        assert "inventory_level" not in df.columns
        assert "stockout_flag" not in df.columns
        result = _prepared(df)
        assert result.feature_columns == ["x"]


# ---------------------------------------------------------------------------
# estimate_naive_promotion_uplift
# ---------------------------------------------------------------------------


class TestEstimateNaivePromotionUplift:
    """Test estimate_naive_promotion_uplift function."""

    def test_basic_uplift_calculation(self):
        df = pd.DataFrame(
            {
                "outcome": [10, 10, 10, 20, 20, 20],
                "treatment": [0, 0, 0, 1, 1, 1],
            }
        )
        result = estimate_naive_promotion_uplift(
            df, outcome_column="outcome", treatment_column="treatment"
        )
        assert result["treated_mean_outcome"] == 20
        assert result["control_mean_outcome"] == 10
        assert result["absolute_uplift"] == 10
        assert result["relative_uplift"] == 1.0
        assert result["treated_observations"] == 3
        assert result["control_observations"] == 3

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"outcome": [1, 2], "treatment": [0, 1]})
        with pytest.raises(ValueError):
            estimate_naive_promotion_uplift(
                df, outcome_column="missing", treatment_column="treatment"
            )

    def test_requires_both_groups(self):
        df = pd.DataFrame({"outcome": [10, 12, 9], "treatment": [1, 1, 1]})
        with pytest.raises(ValueError, match="Both treated and untreated"):
            estimate_naive_promotion_uplift(
                df, outcome_column="outcome", treatment_column="treatment"
            )

    def test_zero_control_mean_gives_nan_relative_uplift(self):
        df = pd.DataFrame(
            {"outcome": [0, 0, 5, 5], "treatment": [0, 0, 1, 1]}
        )
        result = estimate_naive_promotion_uplift(
            df, outcome_column="outcome", treatment_column="treatment"
        )
        assert np.isnan(result["relative_uplift"])

    def test_naive_estimate_is_biased_under_confounding(self):
        """Regression guardrail: the naive estimator does NOT correct for
        a confounder that drives both assignment and outcome, so it should
        land well away from the true uplift used to simulate the data."""
        df = _make_confounded_promotion_dataframe(n=400, true_uplift=6.0, random_state=3)
        result = estimate_naive_promotion_uplift(
            df, outcome_column="outcome", treatment_column="treatment"
        )
        assert result["absolute_uplift"] > 6.0 + 2.0  # inflated by confounding


# ---------------------------------------------------------------------------
# fit_propensity_model
# ---------------------------------------------------------------------------


class TestFitPropensityModel:
    """Test fit_propensity_model function."""

    def test_non_positive_regularization_raises(self):
        df = _make_confounded_promotion_dataframe(n=60, random_state=1)
        prepared = _prepared(df)
        with pytest.raises(ValueError, match="greater than zero"):
            fit_propensity_model(prepared, regularization_strength=0)

    def test_returns_fitted_logistic_regression_with_both_classes(self):
        df = _make_confounded_promotion_dataframe(n=200, random_state=1)
        prepared = _prepared(df)
        model = fit_propensity_model(prepared)
        assert isinstance(model, LogisticRegression)
        assert set(model.classes_.tolist()) == {0, 1}

    def test_recovers_direction_of_confounding(self):
        """x drives promotion assignment upward (higher x -> more likely
        promoted), so the fitted coefficient on x should be positive."""
        df = _make_confounded_promotion_dataframe(n=300, random_state=1)
        prepared = _prepared(df)
        model = fit_propensity_model(prepared)
        x_index = prepared.feature_columns.index("x")
        assert model.coef_[0][x_index] > 0


# ---------------------------------------------------------------------------
# calculate_propensity_scores
# ---------------------------------------------------------------------------


class TestCalculatePropensityScores:
    """Test calculate_propensity_scores function."""

    def _prepared_and_model(self, n=200, random_state=1):
        df = _make_confounded_promotion_dataframe(n=n, random_state=random_state)
        prepared = _prepared(df)
        model = fit_propensity_model(prepared)
        return prepared, model

    def test_scores_within_default_clip_bounds(self):
        prepared, model = self._prepared_and_model()
        scores = calculate_propensity_scores(prepared, model)
        assert (scores >= 0.01).all()
        assert (scores <= 0.99).all()
        assert len(scores) == len(prepared.dataframe)

    def test_custom_clip_bounds_respected(self):
        prepared, model = self._prepared_and_model()
        scores = calculate_propensity_scores(
            prepared, model, minimum_score=0.1, maximum_score=0.9
        )
        assert (scores >= 0.1).all()
        assert (scores <= 0.9).all()

    def test_invalid_bounds_raise(self):
        prepared, model = self._prepared_and_model()
        with pytest.raises(ValueError, match="0 < minimum_score < maximum_score < 1"):
            calculate_propensity_scores(prepared, model, minimum_score=0.9, maximum_score=0.1)

    def test_missing_treated_class_raises(self):
        prepared, _ = self._prepared_and_model()
        with pytest.raises(ValueError, match="does not contain treatment class 1"):
            calculate_propensity_scores(prepared, _StubModelMissingTreatedClass())


# ---------------------------------------------------------------------------
# trim_common_support
# ---------------------------------------------------------------------------


class TestTrimCommonSupport:
    """Test trim_common_support function."""

    def _prepared_with_scores(self, treated_scores, control_scores):
        n_treated = len(treated_scores)
        n_control = len(control_scores)
        treatment = pd.Series([1] * n_treated + [0] * n_control)
        propensity_scores = pd.Series(
            np.concatenate([treated_scores, control_scores])
        )
        dataframe = pd.DataFrame(
            {
                "treatment": treatment,
                "outcome": np.zeros(n_treated + n_control),
                "x": np.zeros(n_treated + n_control),
            }
        )
        prepared = PreparedPromotionData(
            dataframe=dataframe,
            outcome_column="outcome",
            treatment_column="treatment",
            feature_columns=["x"],
        )
        return prepared, propensity_scores

    def test_invalid_quantiles_raise(self):
        prepared, scores = self._prepared_with_scores(
            np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
        )
        with pytest.raises(ValueError, match="lower_quantile < upper_quantile"):
            trim_common_support(prepared, scores, lower_quantile=0.9, upper_quantile=0.1)

    def test_no_overlap_raises(self):
        """Guardrail: propensity distributions with essentially no shared
        support must be rejected rather than silently modelled."""
        prepared, scores = self._prepared_with_scores(
            np.linspace(0.80, 0.99, 50), np.linspace(0.01, 0.20, 50)
        )
        with pytest.raises(ValueError, match="No meaningful common propensity-score support"):
            trim_common_support(prepared, scores)

    def test_poor_overlap_strictly_shrinks_sample(self):
        """Guardrail: when treated/control propensity ranges only partly
        overlap, trimming must actually drop the non-overlapping rows."""
        prepared, scores = self._prepared_with_scores(
            np.linspace(0.30, 0.95, 100), np.linspace(0.05, 0.70, 100)
        )
        original_length = len(prepared.dataframe)
        trimmed_data, trimmed_scores = trim_common_support(prepared, scores)
        assert 0 < len(trimmed_data.dataframe) < original_length
        assert len(trimmed_scores) == len(trimmed_data.dataframe)

    def test_good_overlap_retains_most_of_sample(self):
        prepared, scores = self._prepared_with_scores(
            np.linspace(0.10, 0.90, 100), np.linspace(0.10, 0.90, 100)
        )
        original_length = len(prepared.dataframe)
        trimmed_data, _ = trim_common_support(prepared, scores)
        assert len(trimmed_data.dataframe) >= 0.9 * original_length

    def test_trimmed_scores_within_bounds(self):
        prepared, scores = self._prepared_with_scores(
            np.linspace(0.30, 0.95, 100), np.linspace(0.05, 0.70, 100)
        )
        trimmed_data, trimmed_scores = trim_common_support(prepared, scores)
        treated_mask = trimmed_data.dataframe["treatment"] == 1
        control_mask = trimmed_data.dataframe["treatment"] == 0
        # Overlapping region should contain both groups after trimming.
        assert treated_mask.any()
        assert control_mask.any()


# ---------------------------------------------------------------------------
# calculate_ipw_weights
# ---------------------------------------------------------------------------


class TestCalculateIpwWeights:
    """Test calculate_ipw_weights function."""

    def test_stabilized_weights_match_manual_formula(self):
        treatment = pd.Series([1, 1, 0, 0])
        propensity = pd.Series([0.8, 0.6, 0.3, 0.4])
        weights = calculate_ipw_weights(treatment, propensity, stabilized=True)

        treatment_probability = 0.5
        expected = [
            treatment_probability / 0.8,
            treatment_probability / 0.6,
            (1 - treatment_probability) / (1 - 0.3),
            (1 - treatment_probability) / (1 - 0.4),
        ]
        np.testing.assert_allclose(weights.to_numpy(), expected)

    def test_unstabilized_weights_match_manual_formula(self):
        treatment = pd.Series([1, 1, 0, 0])
        propensity = pd.Series([0.8, 0.6, 0.3, 0.4])
        weights = calculate_ipw_weights(treatment, propensity, stabilized=False)

        expected = [1 / 0.8, 1 / 0.6, 1 / (1 - 0.3), 1 / (1 - 0.4)]
        np.testing.assert_allclose(weights.to_numpy(), expected)

    def test_weights_are_positive(self):
        treatment = pd.Series([1, 0, 1, 0, 1])
        propensity = pd.Series([0.2, 0.9, 0.5, 0.1, 0.99])
        weights = calculate_ipw_weights(treatment, propensity)
        assert (weights > 0).all()

    def test_propensity_is_clipped_before_weighting(self):
        treatment = pd.Series([1, 0])
        propensity = pd.Series([1.5, -0.5])  # out of [0, 1], must be clipped
        weights = calculate_ipw_weights(treatment, propensity, stabilized=False)
        assert np.isfinite(weights).all()


# ---------------------------------------------------------------------------
# estimate_ipw_promotion_effect
# ---------------------------------------------------------------------------


class TestEstimateIpwPromotionEffect:
    """Test estimate_ipw_promotion_effect function."""

    def test_corrects_confounding_better_than_naive(self):
        """Core guardrail: IPW re-weighting on the true confounder should
        land closer to the known simulated uplift than the naive
        difference-in-means, which is biased by the same confounder."""
        true_uplift = 6.0
        df = _make_confounded_promotion_dataframe(
            n=500, true_uplift=true_uplift, random_state=11
        )
        prepared = _prepared(df)
        propensity_model = fit_propensity_model(prepared)
        propensity_scores = calculate_propensity_scores(prepared, propensity_model)

        ipw_result = estimate_ipw_promotion_effect(prepared, propensity_scores)
        naive_result = estimate_naive_promotion_uplift(
            df, outcome_column="outcome", treatment_column="treatment"
        )

        ipw_error = abs(ipw_result["average_treatment_effect"] - true_uplift)
        naive_error = abs(naive_result["absolute_uplift"] - true_uplift)

        assert ipw_error < naive_error
        assert ipw_error < 3.0

    def test_result_structure_and_bounds(self):
        df = _make_confounded_promotion_dataframe(n=200, random_state=2)
        prepared = _prepared(df)
        propensity_model = fit_propensity_model(prepared)
        propensity_scores = calculate_propensity_scores(prepared, propensity_model)
        result = estimate_ipw_promotion_effect(prepared, propensity_scores)

        assert result["observations"] == len(prepared.dataframe)
        assert 0 <= result["minimum_propensity_score"] <= result["maximum_propensity_score"] <= 1
        assert result["mean_ipw_weight"] > 0
        assert result["maximum_ipw_weight"] >= result["mean_ipw_weight"]

    def test_unstabilized_weights_option_runs(self):
        df = _make_confounded_promotion_dataframe(n=150, random_state=4)
        prepared = _prepared(df)
        propensity_model = fit_propensity_model(prepared)
        propensity_scores = calculate_propensity_scores(prepared, propensity_model)
        result = estimate_ipw_promotion_effect(
            prepared, propensity_scores, stabilized_weights=False
        )
        assert np.isfinite(result["average_treatment_effect"])


# ---------------------------------------------------------------------------
# fit_promotion_t_learner
# ---------------------------------------------------------------------------


class TestFitPromotionTLearner:
    """Test fit_promotion_t_learner function."""

    def test_insufficient_treated_observations_raises(self):
        n_control, n_treated = 25, 5
        df = pd.DataFrame(
            {
                "outcome": np.concatenate(
                    [np.random.default_rng(0).normal(10, 1, n_control),
                     np.random.default_rng(1).normal(15, 1, n_treated)]
                ),
                "treatment": [0] * n_control + [1] * n_treated,
                "x": np.random.default_rng(2).uniform(0, 10, n_control + n_treated),
            }
        )
        prepared = _prepared(df)
        with pytest.raises(ValueError, match="Insufficient promoted observations"):
            fit_promotion_t_learner(prepared)

    def test_insufficient_control_observations_raises(self):
        n_control, n_treated = 5, 25
        df = pd.DataFrame(
            {
                "outcome": np.concatenate(
                    [np.random.default_rng(0).normal(10, 1, n_control),
                     np.random.default_rng(1).normal(15, 1, n_treated)]
                ),
                "treatment": [0] * n_control + [1] * n_treated,
                "x": np.random.default_rng(2).uniform(0, 10, n_control + n_treated),
            }
        )
        prepared = _prepared(df)
        with pytest.raises(ValueError, match="Insufficient non-promoted observations"):
            fit_promotion_t_learner(prepared)

    def test_recovers_true_uplift_sign_and_magnitude(self):
        """With a linear DGP and a linear outcome model, the T-learner's
        predicted uplift should land close to the simulated true effect."""
        true_uplift = 6.0
        df = _make_confounded_promotion_dataframe(
            n=400, true_uplift=true_uplift, random_state=9
        )
        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared, outcome_model=LinearRegression())
        predictions = predict_promotion_uplift(bundle)

        average_predicted_uplift = predictions["predicted_uplift"].mean()
        assert average_predicted_uplift > 0
        assert abs(average_predicted_uplift - true_uplift) < 2.0

    def test_negative_true_uplift_recovered_with_correct_sign(self):
        true_uplift = -5.0
        df = _make_confounded_promotion_dataframe(
            n=400, true_uplift=true_uplift, random_state=13
        )
        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared, outcome_model=LinearRegression())
        predictions = predict_promotion_uplift(bundle)

        average_predicted_uplift = predictions["predicted_uplift"].mean()
        assert average_predicted_uplift < 0
        assert abs(average_predicted_uplift - true_uplift) < 2.0

    def test_default_outcome_model_runs_end_to_end(self):
        """Smoke test with the real default HistGradientBoostingRegressor
        (kept small so the suite stays fast)."""
        df = _make_confounded_promotion_dataframe(n=60, random_state=5)
        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared)
        assert isinstance(bundle, PromotionUpliftBundle)
        predictions = predict_promotion_uplift(bundle)
        assert np.isfinite(predictions["predicted_uplift"]).all()

    def test_bundle_carries_prepared_data(self):
        df = _make_confounded_promotion_dataframe(n=60, random_state=5)
        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared, outcome_model=LinearRegression())
        assert bundle.prepared_data is prepared
        assert isinstance(bundle.propensity_model, LogisticRegression)


# ---------------------------------------------------------------------------
# predict_promotion_uplift
# ---------------------------------------------------------------------------


class TestPredictPromotionUplift:
    """Test predict_promotion_uplift function."""

    def _bundle(self, n=300, true_uplift=4.0, random_state=6):
        df = _make_confounded_promotion_dataframe(
            n=n, true_uplift=true_uplift, random_state=random_state
        )
        prepared = _prepared(df)
        return fit_promotion_t_learner(prepared, outcome_model=LinearRegression())

    def test_predicted_uplift_equals_difference_of_group_predictions(self):
        bundle = self._bundle()
        predictions = predict_promotion_uplift(bundle)
        expected = (
            predictions["predicted_promoted_outcome"]
            - predictions["predicted_control_outcome"]
        )
        np.testing.assert_allclose(
            predictions["predicted_uplift"].to_numpy(), expected.to_numpy()
        )

    def test_propensity_score_column_in_unit_interval(self):
        bundle = self._bundle()
        predictions = predict_promotion_uplift(bundle)
        assert (predictions["propensity_score"] >= 0).all()
        assert (predictions["propensity_score"] <= 1).all()

    def test_recommended_promotion_matches_uplift_sign(self):
        bundle = self._bundle()
        predictions = predict_promotion_uplift(bundle)
        expected_recommendation = (predictions["predicted_uplift"] > 0).astype("int8")
        pd.testing.assert_series_equal(
            predictions["recommended_promotion"],
            expected_recommendation,
            check_names=False,
        )

    def test_row_count_matches_prepared_data(self):
        bundle = self._bundle()
        predictions = predict_promotion_uplift(bundle)
        assert len(predictions) == len(bundle.prepared_data.dataframe)


# ---------------------------------------------------------------------------
# summarize_promotion_predictions
# ---------------------------------------------------------------------------


class TestSummarizePromotionPredictions:
    """Test summarize_promotion_predictions function."""

    def _predictions(self):
        return pd.DataFrame(
            {
                "predicted_promoted_outcome": [12.0, 14.0, 9.0, 20.0],
                "predicted_control_outcome": [10.0, 15.0, 9.5, 16.0],
                "predicted_uplift": [2.0, -1.0, -0.5, 4.0],
                "recommended_promotion": [1, 0, 0, 1],
            }
        )

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            summarize_promotion_predictions(pd.DataFrame({"predicted_uplift": [1, 2]}))

    def test_aggregates_match_manual_computation(self):
        predictions = self._predictions()
        result = summarize_promotion_predictions(predictions)

        assert result["observations"] == 4
        assert result["average_predicted_uplift"] == pytest.approx(1.125)
        assert result["median_predicted_uplift"] == pytest.approx(0.75)
        assert result["positive_uplift_observations"] == 2
        assert result["positive_uplift_rate"] == pytest.approx(0.5)
        assert result["recommended_promotions"] == 2
        assert result["average_predicted_promoted_outcome"] == pytest.approx(13.75)
        assert result["average_predicted_control_outcome"] == pytest.approx(12.625)


# ---------------------------------------------------------------------------
# rank_promotion_opportunities
# ---------------------------------------------------------------------------


class TestRankPromotionOpportunities:
    """Test rank_promotion_opportunities function."""

    def _predictions(self):
        return pd.DataFrame(
            {
                "item_id": ["A", "B", "C", "D", "E"],
                "predicted_uplift": [1.0, 5.0, -2.0, 3.0, 0.0],
            }
        )

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            rank_promotion_opportunities(pd.DataFrame({"foo": [1, 2]}))

    def test_sorted_descending_by_predicted_uplift(self):
        result = rank_promotion_opportunities(self._predictions())
        assert result["predicted_uplift"].is_monotonic_decreasing

    def test_priority_rank_starts_at_one_and_increments(self):
        result = rank_promotion_opportunities(self._predictions())
        assert result["promotion_priority_rank"].tolist() == list(
            range(1, len(result) + 1)
        )

    def test_minimum_uplift_filters_rows(self):
        result = rank_promotion_opportunities(self._predictions(), minimum_uplift=1.0)
        assert (result["predicted_uplift"] >= 1.0).all()
        assert len(result) == 3  # 1.0, 5.0, 3.0

    def test_top_n_truncates_results(self):
        result = rank_promotion_opportunities(self._predictions(), top_n=2)
        assert len(result) == 2
        assert result["item_id"].tolist() == ["B", "D"]  # highest two uplifts

    def test_top_n_non_positive_raises(self):
        with pytest.raises(ValueError, match="greater than zero"):
            rank_promotion_opportunities(self._predictions(), top_n=0)


# ---------------------------------------------------------------------------
# evaluate_uplift_by_decile
# ---------------------------------------------------------------------------


class TestEvaluateUpliftByDecile:
    """Test evaluate_uplift_by_decile function."""

    def _deterministic_predictions(self, n=100):
        """predicted_uplift strictly increasing with row index; treatment
        alternates so every decile has both treated and control rows."""
        return pd.DataFrame(
            {
                "outcome": np.arange(n, dtype=float),
                "treatment": [i % 2 for i in range(n)],
                "predicted_uplift": np.arange(n, dtype=float),
            }
        )

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            evaluate_uplift_by_decile(
                pd.DataFrame({"predicted_uplift": [1, 2]}),
                outcome_column="outcome",
                treatment_column="treatment",
            )

    def test_number_of_groups_below_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            evaluate_uplift_by_decile(
                self._deterministic_predictions(),
                outcome_column="outcome",
                treatment_column="treatment",
                number_of_groups=1,
            )

    def test_group_one_holds_highest_predicted_uplift(self):
        """Ranking-quality guardrail: uplift_group 1 must correspond to the
        top decile of predicted uplift, and average_predicted_uplift must
        be monotonically non-increasing as uplift_group increases."""
        result = evaluate_uplift_by_decile(
            self._deterministic_predictions(),
            outcome_column="outcome",
            treatment_column="treatment",
            number_of_groups=10,
        )
        assert result["uplift_group"].tolist() == list(range(1, 11))
        average_uplift_by_group = result.sort_values("uplift_group")[
            "average_predicted_uplift"
        ].to_numpy()
        assert np.all(np.diff(average_uplift_by_group) <= 0)

    def test_top_group_shows_higher_observed_uplift_than_bottom_group(self):
        """A heterogeneous, monotonically increasing true effect should
        surface as a higher observed uplift in the top predicted-uplift
        decile than in the bottom one."""
        df = _make_confounded_promotion_dataframe(
            n=600, heterogeneous=True, random_state=21
        )
        # Random (unconfounded) treatment makes the T-learner comparison
        # cleaner for this ranking-quality check. A distinct seed is used
        # here (rather than reusing 21) because two numpy Generators seeded
        # identically can produce spuriously correlated draws.
        rng = np.random.default_rng(2021)
        df["treatment"] = rng.binomial(1, 0.5, len(df))
        df["outcome"] = 20 + 3 * df["x"] + (1.0 + 0.8 * df["x"]) * df["treatment"]
        df["outcome"] += rng.normal(0, 1.5, len(df))

        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared, outcome_model=LinearRegression())
        predictions = predict_promotion_uplift(bundle)

        decile_summary = evaluate_uplift_by_decile(
            predictions,
            outcome_column=prepared.outcome_column,
            treatment_column=prepared.treatment_column,
            number_of_groups=10,
        )

        top_group = decile_summary.loc[decile_summary["uplift_group"] == 1].iloc[0]
        bottom_group = decile_summary.loc[decile_summary["uplift_group"] == 10].iloc[0]
        assert top_group["observed_uplift"] > bottom_group["observed_uplift"]

    def test_group_with_only_treated_rows_has_nan_observed_uplift(self):
        n = 20
        df = pd.DataFrame(
            {
                "outcome": np.arange(n, dtype=float),
                # First 10 rows (lowest predicted uplift) all treated,
                # so that decile has no control rows to compare against.
                "treatment": [1] * 10 + [0, 1] * 5,
                "predicted_uplift": np.arange(n, dtype=float),
            }
        )
        result = evaluate_uplift_by_decile(
            df,
            outcome_column="outcome",
            treatment_column="treatment",
            number_of_groups=10,
        )
        bottom_group = result.loc[result["uplift_group"] == 10].iloc[0]
        assert np.isnan(bottom_group["observed_uplift"])

    def test_group_observations_sum_to_total_rows(self):
        predictions = self._deterministic_predictions(n=97)  # not evenly divisible
        result = evaluate_uplift_by_decile(
            predictions,
            outcome_column="outcome",
            treatment_column="treatment",
            number_of_groups=10,
        )
        assert result["observations"].sum() == 97


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestPromotionDataclasses:
    """Test PreparedPromotionData / PromotionUpliftBundle containers."""

    def test_prepared_promotion_data_is_slotted(self):
        prepared = PreparedPromotionData(
            dataframe=pd.DataFrame({"a": [1]}),
            outcome_column="a",
            treatment_column="a",
            feature_columns=["a"],
        )
        with pytest.raises(AttributeError):
            prepared.unexpected_attribute = 1  # slots=True forbids this

    def test_promotion_uplift_bundle_holds_all_fitted_components(self):
        df = _make_confounded_promotion_dataframe(n=60, random_state=5)
        prepared = _prepared(df)
        bundle = fit_promotion_t_learner(prepared, outcome_model=LinearRegression())
        assert bundle.prepared_data is prepared
        assert bundle.treated_model is not None
        assert bundle.control_model is not None
        assert bundle.propensity_model is not None
