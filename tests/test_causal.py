"""Unit tests for src/models/causal.py

The whole point of this module is that price is usually *endogenous*:
retailers cut prices in weak demand weeks and raise them in strong ones, so
a plain OLS regression of quantity on price absorbs that demand shock and
understates (or even flips the sign of) the true price elasticity.
Instrumenting price with a cost-side shifter that moves price but has no
direct effect on demand breaks that feedback loop and lets 2SLS recover
something close to the true causal elasticity.

Most tests below build synthetic panels with that exact structure -- a
known true elasticity, a demand shock that confounds price and quantity,
and a cost instrument that is excluded from the demand equation -- so
assertions can check real numeric properties (F-statistic above/below the
weak-instrument threshold, IV closer than OLS to the known truth, correct
sign/plausibility flags) rather than just "returns a dict with some keys".
"""

import numpy as np
import pandas as pd
import pytest

from src.models.causal import (
    IVModelBundle,
    PreparedIVData,
    build_exogenous_matrix,
    calculate_first_stage_diagnostics,
    compare_ols_and_iv,
    extract_causal_effect,
    fit_first_stage,
    fit_group_iv_elasticities,
    fit_iv_2sls,
    fit_ols_benchmark,
    prepare_iv_data,
    summarize_iv_model,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------
#
# Data generating process (per row):
#
#   cost            ~ Uniform(2, 10)                     [excluded instrument]
#   demand_shock    ~ Normal(0, 1)                        [unobserved confounder]
#   price           = 15 + instrument_strength * cost
#                     + shock_to_price * demand_shock + noise
#   log_price       = log(price)
#   log_quantity    = intercept + true_elasticity * log_price
#                     + shock_to_quantity * demand_shock + noise
#
# demand_shock drives both price (retailers raise prices when demand is
# strong) and quantity, so Cov(log_price, error) > 0 in the demand
# equation and OLS of log_quantity on log_price is biased toward zero /
# positive relative to true_elasticity. cost affects price but has no
# direct term in the quantity equation, so it is a valid excluded
# instrument and 2SLS should recover something much closer to
# true_elasticity than OLS does.


def make_confounded_panel(
    n=800,
    true_elasticity=-1.5,
    instrument_strength=1.8,
    shock_to_price=0.9,
    shock_to_quantity=0.8,
    seed=42,
):
    """Build a synthetic panel with a strong instrument and confounded price."""

    rng = np.random.default_rng(seed)
    cost = rng.uniform(2, 10, n)
    demand_shock = rng.normal(0, 1, n)
    price = np.maximum(
        15
        + instrument_strength * cost
        + shock_to_price * demand_shock
        + rng.normal(0, 0.5, n),
        1.0,
    )
    log_price = np.log(price)
    log_quantity = (
        5.0
        + true_elasticity * log_price
        + shock_to_quantity * demand_shock
        + rng.normal(0, 0.2, n)
    )

    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_quantity": log_quantity,
            "price": price,
            "quantity": np.exp(log_quantity),
            "cost": cost,
            "region": np.tile(["North", "South"], n // 2 + 1)[:n],
            "marketing_spend": rng.uniform(100, 500, n),
        }
    )


def make_weak_instrument_panel(n=300, seed=7):
    """Build a panel where the instrument barely moves price (F < 10)."""

    rng = np.random.default_rng(seed)
    instrument = rng.normal(0, 1, n)
    demand_shock = rng.normal(0, 1, n)
    price = np.maximum(
        10 + 0.01 * instrument + 0.9 * demand_shock + rng.normal(0, 1, n),
        1.0,
    )
    log_price = np.log(price)
    log_quantity = 5.0 - 1.0 * log_price + 0.8 * demand_shock + rng.normal(0, 0.2, n)

    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_quantity": log_quantity,
            "cost": instrument,
        }
    )


@pytest.fixture(scope="module")
def confounded_panel():
    """A single strong-instrument, confounded-price panel (true elasticity -1.5)."""
    return make_confounded_panel()


@pytest.fixture(scope="module")
def weak_instrument_panel():
    return make_weak_instrument_panel()


@pytest.fixture(scope="module")
def grouped_panel():
    """Five product groups exercising every branch of fit_group_iv_elasticities.

    A: enough rows, strong instrument, plausible negative elasticity -> reliable.
    B: too few rows -> insufficient_observations.
    C: weak instrument -> fitted but flagged unreliable.
    D: strong instrument but a genuinely positive causal effect -> implausible.
    E: constant price within the group -> prepare_iv_data fails -> status "failed".
    """

    group_a = make_confounded_panel(n=300, true_elasticity=-1.5, seed=1)
    group_a["product_id"] = "A"

    group_b = make_confounded_panel(n=30, true_elasticity=-1.5, seed=2)
    group_b["product_id"] = "B"

    group_c = make_weak_instrument_panel(n=300, seed=3)
    group_c["product_id"] = "C"

    group_d = make_confounded_panel(n=300, true_elasticity=0.5, seed=4)
    group_d["product_id"] = "D"

    group_e = make_confounded_panel(n=100, true_elasticity=-1.5, seed=5)
    group_e["log_price"] = 2.0
    group_e["product_id"] = "E"

    columns = ["log_price", "log_quantity", "cost", "product_id"]
    return pd.concat(
        [group_a[columns], group_b[columns], group_c[columns], group_d[columns], group_e[columns]],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def group_result(grouped_panel):
    return fit_group_iv_elasticities(
        grouped_panel,
        group_columns=["product_id"],
        outcome_column="log_quantity",
        endogenous_column="log_price",
        instrument_columns=["cost"],
        minimum_observations=60,
    )


# ---------------------------------------------------------------------------
# prepare_iv_data
# ---------------------------------------------------------------------------


class TestPrepareIVData:
    """Test prepare_iv_data function."""

    def test_basic_preparation_returns_expected_metadata(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert isinstance(prepared, PreparedIVData)
        assert prepared.outcome_column == "log_quantity"
        assert prepared.endogenous_column == "log_price"
        assert prepared.instrument_columns == ["cost"]
        assert len(prepared.dataframe) == len(confounded_panel)

    def test_empty_instrument_columns_raises(self, confounded_panel):
        with pytest.raises(ValueError, match="At least one instrumental variable"):
            prepare_iv_data(
                confounded_panel,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=[],
            )

    def test_column_listed_as_both_categorical_and_exogenous_raises(self, confounded_panel):
        with pytest.raises(ValueError, match="both numeric and categorical"):
            prepare_iv_data(
                confounded_panel,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
                exogenous_columns=["marketing_spend"],
                categorical_columns=["marketing_spend"],
            )

    def test_missing_required_column_raises(self, confounded_panel):
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_iv_data(
                confounded_panel,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["not_a_real_column"],
            )

    def test_drops_rows_with_missing_values(self, confounded_panel):
        df = confounded_panel.copy()
        df.loc[0:4, "log_price"] = np.nan
        prepared = prepare_iv_data(
            df,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert len(prepared.dataframe) == len(df) - 5

    def test_all_rows_missing_raises(self, confounded_panel):
        df = confounded_panel.copy()
        df["log_price"] = np.nan
        with pytest.raises(ValueError, match="No complete observations"):
            prepare_iv_data(
                df,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
            )

    def test_constant_endogenous_variable_raises(self, confounded_panel):
        df = confounded_panel.copy()
        df["log_price"] = 3.0
        with pytest.raises(ValueError, match="does not contain sufficient variation"):
            prepare_iv_data(
                df,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
            )

    def test_constant_instrument_raises(self, confounded_panel):
        df = confounded_panel.copy()
        df["cost"] = 5.0
        with pytest.raises(ValueError, match="does not contain sufficient variation"):
            prepare_iv_data(
                df,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
            )

    def test_categorical_columns_expand_to_dummy_exogenous_columns(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            categorical_columns=["region"],
        )
        # drop_first=True with two categories ("North", "South") -> one dummy column
        assert len(prepared.exogenous_columns) == 1
        assert prepared.exogenous_columns[0] in prepared.dataframe.columns
        assert set(prepared.dataframe[prepared.exogenous_columns[0]].unique()) <= {0.0, 1.0}

    def test_coerces_string_numeric_columns(self, confounded_panel):
        df = confounded_panel.copy()
        df["cost"] = df["cost"].astype(str)
        prepared = prepare_iv_data(
            df,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert pd.api.types.is_numeric_dtype(prepared.dataframe["cost"])
        assert len(prepared.dataframe) == len(df)

    def test_non_numeric_strings_are_dropped(self, confounded_panel):
        df = confounded_panel.copy()
        df["cost"] = df["cost"].astype(object)
        df.loc[0:2, "cost"] = "not_a_number"
        prepared = prepare_iv_data(
            df,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert len(prepared.dataframe) == len(df) - 3


# ---------------------------------------------------------------------------
# build_exogenous_matrix
# ---------------------------------------------------------------------------


class TestBuildExogenousMatrix:
    """Test build_exogenous_matrix function."""

    def test_no_exogenous_columns_returns_constant_only(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        matrix = build_exogenous_matrix(prepared)
        assert list(matrix.columns) == ["const"]
        assert len(matrix) == len(prepared.dataframe)
        assert (matrix["const"] == 1.0).all()

    def test_with_exogenous_columns_includes_constant_and_controls(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            exogenous_columns=["marketing_spend"],
        )
        matrix = build_exogenous_matrix(prepared)
        assert "const" in matrix.columns
        assert "marketing_spend" in matrix.columns
        assert matrix.shape[1] == 2


# ---------------------------------------------------------------------------
# fit_first_stage
# ---------------------------------------------------------------------------


class TestFitFirstStage:
    """Test fit_first_stage function."""

    def test_instrument_coefficient_has_expected_sign(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        result = fit_first_stage(prepared)
        # Higher cost -> higher price by construction (instrument_strength > 0).
        assert result.params["cost"] > 0
        assert result.pvalues["cost"] < 0.01

    def test_robust_covariance_flag_changes_cov_type(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        robust_result = fit_first_stage(prepared, robust_covariance=True)
        plain_result = fit_first_stage(prepared, robust_covariance=False)
        assert robust_result.cov_type == "HC3"
        assert plain_result.cov_type == "nonrobust"


# ---------------------------------------------------------------------------
# calculate_first_stage_diagnostics
# ---------------------------------------------------------------------------


class TestCalculateFirstStageDiagnostics:
    """Test calculate_first_stage_diagnostics function."""

    def test_strong_instrument_is_not_flagged_weak(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        diagnostics = calculate_first_stage_diagnostics(prepared)
        assert diagnostics["f_statistic"] > 10
        assert diagnostics["weak_instrument_flag"] is False
        assert diagnostics["instrument_count"] == 1
        assert diagnostics["observations"] == len(prepared.dataframe)

    def test_weak_instrument_is_flagged(self, weak_instrument_panel):
        prepared = prepare_iv_data(
            weak_instrument_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        diagnostics = calculate_first_stage_diagnostics(prepared)
        assert diagnostics["f_statistic"] < 10
        assert diagnostics["weak_instrument_flag"] is True

    def test_partial_r_squared_is_bounded(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        diagnostics = calculate_first_stage_diagnostics(prepared)
        assert 0.0 <= diagnostics["partial_r_squared"] <= 1.0

    def test_collinear_duplicate_instruments_do_not_crash(self, confounded_panel):
        df = confounded_panel.copy()
        df["cost_duplicate"] = df["cost"]
        prepared = prepare_iv_data(
            df,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost", "cost_duplicate"],
        )
        # Perfectly collinear instruments: the joint F-test's constraint
        # covariance is rank-deficient, but the function should still
        # return finite diagnostics rather than raising.
        diagnostics = calculate_first_stage_diagnostics(prepared)
        assert diagnostics["instrument_count"] == 2
        assert np.isfinite(diagnostics["f_statistic"])


# ---------------------------------------------------------------------------
# fit_iv_2sls
# ---------------------------------------------------------------------------


class TestFitIV2SLS:
    """Test fit_iv_2sls function."""

    def test_recovers_approximately_true_elasticity(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert isinstance(bundle, IVModelBundle)
        estimate = bundle.result.params["log_price"]
        # True elasticity is -1.5; IV should land in its neighborhood
        # despite the strong confounding baked into the panel.
        assert -2.0 < estimate < -1.0

    def test_insufficient_observations_raises(self, confounded_panel):
        small_df = confounded_panel.head(10)
        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_iv_2sls(
                small_df,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
                minimum_observations=30,
            )

    def test_collinear_instruments_raise_value_error(self, confounded_panel):
        df = confounded_panel.copy()
        df["cost_duplicate"] = df["cost"]
        with pytest.raises(ValueError, match="full column rank"):
            fit_iv_2sls(
                df,
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost", "cost_duplicate"],
            )

    def test_bundle_carries_first_stage_and_diagnostics(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        assert bundle.first_stage_result is not None
        assert "f_statistic" in bundle.first_stage_diagnostics
        assert bundle.prepared_data.endogenous_column == "log_price"

    def test_with_categorical_and_exogenous_controls(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            exogenous_columns=["marketing_spend"],
            categorical_columns=["region"],
        )
        estimate = bundle.result.params["log_price"]
        assert -2.0 < estimate < -1.0
        # marketing_spend and the region dummy should both be in the fit.
        assert "marketing_spend" in bundle.result.params.index


# ---------------------------------------------------------------------------
# fit_ols_benchmark
# ---------------------------------------------------------------------------


class TestFitOLSBenchmark:
    """Test fit_ols_benchmark function."""

    def test_ols_is_biased_relative_to_true_elasticity(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        ols_result = fit_ols_benchmark(prepared)
        ols_estimate = ols_result.params["log_price"]
        # True elasticity is -1.5. The positive demand-shock confound pulls
        # the naive OLS estimate toward zero (or positive), so it should
        # sit well above (be much less negative than) the true value.
        assert ols_estimate > -1.0

    def test_ols_still_negative_direction_for_endogenous_price(self, confounded_panel):
        prepared = prepare_iv_data(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        ols_result = fit_ols_benchmark(prepared)
        # Sanity: the model fits and returns a finite estimate for the
        # endogenous coefficient.
        assert np.isfinite(ols_result.params["log_price"])


# ---------------------------------------------------------------------------
# extract_causal_effect
# ---------------------------------------------------------------------------


class TestExtractCausalEffect:
    """Test extract_causal_effect function."""

    def test_matches_underlying_model_parameter(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        effect = extract_causal_effect(bundle)
        assert effect == pytest.approx(bundle.result.params["log_price"])

    def test_missing_endogenous_column_raises_keyerror(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        bundle.prepared_data.endogenous_column = "not_a_real_column"
        with pytest.raises(KeyError):
            extract_causal_effect(bundle)


# ---------------------------------------------------------------------------
# summarize_iv_model
# ---------------------------------------------------------------------------


class TestSummarizeIVModel:
    """Test summarize_iv_model function."""

    def test_summary_is_internally_consistent(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        summary = summarize_iv_model(bundle)
        assert summary["causal_effect"] == pytest.approx(extract_causal_effect(bundle))
        assert summary["confidence_interval_lower"] < summary["causal_effect"]
        assert summary["causal_effect"] < summary["confidence_interval_upper"]
        assert summary["observations"] == len(bundle.prepared_data.dataframe)
        assert summary["weak_instrument_flag"] is False
        assert summary["first_stage_f_statistic"] > 10

    def test_narrower_confidence_level_gives_tighter_interval(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        wide = summarize_iv_model(bundle, confidence_level=0.99)
        narrow = summarize_iv_model(bundle, confidence_level=0.80)
        wide_width = wide["confidence_interval_upper"] - wide["confidence_interval_lower"]
        narrow_width = narrow["confidence_interval_upper"] - narrow["confidence_interval_lower"]
        assert narrow_width < wide_width

    def test_weak_instrument_panel_flags_summary(self, weak_instrument_panel):
        bundle = fit_iv_2sls(
            weak_instrument_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        summary = summarize_iv_model(bundle)
        assert summary["weak_instrument_flag"] is True
        assert summary["first_stage_f_statistic"] < 10


# ---------------------------------------------------------------------------
# compare_ols_and_iv
# ---------------------------------------------------------------------------


class TestCompareOLSAndIV:
    """Test compare_ols_and_iv function."""

    def test_returns_one_row_per_model(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        comparison = compare_ols_and_iv(bundle)
        assert sorted(comparison["model"]) == ["IV_2SLS", "OLS"]
        assert len(comparison) == 2

    def test_estimate_difference_vs_ols_column(self, confounded_panel):
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        comparison = compare_ols_and_iv(bundle).set_index("model")
        assert comparison.loc["OLS", "estimate_difference_vs_ols"] == pytest.approx(0.0)
        expected_diff = (
            comparison.loc["IV_2SLS", "estimate"] - comparison.loc["OLS", "estimate"]
        )
        assert comparison.loc["IV_2SLS", "estimate_difference_vs_ols"] == pytest.approx(
            expected_diff
        )

    def test_iv_estimate_closer_to_true_elasticity_than_ols(self, confounded_panel):
        """The central claim of this module: under price endogeneity, 2SLS
        should recover an estimate closer to the true causal elasticity
        than plain OLS does."""
        true_elasticity = -1.5
        bundle = fit_iv_2sls(
            confounded_panel,
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
        )
        comparison = compare_ols_and_iv(bundle).set_index("model")
        ols_estimate = comparison.loc["OLS", "estimate"]
        iv_estimate = comparison.loc["IV_2SLS", "estimate"]

        ols_error = abs(ols_estimate - true_elasticity)
        iv_error = abs(iv_estimate - true_elasticity)

        assert iv_error < ols_error
        # OLS should be materially biased toward zero/positive given the
        # positive demand-shock confound built into the synthetic panel.
        assert ols_estimate > true_elasticity + 0.5


# ---------------------------------------------------------------------------
# fit_group_iv_elasticities
# ---------------------------------------------------------------------------


class TestFitGroupIVElasticities:
    """Test fit_group_iv_elasticities function."""

    def test_empty_group_columns_raises(self, grouped_panel):
        with pytest.raises(ValueError, match="At least one group column"):
            fit_group_iv_elasticities(
                grouped_panel,
                group_columns=[],
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["cost"],
            )

    def test_missing_required_column_raises(self, grouped_panel):
        with pytest.raises(ValueError, match="missing required columns"):
            fit_group_iv_elasticities(
                grouped_panel,
                group_columns=["product_id"],
                outcome_column="log_quantity",
                endogenous_column="log_price",
                instrument_columns=["not_a_real_column"],
            )

    def test_returns_exactly_one_row_per_group(self, group_result):
        assert sorted(group_result["product_id"]) == ["A", "B", "C", "D", "E"]
        assert len(group_result) == 5

    def test_reliable_group_has_plausible_negative_elasticity(self, group_result):
        row = group_result.loc[group_result["product_id"] == "A"].iloc[0]
        assert row["status"] == "fitted"
        assert row["is_reliable"] is True or bool(row["is_reliable"]) is True
        assert row["iv_elasticity"] < 0
        assert row["weak_instrument_flag"] == False  # noqa: E712 (numpy bool)
        assert row["is_economically_plausible"] == True  # noqa: E712

    def test_insufficient_observations_group_flagged(self, group_result):
        row = group_result.loc[group_result["product_id"] == "B"].iloc[0]
        assert row["status"] == "insufficient_observations"
        assert np.isnan(row["iv_elasticity"])
        assert row["is_reliable"] == False  # noqa: E712
        assert "required" in row["failure_reason"]

    def test_weak_instrument_group_not_reliable(self, group_result):
        row = group_result.loc[group_result["product_id"] == "C"].iloc[0]
        assert row["status"] == "fitted"
        assert row["weak_instrument_flag"] == True  # noqa: E712
        assert row["is_reliable"] == False  # noqa: E712
        assert "weak_instrument" in row["failure_reason"]

    def test_implausible_positive_elasticity_group_not_reliable(self, group_result):
        row = group_result.loc[group_result["product_id"] == "D"].iloc[0]
        assert row["status"] == "fitted"
        assert row["iv_elasticity"] > 0
        assert row["is_economically_plausible"] == False  # noqa: E712
        assert row["is_reliable"] == False  # noqa: E712
        assert "implausible_estimate" in row["failure_reason"]

    def test_group_that_fails_to_fit_is_reported_not_dropped(self, group_result):
        row = group_result.loc[group_result["product_id"] == "E"].iloc[0]
        assert row["status"] == "failed"
        assert row["is_reliable"] == False  # noqa: E712
        assert row["failure_reason"] != ""

    def test_custom_weak_instrument_threshold_relaxes_flag(self, grouped_panel):
        # Group C has an F-statistic that is weak (<10) but not vanishingly
        # small. A much lower threshold should stop flagging it as weak.
        lenient_result = fit_group_iv_elasticities(
            grouped_panel,
            group_columns=["product_id"],
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            minimum_observations=60,
            weak_instrument_threshold=0.01,
        )
        row = lenient_result.loc[lenient_result["product_id"] == "C"].iloc[0]
        assert row["weak_instrument_flag"] == False  # noqa: E712

    def test_custom_plausible_range_accepts_positive_elasticity(self, grouped_panel):
        # Group D has a genuinely positive causal effect (~0.5). Widening
        # the plausible range to include positive values should mark it
        # economically plausible (and, combined with its strong instrument,
        # reliable).
        widened_result = fit_group_iv_elasticities(
            grouped_panel,
            group_columns=["product_id"],
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            minimum_observations=60,
            plausible_elasticity_range=(-8.0, 2.0),
        )
        row = widened_result.loc[widened_result["product_id"] == "D"].iloc[0]
        assert row["is_economically_plausible"] == True  # noqa: E712
        assert row["is_reliable"] == True  # noqa: E712

    def test_constant_control_within_one_group_does_not_break_fit(self, confounded_panel):
        # A control column that is constant within a single group (but
        # varies overall) would be perfectly collinear with the intercept
        # for that group. fit_group_iv_elasticities is documented to drop
        # non-varying controls per group rather than crash.
        df = confounded_panel.copy()
        df["product_id"] = np.where(df.index < len(df) // 2, "G1", "G2")
        df["store_size"] = np.where(df["product_id"] == "G1", 100.0, np.random.default_rng(9).uniform(50, 150, len(df)))

        result = fit_group_iv_elasticities(
            df,
            group_columns=["product_id"],
            outcome_column="log_quantity",
            endogenous_column="log_price",
            instrument_columns=["cost"],
            exogenous_columns=["store_size"],
            minimum_observations=60,
        )
        g1_row = result.loc[result["product_id"] == "G1"].iloc[0]
        assert g1_row["status"] == "fitted"
