"""Instrumental-variable and two-stage least-squares utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.validation import validate_required_columns


@dataclass(slots=True)
class PreparedIVData:
    """Prepared data and column metadata for an IV model."""

    dataframe: pd.DataFrame
    outcome_column: str
    endogenous_column: str
    instrument_columns: list[str]
    exogenous_columns: list[str]


@dataclass(slots=True)
class IVModelBundle:
    """Store the IV result, first-stage model, and diagnostics."""

    result: Any
    first_stage_result: Any
    prepared_data: PreparedIVData
    first_stage_diagnostics: dict[str, float | int | bool]


def prepare_iv_data(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    endogenous_column: str,
    instrument_columns: Sequence[str],
    exogenous_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> PreparedIVData:
    """Prepare complete numeric observations for IV estimation."""

    instrument_columns = list(instrument_columns)
    exogenous_columns = list(exogenous_columns or [])
    categorical_columns = list(categorical_columns or [])

    if not instrument_columns:
        raise ValueError(
            "At least one instrumental variable is required."
        )

    overlapping_columns = (
        set(categorical_columns)
        & set(exogenous_columns)
    )

    if overlapping_columns:
        raise ValueError(
            "Columns cannot be listed as both numeric and categorical "
            f"controls: {sorted(overlapping_columns)}"
        )

    required_columns = [
        outcome_column,
        endogenous_column,
        *instrument_columns,
        *exogenous_columns,
        *categorical_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="IV data",
    )

    numeric_columns = [
        outcome_column,
        endogenous_column,
        *instrument_columns,
        *exogenous_columns,
    ]

    prepared_dataframe = (
        dataframe[numeric_columns]
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    generated_control_columns: list[str] = []

    if categorical_columns:
        categorical_features = pd.get_dummies(
            dataframe[categorical_columns].astype("string"),
            columns=categorical_columns,
            prefix=categorical_columns,
            drop_first=True,
            dtype=float,
        )

        generated_control_columns = list(
            categorical_features.columns
        )

        prepared_dataframe = pd.concat(
            [
                prepared_dataframe,
                categorical_features,
            ],
            axis=1,
        )

    prepared_dataframe = (
        prepared_dataframe
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )

    if prepared_dataframe.empty:
        raise ValueError(
            "No complete observations remain after preparing IV data."
        )

    if (
        prepared_dataframe[endogenous_column]
        .nunique(dropna=True)
        < 2
    ):
        raise ValueError(
            f"Endogenous variable '{endogenous_column}' "
            "does not contain sufficient variation."
        )

    for instrument_column in instrument_columns:
        if (
            prepared_dataframe[instrument_column]
            .nunique(dropna=True)
            < 2
        ):
            raise ValueError(
                f"Instrument '{instrument_column}' "
                "does not contain sufficient variation."
            )

    final_exogenous_columns = [
        *exogenous_columns,
        *generated_control_columns,
    ]

    return PreparedIVData(
        dataframe=prepared_dataframe,
        outcome_column=outcome_column,
        endogenous_column=endogenous_column,
        instrument_columns=instrument_columns,
        exogenous_columns=final_exogenous_columns,
    )


def build_exogenous_matrix(
    prepared_data: PreparedIVData,
) -> pd.DataFrame:
    """Create the exogenous matrix with an intercept."""

    dataframe = prepared_data.dataframe

    if prepared_data.exogenous_columns:
        exogenous_matrix = dataframe[
            prepared_data.exogenous_columns
        ].astype(float)
    else:
        exogenous_matrix = pd.DataFrame(
            index=dataframe.index
        )

    return sm.add_constant(
        exogenous_matrix,
        has_constant="add",
    ).astype(float)


def fit_first_stage(
    prepared_data: PreparedIVData,
    *,
    robust_covariance: bool = True,
) -> Any:
    """Regress the endogenous variable on instruments and controls."""

    dataframe = prepared_data.dataframe

    first_stage_columns = [
        *prepared_data.exogenous_columns,
        *prepared_data.instrument_columns,
    ]

    first_stage_matrix = dataframe[
        first_stage_columns
    ].astype(float)

    first_stage_matrix = sm.add_constant(
        first_stage_matrix,
        has_constant="add",
    )

    target = dataframe[
        prepared_data.endogenous_column
    ].astype(float)

    model = sm.OLS(
        target,
        first_stage_matrix,
    )

    if robust_covariance:
        return model.fit(cov_type="HC3")

    return model.fit()


def calculate_first_stage_diagnostics(
    prepared_data: PreparedIVData,
) -> dict[str, float | int | bool]:
    """Calculate instrument relevance and weak-instrument diagnostics."""

    dataframe = prepared_data.dataframe

    unrestricted_columns = [
        *prepared_data.exogenous_columns,
        *prepared_data.instrument_columns,
    ]

    unrestricted_matrix = sm.add_constant(
        dataframe[unrestricted_columns].astype(float),
        has_constant="add",
    )

    if prepared_data.exogenous_columns:
        restricted_matrix = sm.add_constant(
            dataframe[
                prepared_data.exogenous_columns
            ].astype(float),
            has_constant="add",
        )
    else:
        restricted_matrix = pd.DataFrame(
            {
                "const": np.ones(
                    len(dataframe),
                    dtype=float,
                )
            },
            index=dataframe.index,
        )

    target = dataframe[
        prepared_data.endogenous_column
    ].astype(float)

    unrestricted_model = sm.OLS(
        target,
        unrestricted_matrix,
    ).fit()

    restricted_model = sm.OLS(
        target,
        restricted_matrix,
    ).fit()

    unrestricted_ssr = float(
        unrestricted_model.ssr
    )

    restricted_ssr = float(
        restricted_model.ssr
    )

    instrument_count = len(
        prepared_data.instrument_columns
    )

    residual_degrees_of_freedom = int(
        unrestricted_model.df_resid
    )

    if (
        unrestricted_ssr <= 0
        or residual_degrees_of_freedom <= 0
    ):
        f_statistic = np.nan
    else:
        numerator = (
            restricted_ssr
            - unrestricted_ssr
        ) / instrument_count

        denominator = (
            unrestricted_ssr
            / residual_degrees_of_freedom
        )

        f_statistic = float(
            numerator / denominator
        )

    if restricted_ssr <= 0:
        partial_r_squared = np.nan
    else:
        partial_r_squared = float(
            (
                restricted_ssr
                - unrestricted_ssr
            )
            / restricted_ssr
        )

    instrument_indices = [
        unrestricted_model.model.exog_names.index(
            instrument_column
        )
        for instrument_column
        in prepared_data.instrument_columns
    ]

    restriction_matrix = np.zeros(
        (
            instrument_count,
            len(
                unrestricted_model.model.exog_names
            ),
        )
    )

    for row_index, column_index in enumerate(
        instrument_indices
    ):
        restriction_matrix[
            row_index,
            column_index,
        ] = 1.0

    joint_test = unrestricted_model.f_test(
        restriction_matrix
    )

    joint_p_value = float(
        np.asarray(joint_test.pvalue).item()
    )

    weak_instrument_flag = bool(
        np.isfinite(f_statistic)
        and f_statistic < 10
    )

    return {
        "observations": int(
            unrestricted_model.nobs
        ),
        "instrument_count": instrument_count,
        "first_stage_r_squared": float(
            unrestricted_model.rsquared
        ),
        "first_stage_adjusted_r_squared": float(
            unrestricted_model.rsquared_adj
        ),
        "partial_r_squared": partial_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": joint_p_value,
        "weak_instrument_flag": (
            weak_instrument_flag
        ),
    }


def fit_iv_2sls(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    endogenous_column: str,
    instrument_columns: Sequence[str],
    exogenous_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    minimum_observations: int = 30,
    robust_covariance: bool = True,
) -> IVModelBundle:
    """Estimate a causal effect using instrumental-variable 2SLS."""

    try:
        from linearmodels.iv import IV2SLS
    except ImportError as error:
        raise ImportError(
            "The 'linearmodels' package is required. "
            "Install it with: python -m pip install linearmodels"
        ) from error

    prepared_data = prepare_iv_data(
        dataframe,
        outcome_column=outcome_column,
        endogenous_column=endogenous_column,
        instrument_columns=instrument_columns,
        exogenous_columns=exogenous_columns,
        categorical_columns=categorical_columns,
    )

    model_dataframe = prepared_data.dataframe

    if len(model_dataframe) < minimum_observations:
        raise ValueError(
            "Insufficient observations for IV estimation. "
            f"Required: {minimum_observations:,}; "
            f"available: {len(model_dataframe):,}."
        )

    dependent_variable = model_dataframe[
        outcome_column
    ].astype(float)

    exogenous_matrix = build_exogenous_matrix(
        prepared_data
    )

    endogenous_matrix = model_dataframe[
        [endogenous_column]
    ].astype(float)

    instrument_matrix = model_dataframe[
        prepared_data.instrument_columns
    ].astype(float)

    model = IV2SLS(
        dependent=dependent_variable,
        exog=exogenous_matrix,
        endog=endogenous_matrix,
        instruments=instrument_matrix,
    )

    covariance_type = (
        "robust"
        if robust_covariance
        else "unadjusted"
    )

    fitted_model = model.fit(
        cov_type=covariance_type,
        debiased=True,
    )

    first_stage_result = fit_first_stage(
        prepared_data,
        robust_covariance=robust_covariance,
    )

    diagnostics = (
        calculate_first_stage_diagnostics(
            prepared_data
        )
    )

    return IVModelBundle(
        result=fitted_model,
        first_stage_result=first_stage_result,
        prepared_data=prepared_data,
        first_stage_diagnostics=diagnostics,
    )


def fit_ols_benchmark(
    prepared_data: PreparedIVData,
    *,
    robust_covariance: bool = True,
) -> Any:
    """Estimate the corresponding OLS model for comparison."""

    dataframe = prepared_data.dataframe

    predictor_columns = [
        prepared_data.endogenous_column,
        *prepared_data.exogenous_columns,
    ]

    predictor_matrix = sm.add_constant(
        dataframe[predictor_columns].astype(float),
        has_constant="add",
    )

    target = dataframe[
        prepared_data.outcome_column
    ].astype(float)

    model = sm.OLS(
        target,
        predictor_matrix,
    )

    if robust_covariance:
        return model.fit(cov_type="HC3")

    return model.fit()


def extract_causal_effect(
    model_bundle: IVModelBundle,
) -> float:
    """Extract the estimated coefficient of the endogenous variable."""

    endogenous_column = (
        model_bundle
        .prepared_data
        .endogenous_column
    )

    if endogenous_column not in (
        model_bundle.result.params.index
    ):
        raise KeyError(
            "The fitted IV model does not contain "
            f"'{endogenous_column}'."
        )

    return float(
        model_bundle.result.params[
            endogenous_column
        ]
    )


def summarize_iv_model(
    model_bundle: IVModelBundle,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int | bool]:
    """Create a compact statistical summary of a fitted IV model."""

    fitted_model = model_bundle.result

    endogenous_column = (
        model_bundle
        .prepared_data
        .endogenous_column
    )

    confidence_interval = fitted_model.conf_int(
        level=confidence_level
    ).loc[endogenous_column]

    diagnostics = (
        model_bundle.first_stage_diagnostics
    )

    return {
        "causal_effect": float(
            fitted_model.params[
                endogenous_column
            ]
        ),
        "standard_error": float(
            fitted_model.std_errors[
                endogenous_column
            ]
        ),
        "p_value": float(
            fitted_model.pvalues[
                endogenous_column
            ]
        ),
        "confidence_level": float(
            confidence_level
        ),
        "confidence_interval_lower": float(
            confidence_interval.iloc[0]
        ),
        "confidence_interval_upper": float(
            confidence_interval.iloc[1]
        ),
        "r_squared": float(
            fitted_model.rsquared
        ),
        "adjusted_r_squared": float(
            fitted_model.rsquared_adj
        ),
        "observations": int(
            fitted_model.nobs
        ),
        "first_stage_f_statistic": float(
            diagnostics["f_statistic"]
        ),
        "first_stage_partial_r_squared": float(
            diagnostics["partial_r_squared"]
        ),
        "weak_instrument_flag": bool(
            diagnostics[
                "weak_instrument_flag"
            ]
        ),
    }


def compare_ols_and_iv(
    model_bundle: IVModelBundle,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compare observational OLS and causal IV estimates."""

    prepared_data = (
        model_bundle.prepared_data
    )

    endogenous_column = (
        prepared_data.endogenous_column
    )

    ols_result = fit_ols_benchmark(
        prepared_data
    )

    iv_result = model_bundle.result

    alpha = 1.0 - confidence_level

    ols_confidence_interval = (
        ols_result
        .conf_int(alpha=alpha)
        .loc[endogenous_column]
    )

    iv_confidence_interval = (
        iv_result
        .conf_int(level=confidence_level)
        .loc[endogenous_column]
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "OLS",
                "estimate": float(
                    ols_result.params[
                        endogenous_column
                    ]
                ),
                "standard_error": float(
                    ols_result.bse[
                        endogenous_column
                    ]
                ),
                "p_value": float(
                    ols_result.pvalues[
                        endogenous_column
                    ]
                ),
                "confidence_interval_lower": float(
                    ols_confidence_interval.iloc[0]
                ),
                "confidence_interval_upper": float(
                    ols_confidence_interval.iloc[1]
                ),
                "observations": int(
                    ols_result.nobs
                ),
            },
            {
                "model": "IV_2SLS",
                "estimate": float(
                    iv_result.params[
                        endogenous_column
                    ]
                ),
                "standard_error": float(
                    iv_result.std_errors[
                        endogenous_column
                    ]
                ),
                "p_value": float(
                    iv_result.pvalues[
                        endogenous_column
                    ]
                ),
                "confidence_interval_lower": float(
                    iv_confidence_interval.iloc[0]
                ),
                "confidence_interval_upper": float(
                    iv_confidence_interval.iloc[1]
                ),
                "observations": int(
                    iv_result.nobs
                ),
            },
        ]
    )

    comparison["estimate_difference_vs_ols"] = (
        comparison["estimate"]
        - comparison.loc[
            comparison["model"] == "OLS",
            "estimate",
        ].iloc[0]
    )

    return comparison

# %%
def fit_group_iv_elasticities(
    dataframe: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    outcome_column: str,
    endogenous_column: str,
    instrument_columns: Sequence[str],
    exogenous_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    minimum_observations: int = 60,
    weak_instrument_threshold: float = 10.0,
    plausible_elasticity_range: tuple[float, float] = (-8.0, -0.05),
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Estimate a causal price elasticity separately for each group via 2SLS.

    This is the per-product counterpart of the pooled IV model. It exists
    because ordinary least squares run product-by-product is badly biased
    when price responds to demand: prices get cut in weak weeks and raised
    in strong ones, so the OLS slope absorbs the demand shock and can even
    come out positive. Instrumenting price with cost-side shifters (which
    move price but not demand) breaks that loop.

    Each group's row carries the machinery needed to decide whether to
    trust it downstream:

    ``first_stage_f_statistic``
        Instrument relevance within this group. Below
        ``weak_instrument_threshold`` the 2SLS estimate is unreliable
        regardless of how tight its standard error looks.
    ``is_economically_plausible``
        Whether the point estimate falls inside
        ``plausible_elasticity_range``. A positive elasticity for ordinary
        demand indicates residual confounding, not an upward-sloping
        demand curve.
    ``is_reliable``
        Converged, enough observations, strong instrument, plausible sign
        and magnitude. Only these estimates should drive a price change.

    Returns one row per group, always -- groups that fail to fit are
    returned with ``status`` set and ``is_reliable`` false, so the caller
    can account for every product rather than silently losing some.
    """

    if not group_columns:
        raise ValueError(
            "At least one group column is required."
        )

    group_columns = list(group_columns)
    instrument_columns = list(instrument_columns)
    exogenous_columns = list(exogenous_columns or [])
    categorical_columns = list(categorical_columns or [])

    validate_required_columns(
        dataframe,
        [
            *group_columns,
            outcome_column,
            endogenous_column,
            *instrument_columns,
            *exogenous_columns,
            *categorical_columns,
        ],
        dataframe_name="group IV data",
    )

    lower_plausible, upper_plausible = plausible_elasticity_range

    groupby_columns = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    result_rows: list[dict[str, Any]] = []

    for group_key, group_data in dataframe.groupby(
        groupby_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base_result: dict[str, Any] = {
            **dict(
                zip(
                    group_columns,
                    group_key,
                    strict=True,
                )
            ),
            "available_observations": int(len(group_data)),
            "status": "not_fitted",
            "iv_elasticity": np.nan,
            "standard_error": np.nan,
            "p_value": np.nan,
            "confidence_interval_lower": np.nan,
            "confidence_interval_upper": np.nan,
            "observations": 0,
            "first_stage_f_statistic": np.nan,
            "first_stage_partial_r_squared": np.nan,
            "weak_instrument_flag": True,
            "is_economically_plausible": False,
            "is_statistically_significant": False,
            "is_reliable": False,
            "failure_reason": "",
        }

        if len(group_data) < minimum_observations:
            base_result["status"] = "insufficient_observations"
            base_result["failure_reason"] = (
                f"{len(group_data)} rows available, "
                f"{minimum_observations} required"
            )
            result_rows.append(base_result)
            continue

        # Within a single group, any control that does not vary is
        # collinear with the intercept and would make the design matrix
        # singular; drop those rather than failing the whole group.
        def _varying(columns: Sequence[str]) -> list[str]:
            return [
                column
                for column in columns
                if group_data[column].nunique(dropna=True) > 1
            ]

        try:
            model_bundle = fit_iv_2sls(
                group_data,
                outcome_column=outcome_column,
                endogenous_column=endogenous_column,
                instrument_columns=instrument_columns,
                exogenous_columns=_varying(exogenous_columns),
                categorical_columns=_varying(categorical_columns),
                minimum_observations=minimum_observations,
                robust_covariance=True,
            )

            summary = summarize_iv_model(
                model_bundle,
                confidence_level=confidence_level,
            )

        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            base_result["status"] = "failed"
            base_result["failure_reason"] = str(error)
            result_rows.append(base_result)
            continue

        elasticity = float(summary["causal_effect"])

        is_plausible = bool(
            np.isfinite(elasticity)
            and lower_plausible <= elasticity <= upper_plausible
        )

        weak_instrument = bool(
            not np.isfinite(summary["first_stage_f_statistic"])
            or summary["first_stage_f_statistic"]
            < weak_instrument_threshold
        )

        is_significant = bool(
            np.isfinite(summary["p_value"])
            and summary["p_value"] < (1.0 - confidence_level)
        )

        failure_reasons = []

        if weak_instrument:
            failure_reasons.append("weak_instrument")

        if not is_plausible:
            failure_reasons.append("implausible_estimate")

        base_result.update(
            {
                "status": "fitted",
                "iv_elasticity": elasticity,
                "standard_error": float(summary["standard_error"]),
                "p_value": float(summary["p_value"]),
                "confidence_interval_lower": float(
                    summary["confidence_interval_lower"]
                ),
                "confidence_interval_upper": float(
                    summary["confidence_interval_upper"]
                ),
                "observations": int(summary["observations"]),
                "first_stage_f_statistic": float(
                    summary["first_stage_f_statistic"]
                ),
                "first_stage_partial_r_squared": float(
                    summary["first_stage_partial_r_squared"]
                ),
                "weak_instrument_flag": weak_instrument,
                "is_economically_plausible": is_plausible,
                "is_statistically_significant": is_significant,
                "is_reliable": bool(
                    is_plausible and not weak_instrument
                ),
                "failure_reason": ", ".join(failure_reasons),
            }
        )

        result_rows.append(base_result)

    return pd.DataFrame(result_rows)
