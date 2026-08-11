"""Price-elasticity estimation and simulation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.validation import validate_required_columns


def calculate_arc_elasticity(
    old_quantity: float,
    new_quantity: float,
    old_price: float,
    new_price: float,
) -> float:
    """
    Calculate midpoint, or arc, price elasticity of demand.

    The midpoint formula reduces sensitivity to the direction of change.
    """

    values = [
        old_quantity,
        new_quantity,
        old_price,
        new_price,
    ]

    if not all(np.isfinite(value) for value in values):
        return np.nan

    # A zero endpoint (e.g. a stockout or a not-yet-launched product) makes
    # "percent change" economically ill-defined even though the midpoint
    # formula avoids a literal division by zero; treat it as undefined
    # rather than reporting a technically-computable but misleading value.
    if any(value == 0 for value in values):
        return np.nan

    average_quantity = (
        float(old_quantity) + float(new_quantity)
    ) / 2

    average_price = (
        float(old_price) + float(new_price)
    ) / 2

    quantity_change = (
        float(new_quantity) - float(old_quantity)
    )

    price_change = (
        float(new_price) - float(old_price)
    )

    if (
        average_quantity == 0
        or average_price == 0
        or price_change == 0
    ):
        return np.nan

    quantity_change_rate = (
        quantity_change / average_quantity
    )

    price_change_rate = (
        price_change / average_price
    )

    return float(
        quantity_change_rate / price_change_rate
    )


def classify_price_elasticity(
    elasticity: float,
    *,
    unitary_tolerance: float = 0.05,
) -> str:
    """Classify a price-elasticity estimate."""

    if elasticity is None or not np.isfinite(elasticity):
        return "unknown"

    elasticity = float(elasticity)
    absolute_elasticity = abs(elasticity)

    if elasticity > 0:
        return "positive"

    if absolute_elasticity < (
        1.0 - unitary_tolerance
    ):
        return "inelastic"

    if absolute_elasticity > (
        1.0 + unitary_tolerance
    ):
        return "elastic"

    return "unit_elastic"


def prepare_elasticity_data(
    dataframe: pd.DataFrame,
    *,
    quantity_column: str = "quantity",
    price_column: str = "price",
    control_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare clean positive-valued observations for a log-log model.

    Rows with missing, zero, or negative quantities or prices are removed.
    """

    control_columns = list(control_columns or [])
    categorical_columns = list(
        categorical_columns or []
    )

    required_columns = [
        quantity_column,
        price_column,
        *control_columns,
        *categorical_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="elasticity data",
    )

    result = dataframe[
        required_columns
    ].copy()

    result[quantity_column] = pd.to_numeric(
        result[quantity_column],
        errors="coerce",
    )

    result[price_column] = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    for column in control_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    valid_mask = (
        result[quantity_column].gt(0)
        & result[price_column].gt(0)
        & result[quantity_column].notna()
        & result[price_column].notna()
    )

    result = result.loc[
        valid_mask
    ].copy()

    result["log_quantity"] = np.log(
        result[quantity_column]
    )

    result["log_price"] = np.log(
        result[price_column]
    )

    return result.reset_index(drop=True)


def build_elasticity_design_matrix(
    dataframe: pd.DataFrame,
    *,
    control_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Create the explanatory-variable matrix for an elasticity model."""

    control_columns = list(control_columns or [])
    categorical_columns = list(
        categorical_columns or []
    )

    validate_required_columns(
        dataframe,
        [
            "log_price",
            *control_columns,
            *categorical_columns,
        ],
        dataframe_name="prepared elasticity data",
    )

    numeric_parts: list[pd.DataFrame] = [
        dataframe[["log_price"]].copy()
    ]

    if control_columns:
        numeric_controls = dataframe[
            control_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        numeric_parts.append(
            numeric_controls
        )

    design_matrix = pd.concat(
        numeric_parts,
        axis=1,
    )

    if categorical_columns:
        categorical_features = pd.get_dummies(
            dataframe[categorical_columns].astype(
                "string"
            ),
            columns=categorical_columns,
            prefix=categorical_columns,
            drop_first=True,
            dtype=float,
        )

        design_matrix = pd.concat(
            [
                design_matrix,
                categorical_features,
            ],
            axis=1,
        )

    design_matrix = design_matrix.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if add_intercept:
        design_matrix = sm.add_constant(
            design_matrix,
            has_constant="add",
        )

    return design_matrix.astype(float)


def fit_log_log_elasticity(
    dataframe: pd.DataFrame,
    *,
    quantity_column: str = "quantity",
    price_column: str = "price",
    control_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    minimum_observations: int = 20,
    robust_covariance: bool = True,
) -> Any:
    """
    Estimate price elasticity using an ordinary least-squares log-log model.

    In this specification, the coefficient on ``log_price`` is the
    estimated price elasticity of demand.
    """

    prepared_data = prepare_elasticity_data(
        dataframe,
        quantity_column=quantity_column,
        price_column=price_column,
        control_columns=control_columns,
        categorical_columns=categorical_columns,
    )

    design_matrix = build_elasticity_design_matrix(
        prepared_data,
        control_columns=control_columns,
        categorical_columns=categorical_columns,
        add_intercept=True,
    )

    model_data = pd.concat(
        [
            prepared_data[["log_quantity"]],
            design_matrix,
        ],
        axis=1,
    ).dropna()

    if len(model_data) < minimum_observations:
        raise ValueError(
            "Insufficient valid observations for elasticity "
            f"estimation. Required: {minimum_observations:,}; "
            f"available: {len(model_data):,}."
        )

    target = model_data["log_quantity"]

    predictors = model_data.drop(
        columns="log_quantity"
    )

    model = sm.OLS(
        target,
        predictors,
    )

    if robust_covariance:
        fitted_model = model.fit(
            cov_type="HC3"
        )
    else:
        fitted_model = model.fit()

    fitted_model.elasticity_metadata = {
        "quantity_column": quantity_column,
        "price_column": price_column,
        "control_columns": list(
            control_columns or []
        ),
        "categorical_columns": list(
            categorical_columns or []
        ),
        "minimum_observations": minimum_observations,
        "robust_covariance": robust_covariance,
        "observations_used": int(
            fitted_model.nobs
        ),
    }

    return fitted_model


def extract_price_elasticity(
    fitted_model: Any,
    *,
    price_term: str = "log_price",
) -> float:
    """Extract the price-elasticity coefficient from a fitted model."""

    if not hasattr(fitted_model, "params"):
        raise TypeError(
            "fitted_model must be a fitted statsmodels result."
        )

    if price_term not in fitted_model.params.index:
        raise KeyError(
            f"Model does not contain the term '{price_term}'."
        )

    return float(
        fitted_model.params[price_term]
    )


def summarize_elasticity_model(
    fitted_model: Any,
    *,
    price_term: str = "log_price",
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Create a compact statistical summary of an elasticity model."""

    elasticity = extract_price_elasticity(
        fitted_model,
        price_term=price_term,
    )

    alpha = 1.0 - confidence_level

    confidence_interval = fitted_model.conf_int(
        alpha=alpha
    ).loc[price_term]

    return {
        "elasticity": elasticity,
        "elasticity_class": (
            classify_price_elasticity(
                elasticity
            )
        ),
        "standard_error": float(
            fitted_model.bse[price_term]
        ),
        "p_value": float(
            fitted_model.pvalues[price_term]
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
    }


def predict_quantity_from_elasticity(
    base_quantity: float | np.ndarray,
    base_price: float | np.ndarray,
    new_price: float | np.ndarray,
    elasticity: float,
) -> float | np.ndarray:
    """
    Predict quantity after a price change using constant elasticity.

    Formula:

    new quantity =
    base quantity × (new price / base price) ** elasticity
    """

    base_quantity_array = np.asarray(
        base_quantity,
        dtype=float,
    )

    base_price_array = np.asarray(
        base_price,
        dtype=float,
    )

    new_price_array = np.asarray(
        new_price,
        dtype=float,
    )

    if np.any(base_quantity_array < 0):
        raise ValueError(
            "base_quantity cannot contain negative values."
        )

    if np.any(base_price_array <= 0):
        raise ValueError(
            "base_price must contain positive values."
        )

    if np.any(new_price_array <= 0):
        raise ValueError(
            "new_price must contain positive values."
        )

    predicted_quantity = (
        base_quantity_array
        * (
            new_price_array
            / base_price_array
        )
        ** float(elasticity)
    )

    if predicted_quantity.ndim == 0:
        return float(predicted_quantity)

    return predicted_quantity


def simulate_price_scenarios(
    *,
    base_price: float,
    base_quantity: float,
    elasticity: float,
    price_change_rates: Sequence[float] = (
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ),
    unit_cost: float | None = None,
) -> pd.DataFrame:
    """Simulate quantity, revenue, and optional profit by price scenario."""

    if base_price <= 0:
        raise ValueError(
            "base_price must be greater than zero."
        )

    if base_quantity < 0:
        raise ValueError(
            "base_quantity cannot be negative."
        )

    if unit_cost is not None and unit_cost < 0:
        raise ValueError(
            "unit_cost cannot be negative."
        )

    scenario_data = pd.DataFrame(
        {
            "price_change_rate": list(
                price_change_rates
            )
        }
    )

    scenario_data["price_multiplier"] = (
        1.0
        + scenario_data["price_change_rate"]
    )

    if (
        scenario_data["price_multiplier"]
        <= 0
    ).any():
        raise ValueError(
            "Every simulated price must be greater than zero."
        )

    scenario_data["candidate_price"] = (
        float(base_price)
        * scenario_data["price_multiplier"]
    )

    scenario_data["expected_quantity"] = (
        predict_quantity_from_elasticity(
            base_quantity=float(base_quantity),
            base_price=float(base_price),
            new_price=scenario_data[
                "candidate_price"
            ].to_numpy(),
            elasticity=float(elasticity),
        )
    )

    scenario_data["expected_revenue"] = (
        scenario_data["candidate_price"]
        * scenario_data["expected_quantity"]
    )

    base_revenue = (
        float(base_price)
        * float(base_quantity)
    )

    scenario_data["revenue_change"] = (
        scenario_data["expected_revenue"]
        - base_revenue
    )

    if base_revenue == 0:
        scenario_data[
            "revenue_change_rate"
        ] = np.nan
    else:
        scenario_data[
            "revenue_change_rate"
        ] = (
            scenario_data["revenue_change"]
            / base_revenue
        )

    if unit_cost is not None:
        scenario_data["unit_margin"] = (
            scenario_data["candidate_price"]
            - float(unit_cost)
        )

        scenario_data["expected_profit"] = (
            scenario_data["unit_margin"]
            * scenario_data[
                "expected_quantity"
            ]
        )

        base_profit = (
            float(base_price)
            - float(unit_cost)
        ) * float(base_quantity)

        scenario_data["profit_change"] = (
            scenario_data["expected_profit"]
            - base_profit
        )

        if base_profit == 0:
            scenario_data[
                "profit_change_rate"
            ] = np.nan
        else:
            scenario_data[
                "profit_change_rate"
            ] = (
                scenario_data["profit_change"]
                / abs(base_profit)
            )

    return scenario_data.sort_values(
        "candidate_price"
    ).reset_index(drop=True)


def fit_group_elasticities(
    dataframe: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    quantity_column: str = "quantity",
    price_column: str = "price",
    control_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    minimum_observations: int = 20,
    robust_covariance: bool = True,
) -> pd.DataFrame:
    """Estimate separate price-elasticity models by product or segment."""

    if not group_columns:
        raise ValueError(
            "At least one group column is required."
        )

    required_columns = [
        *group_columns,
        quantity_column,
        price_column,
        *(control_columns or []),
        *(categorical_columns or []),
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="group elasticity data",
    )

    result_rows: list[
        dict[str, object]
    ] = []

    groupby_columns: str | list[str]

    if len(group_columns) == 1:
        groupby_columns = group_columns[0]
    else:
        groupby_columns = list(group_columns)

    grouped_data = dataframe.groupby(
        groupby_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group_data in grouped_data:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (group_key,)

        group_values = dict(
            zip(
                group_columns,
                group_key,
                strict=True,
            )
        )

        valid_observations = int(
            (
                pd.to_numeric(
                    group_data[quantity_column],
                    errors="coerce",
                ).gt(0)
                & pd.to_numeric(
                    group_data[price_column],
                    errors="coerce",
                ).gt(0)
            ).sum()
        )

        base_result: dict[str, object] = {
            **group_values,
            "available_observations": (
                len(group_data)
            ),
            "valid_observations": (
                valid_observations
            ),
        }

        if valid_observations < minimum_observations:
            result_rows.append(
                {
                    **base_result,
                    "status": (
                        "insufficient_observations"
                    ),
                    "elasticity": np.nan,
                    "elasticity_class": "unknown",
                    "standard_error": np.nan,
                    "p_value": np.nan,
                    "confidence_interval_lower": (
                        np.nan
                    ),
                    "confidence_interval_upper": (
                        np.nan
                    ),
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                }
            )

            continue

        try:
            fitted_model = fit_log_log_elasticity(
                group_data,
                quantity_column=quantity_column,
                price_column=price_column,
                control_columns=control_columns,
                categorical_columns=(
                    categorical_columns
                ),
                minimum_observations=(
                    minimum_observations
                ),
                robust_covariance=(
                    robust_covariance
                ),
            )

            model_summary = (
                summarize_elasticity_model(
                    fitted_model
                )
            )

            result_rows.append(
                {
                    **base_result,
                    "status": "success",
                    **model_summary,
                }
            )

        except Exception as error:
            result_rows.append(
                {
                    **base_result,
                    "status": "model_failed",
                    "error_message": str(error),
                    "elasticity": np.nan,
                    "elasticity_class": "unknown",
                    "standard_error": np.nan,
                    "p_value": np.nan,
                    "confidence_interval_lower": (
                        np.nan
                    ),
                    "confidence_interval_upper": (
                        np.nan
                    ),
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                }
            )

    return pd.DataFrame(result_rows)


def estimate_category_heterogeneity(
    product_elasticities: np.ndarray,
    product_standard_errors: np.ndarray,
    category_elasticity: float,
) -> float:
    """
    Estimate how much products in a category truly vary beyond sampling noise.

    A method-of-moments heterogeneity estimate (tau-squared) in the spirit
    of random-effects meta-analysis: the amount by which the weighted sum
    of squared deviations from the category's elasticity exceeds what pure
    sampling noise would produce, given each product's own standard error.
    """

    if len(product_elasticities) < 2:
        return 0.0

    weights = 1.0 / np.square(product_standard_errors)

    q_statistic = float(
        np.sum(
            weights
            * np.square(product_elasticities - category_elasticity)
        )
    )

    degrees_of_freedom = len(product_elasticities)

    return float(
        max(
            0.0,
            (q_statistic - degrees_of_freedom) / np.sum(weights),
        )
    )


def shrink_product_elasticities(
    product_estimates: pd.DataFrame,
    category_estimates: pd.DataFrame,
    *,
    category_column: str = "category",
) -> pd.DataFrame:
    """
    Partially pool noisy per-product elasticity estimates toward their
    category's directly-estimated elasticity (empirical-Bayes shrinkage).

    Each product's own OLS estimate is a normal likelihood with variance
    ``standard_error ** 2``; each category's elasticity (already fit on
    far more data in ``category_estimates``) is treated as a fixed prior
    mean. The prior's spread around that mean -- how much products in a
    category genuinely differ from one another -- is estimated from the
    data itself via ``estimate_category_heterogeneity``, then combined
    with each product's own precision to compute a posterior (shrunk)
    elasticity and standard error:

        weight   = tau_squared / (tau_squared + standard_error ** 2)
        shrunk   = weight * elasticity + (1 - weight) * category_elasticity
        variance = weight * standard_error ** 2

    A product with a small standard error (lots of clean data) keeps a
    weight near 1 and stays close to its own estimate; a product with a
    large standard error (sparse or noisy data) gets pulled toward the
    category baseline. This replaces an all-or-nothing "trust it or use a
    flat default" rule with a continuous, data-driven one.

    As a final backstop, a shrunk estimate that is still non-negative
    (economically implausible for ordinary demand) is replaced with the
    category's own elasticity, which is fit on enough data to be far less
    likely to have the wrong sign.

    Caveat: the heterogeneity estimate (tau-squared) is computed from
    every product with a valid standard error, including wrong-signed or
    otherwise implausible ones. If a category's products share a common
    source of confounding bias -- not just genuine differences in true
    price sensitivity -- that bias inflates tau-squared and can grant a
    noisy-but-plausible product more trust in its own estimate than a
    p-value alone would suggest (the backstop above still prevents the
    worst case: a final estimate with the wrong sign).
    """

    required_columns = ["status", "elasticity", "standard_error"]

    validate_required_columns(
        product_estimates,
        [*required_columns, category_column],
        dataframe_name="product elasticity estimates",
    )

    validate_required_columns(
        category_estimates,
        ["status", "elasticity", "standard_error", category_column],
        dataframe_name="category elasticity estimates",
    )

    category_lookup = category_estimates.set_index(category_column)

    result = product_estimates.copy()

    result["category_elasticity_prior"] = np.nan
    result["category_tau_squared"] = np.nan
    result["shrinkage_weight"] = np.nan
    result["shrunk_elasticity"] = np.nan
    result["shrunk_standard_error"] = np.nan

    for category, group in result.groupby(category_column):
        if (
            category not in category_lookup.index
            or category_lookup.loc[category, "status"] != "success"
        ):
            continue

        category_elasticity = float(
            category_lookup.loc[category, "elasticity"]
        )

        category_standard_error = float(
            category_lookup.loc[category, "standard_error"]
        )

        reliable_rows = group.loc[
            (group["status"] == "success")
            & group["standard_error"].gt(0)
            & group["standard_error"].notna()
        ]

        tau_squared = estimate_category_heterogeneity(
            reliable_rows["elasticity"].to_numpy(),
            reliable_rows["standard_error"].to_numpy(),
            category_elasticity,
        )

        result.loc[group.index, "category_elasticity_prior"] = (
            category_elasticity
        )

        result.loc[group.index, "category_tau_squared"] = tau_squared

        for row_index in group.index:
            has_own_estimate = row_index in reliable_rows.index

            if not has_own_estimate:
                weight = 0.0
                shrunk = category_elasticity
                shrunk_se = category_standard_error
            elif tau_squared <= 0:
                weight = 0.0
                shrunk = category_elasticity
                shrunk_se = category_standard_error
            else:
                own_elasticity = float(result.loc[row_index, "elasticity"])
                own_se = float(result.loc[row_index, "standard_error"])
                weight = tau_squared / (tau_squared + own_se**2)
                shrunk = (
                    weight * own_elasticity
                    + (1.0 - weight) * category_elasticity
                )
                shrunk_se = float(np.sqrt(weight * own_se**2))

            # Economic-plausibility backstop: a shrunk estimate that is
            # still non-negative falls all the way back to the
            # category's own (far better-estimated) elasticity.
            if not np.isfinite(shrunk) or shrunk >= 0:
                weight = 0.0
                shrunk = category_elasticity
                shrunk_se = category_standard_error

            result.loc[row_index, "shrinkage_weight"] = weight
            result.loc[row_index, "shrunk_elasticity"] = shrunk
            result.loc[row_index, "shrunk_standard_error"] = shrunk_se

    result["shrunk_confidence_interval_lower"] = (
        result["shrunk_elasticity"] - 1.96 * result["shrunk_standard_error"]
    )

    result["shrunk_confidence_interval_upper"] = (
        result["shrunk_elasticity"] + 1.96 * result["shrunk_standard_error"]
    )

    return result