"""Promotion uplift estimation and targeting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src.data.validation import validate_required_columns


@dataclass(slots=True)
class PreparedPromotionData:
    """Prepared modelling data for promotion-uplift analysis."""

    dataframe: pd.DataFrame
    outcome_column: str
    treatment_column: str
    feature_columns: list[str]


@dataclass(slots=True)
class PromotionUpliftBundle:
    """Store fitted promotion propensity and outcome models."""

    propensity_model: Any
    treated_model: Any
    control_model: Any
    prepared_data: PreparedPromotionData


def normalize_promotion_indicator(
    values: pd.Series,
) -> pd.Series:
    """Convert promotion values into a binary zero-one indicator."""

    if pd.api.types.is_bool_dtype(values):
        return values.astype("int8")

    if pd.api.types.is_numeric_dtype(values):
        return (
            pd.to_numeric(
                values,
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype("int8")
        )

    normalized_values = (
        values
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "promotion",
        "promoted",
        "promo",
        "active",
        "treated",
        "treatment",
    }

    return (
        normalized_values
        .isin(positive_values)
        .astype("int8")
    )


def prepare_promotion_data(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    treatment_column: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
) -> PreparedPromotionData:
    """Prepare numeric and encoded features for uplift modelling."""

    feature_columns = list(feature_columns)
    categorical_columns = list(
        categorical_columns or []
    )

    if not feature_columns and not categorical_columns:
        raise ValueError(
            "At least one numeric or categorical feature is required."
        )

    overlapping_columns = (
        set(feature_columns)
        & set(categorical_columns)
    )

    if overlapping_columns:
        raise ValueError(
            "Columns cannot be included as both numeric and "
            f"categorical features: {sorted(overlapping_columns)}"
        )

    required_columns = [
        outcome_column,
        treatment_column,
        *feature_columns,
        *categorical_columns,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        dataframe_name="promotion data",
    )

    prepared_dataframe = pd.DataFrame(
        index=dataframe.index
    )

    prepared_dataframe[outcome_column] = pd.to_numeric(
        dataframe[outcome_column],
        errors="coerce",
    )

    prepared_dataframe[treatment_column] = (
        normalize_promotion_indicator(
            dataframe[treatment_column]
        )
    )

    for column in feature_columns:
        prepared_dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    generated_feature_columns: list[str] = []

    if categorical_columns:
        categorical_features = pd.get_dummies(
            dataframe[
                categorical_columns
            ].astype("string"),
            columns=categorical_columns,
            prefix=categorical_columns,
            drop_first=True,
            dtype=float,
        )

        generated_feature_columns = list(
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
            "No complete observations remain after preparing "
            "promotion data."
        )

    treatment_counts = (
        prepared_dataframe[
            treatment_column
        ].value_counts()
    )

    if not {0, 1}.issubset(
        treatment_counts.index
    ):
        raise ValueError(
            "Promotion data must contain both treated and "
            "untreated observations."
        )

    final_feature_columns = [
        *feature_columns,
        *generated_feature_columns,
    ]

    for column in final_feature_columns:
        if (
            prepared_dataframe[column]
            .nunique(dropna=True)
            < 2
        ):
            prepared_dataframe = (
                prepared_dataframe.drop(
                    columns=column
                )
            )

    final_feature_columns = [
        column
        for column in final_feature_columns
        if column in prepared_dataframe.columns
    ]

    if not final_feature_columns:
        raise ValueError(
            "No usable modelling features remain after preparation."
        )

    return PreparedPromotionData(
        dataframe=prepared_dataframe,
        outcome_column=outcome_column,
        treatment_column=treatment_column,
        feature_columns=final_feature_columns,
    )


def estimate_naive_promotion_uplift(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    treatment_column: str,
) -> dict[str, float | int]:
    """Compare average outcomes for promoted and non-promoted rows."""

    validate_required_columns(
        dataframe,
        [
            outcome_column,
            treatment_column,
        ],
        dataframe_name="promotion data",
    )

    outcome = pd.to_numeric(
        dataframe[outcome_column],
        errors="coerce",
    )

    treatment = normalize_promotion_indicator(
        dataframe[treatment_column]
    )

    comparison_data = pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
        }
    ).dropna()

    treated_outcomes = comparison_data.loc[
        comparison_data["treatment"] == 1,
        "outcome",
    ]

    control_outcomes = comparison_data.loc[
        comparison_data["treatment"] == 0,
        "outcome",
    ]

    if treated_outcomes.empty or control_outcomes.empty:
        raise ValueError(
            "Both treated and untreated observations are required."
        )

    treated_mean = float(
        treated_outcomes.mean()
    )

    control_mean = float(
        control_outcomes.mean()
    )

    absolute_uplift = (
        treated_mean
        - control_mean
    )

    if control_mean == 0:
        relative_uplift = np.nan
    else:
        relative_uplift = (
            absolute_uplift
            / abs(control_mean)
        )

    return {
        "treated_observations": int(
            len(treated_outcomes)
        ),
        "control_observations": int(
            len(control_outcomes)
        ),
        "treated_mean_outcome": treated_mean,
        "control_mean_outcome": control_mean,
        "absolute_uplift": float(
            absolute_uplift
        ),
        "relative_uplift": float(
            relative_uplift
        ),
    }


def fit_propensity_model(
    prepared_data: PreparedPromotionData,
    *,
    maximum_iterations: int = 2_000,
    regularization_strength: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    """Estimate the probability of receiving a promotion."""

    if regularization_strength <= 0:
        raise ValueError(
            "regularization_strength must be greater than zero."
        )

    dataframe = prepared_data.dataframe

    predictors = dataframe[
        prepared_data.feature_columns
    ].astype(float)

    treatment = dataframe[
        prepared_data.treatment_column
    ].astype(int)

    model = LogisticRegression(
        C=regularization_strength,
        max_iter=maximum_iterations,
        solver="lbfgs",
        random_state=random_state,
    )

    model.fit(
        predictors,
        treatment,
    )

    return model


def calculate_propensity_scores(
    prepared_data: PreparedPromotionData,
    propensity_model: Any,
    *,
    minimum_score: float = 0.01,
    maximum_score: float = 0.99,
) -> pd.Series:
    """Calculate and safely clip promotion propensity scores."""

    if not 0 < minimum_score < maximum_score < 1:
        raise ValueError(
            "Propensity-score limits must satisfy "
            "0 < minimum_score < maximum_score < 1."
        )

    predictors = prepared_data.dataframe[
        prepared_data.feature_columns
    ].astype(float)

    probabilities = propensity_model.predict_proba(
        predictors
    )

    class_values = list(
        propensity_model.classes_
    )

    if 1 not in class_values:
        raise ValueError(
            "The propensity model does not contain treatment class 1."
        )

    treated_class_index = class_values.index(1)

    propensity_scores = probabilities[
        :,
        treated_class_index,
    ]

    propensity_scores = np.clip(
        propensity_scores,
        minimum_score,
        maximum_score,
    )

    return pd.Series(
        propensity_scores,
        index=prepared_data.dataframe.index,
        name="propensity_score",
    )


def trim_common_support(
    prepared_data: PreparedPromotionData,
    propensity_scores: pd.Series,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[PreparedPromotionData, pd.Series]:
    """Remove observations outside the overlapping propensity region."""

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Quantiles must satisfy "
            "0 <= lower_quantile < upper_quantile <= 1."
        )

    dataframe = prepared_data.dataframe.copy()

    treatment_column = (
        prepared_data.treatment_column
    )

    treated_scores = propensity_scores.loc[
        dataframe[treatment_column] == 1
    ]

    control_scores = propensity_scores.loc[
        dataframe[treatment_column] == 0
    ]

    lower_bound = max(
        float(
            treated_scores.quantile(
                lower_quantile
            )
        ),
        float(
            control_scores.quantile(
                lower_quantile
            )
        ),
    )

    upper_bound = min(
        float(
            treated_scores.quantile(
                upper_quantile
            )
        ),
        float(
            control_scores.quantile(
                upper_quantile
            )
        ),
    )

    if lower_bound >= upper_bound:
        raise ValueError(
            "No meaningful common propensity-score support was found."
        )

    support_mask = propensity_scores.between(
        lower_bound,
        upper_bound,
        inclusive="both",
    )

    trimmed_dataframe = (
        dataframe.loc[support_mask]
        .reset_index(drop=True)
    )

    trimmed_scores = (
        propensity_scores.loc[support_mask]
        .reset_index(drop=True)
    )

    trimmed_data = PreparedPromotionData(
        dataframe=trimmed_dataframe,
        outcome_column=(
            prepared_data.outcome_column
        ),
        treatment_column=(
            prepared_data.treatment_column
        ),
        feature_columns=list(
            prepared_data.feature_columns
        ),
    )

    return trimmed_data, trimmed_scores


def calculate_ipw_weights(
    treatment: pd.Series,
    propensity_scores: pd.Series,
    *,
    stabilized: bool = True,
) -> pd.Series:
    """Calculate inverse-probability treatment weights."""

    treatment_values = pd.to_numeric(
        treatment,
        errors="coerce",
    ).astype(float)

    propensity_values = pd.to_numeric(
        propensity_scores,
        errors="coerce",
    ).clip(
        lower=0.01,
        upper=0.99,
    )

    if stabilized:
        treatment_probability = float(
            treatment_values.mean()
        )

        treated_weight = (
            treatment_probability
            / propensity_values
        )

        control_weight = (
            1.0
            - treatment_probability
        ) / (
            1.0
            - propensity_values
        )

    else:
        treated_weight = (
            1.0
            / propensity_values
        )

        control_weight = (
            1.0
            / (
                1.0
                - propensity_values
            )
        )

    weights = np.where(
        treatment_values == 1,
        treated_weight,
        control_weight,
    )

    return pd.Series(
        weights,
        index=treatment.index,
        name="ipw_weight",
    )


def estimate_ipw_promotion_effect(
    prepared_data: PreparedPromotionData,
    propensity_scores: pd.Series,
    *,
    stabilized_weights: bool = True,
) -> dict[str, float | int]:
    """Estimate the average promotion effect using IPW."""

    dataframe = prepared_data.dataframe

    outcome = dataframe[
        prepared_data.outcome_column
    ].astype(float)

    treatment = dataframe[
        prepared_data.treatment_column
    ].astype(int)

    weights = calculate_ipw_weights(
        treatment,
        propensity_scores,
        stabilized=stabilized_weights,
    )

    treated_mask = treatment == 1
    control_mask = treatment == 0

    treated_weighted_mean = float(
        np.average(
            outcome.loc[treated_mask],
            weights=weights.loc[
                treated_mask
            ],
        )
    )

    control_weighted_mean = float(
        np.average(
            outcome.loc[control_mask],
            weights=weights.loc[
                control_mask
            ],
        )
    )

    average_treatment_effect = (
        treated_weighted_mean
        - control_weighted_mean
    )

    if control_weighted_mean == 0:
        relative_effect = np.nan
    else:
        relative_effect = (
            average_treatment_effect
            / abs(control_weighted_mean)
        )

    return {
        "observations": int(
            len(dataframe)
        ),
        "weighted_treated_outcome": (
            treated_weighted_mean
        ),
        "weighted_control_outcome": (
            control_weighted_mean
        ),
        "average_treatment_effect": float(
            average_treatment_effect
        ),
        "relative_treatment_effect": float(
            relative_effect
        ),
        "minimum_propensity_score": float(
            propensity_scores.min()
        ),
        "maximum_propensity_score": float(
            propensity_scores.max()
        ),
        "mean_ipw_weight": float(
            weights.mean()
        ),
        "maximum_ipw_weight": float(
            weights.max()
        ),
    }


def fit_promotion_t_learner(
    prepared_data: PreparedPromotionData,
    *,
    outcome_model: Any | None = None,
    minimum_group_observations: int = 20,
    random_state: int = 42,
) -> PromotionUpliftBundle:
    """Fit separate promoted and non-promoted outcome models."""

    dataframe = prepared_data.dataframe

    treatment_column = (
        prepared_data.treatment_column
    )

    outcome_column = (
        prepared_data.outcome_column
    )

    treated_data = dataframe.loc[
        dataframe[treatment_column] == 1
    ]

    control_data = dataframe.loc[
        dataframe[treatment_column] == 0
    ]

    if len(treated_data) < minimum_group_observations:
        raise ValueError(
            "Insufficient promoted observations. "
            f"Required: {minimum_group_observations:,}; "
            f"available: {len(treated_data):,}."
        )

    if len(control_data) < minimum_group_observations:
        raise ValueError(
            "Insufficient non-promoted observations. "
            f"Required: {minimum_group_observations:,}; "
            f"available: {len(control_data):,}."
        )

    if outcome_model is None:
        base_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )
    else:
        base_model = outcome_model

    treated_model = clone(
        base_model
    )

    control_model = clone(
        base_model
    )

    treated_model.fit(
        treated_data[
            prepared_data.feature_columns
        ].astype(float),
        treated_data[
            outcome_column
        ].astype(float),
    )

    control_model.fit(
        control_data[
            prepared_data.feature_columns
        ].astype(float),
        control_data[
            outcome_column
        ].astype(float),
    )

    propensity_model = fit_propensity_model(
        prepared_data,
        random_state=random_state,
    )

    return PromotionUpliftBundle(
        propensity_model=propensity_model,
        treated_model=treated_model,
        control_model=control_model,
        prepared_data=prepared_data,
    )


def predict_promotion_uplift(
    model_bundle: PromotionUpliftBundle,
) -> pd.DataFrame:
    """Predict treated, untreated, and incremental outcomes."""

    prepared_data = (
        model_bundle.prepared_data
    )

    dataframe = (
        prepared_data.dataframe.copy()
    )

    predictors = dataframe[
        prepared_data.feature_columns
    ].astype(float)

    predicted_treated_outcome = (
        model_bundle.treated_model.predict(
            predictors
        )
    )

    predicted_control_outcome = (
        model_bundle.control_model.predict(
            predictors
        )
    )

    predictions = dataframe.copy()

    predictions[
        "predicted_promoted_outcome"
    ] = predicted_treated_outcome

    predictions[
        "predicted_control_outcome"
    ] = predicted_control_outcome

    predictions["predicted_uplift"] = (
        predictions[
            "predicted_promoted_outcome"
        ]
        - predictions[
            "predicted_control_outcome"
        ]
    )

    control_denominator = predictions[
        "predicted_control_outcome"
    ].replace(
        0,
        np.nan,
    )

    predictions["predicted_uplift_rate"] = (
        predictions["predicted_uplift"]
        / control_denominator.abs()
    )

    predictions["propensity_score"] = (
        calculate_propensity_scores(
            prepared_data,
            model_bundle.propensity_model,
        )
    )

    predictions["recommended_promotion"] = (
        predictions["predicted_uplift"] > 0
    ).astype("int8")

    return predictions


def summarize_promotion_predictions(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize predicted promotion effectiveness."""

    validate_required_columns(
        predictions,
        [
            "predicted_promoted_outcome",
            "predicted_control_outcome",
            "predicted_uplift",
            "recommended_promotion",
        ],
        dataframe_name="promotion predictions",
    )

    positive_uplift_mask = (
        predictions[
            "predicted_uplift"
        ] > 0
    )

    return {
        "observations": int(
            len(predictions)
        ),
        "average_predicted_promoted_outcome": float(
            predictions[
                "predicted_promoted_outcome"
            ].mean()
        ),
        "average_predicted_control_outcome": float(
            predictions[
                "predicted_control_outcome"
            ].mean()
        ),
        "average_predicted_uplift": float(
            predictions[
                "predicted_uplift"
            ].mean()
        ),
        "median_predicted_uplift": float(
            predictions[
                "predicted_uplift"
            ].median()
        ),
        "positive_uplift_observations": int(
            positive_uplift_mask.sum()
        ),
        "positive_uplift_rate": float(
            positive_uplift_mask.mean()
        ),
        "recommended_promotions": int(
            predictions[
                "recommended_promotion"
            ].sum()
        ),
    }


def rank_promotion_opportunities(
    predictions: pd.DataFrame,
    *,
    top_n: int | None = None,
    minimum_uplift: float = 0.0,
) -> pd.DataFrame:
    """Rank observations by predicted incremental promotion impact."""

    validate_required_columns(
        predictions,
        ["predicted_uplift"],
        dataframe_name="promotion predictions",
    )

    ranked_predictions = (
        predictions.loc[
            predictions[
                "predicted_uplift"
            ] >= minimum_uplift
        ]
        .sort_values(
            "predicted_uplift",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranked_predictions[
        "promotion_priority_rank"
    ] = np.arange(
        1,
        len(ranked_predictions) + 1,
    )

    if top_n is not None:
        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        ranked_predictions = (
            ranked_predictions.head(
                top_n
            )
        )

    return ranked_predictions


def evaluate_uplift_by_decile(
    predictions: pd.DataFrame,
    *,
    outcome_column: str,
    treatment_column: str,
    number_of_groups: int = 10,
) -> pd.DataFrame:
    """Evaluate observed promotion performance by uplift ranking group."""

    validate_required_columns(
        predictions,
        [
            outcome_column,
            treatment_column,
            "predicted_uplift",
        ],
        dataframe_name="promotion predictions",
    )

    if number_of_groups < 2:
        raise ValueError(
            "number_of_groups must be at least 2."
        )

    evaluation_data = predictions.copy()

    percentile_rank = (
        evaluation_data[
            "predicted_uplift"
        ]
        .rank(
            method="first",
            pct=True,
        )
    )

    evaluation_data["uplift_group"] = pd.cut(
        percentile_rank,
        bins=np.linspace(
            0,
            1,
            number_of_groups + 1,
        ),
        labels=list(
            range(
                number_of_groups,
                0,
                -1,
            )
        ),
        include_lowest=True,
    ).astype(int)

    summary_rows: list[
        dict[str, float | int]
    ] = []

    for uplift_group, group_data in (
        evaluation_data.groupby(
            "uplift_group",
            sort=True,
        )
    ):
        treated_outcomes = group_data.loc[
            group_data[treatment_column] == 1,
            outcome_column,
        ]

        control_outcomes = group_data.loc[
            group_data[treatment_column] == 0,
            outcome_column,
        ]

        if (
            treated_outcomes.empty
            or control_outcomes.empty
        ):
            observed_uplift = np.nan
        else:
            observed_uplift = float(
                treated_outcomes.mean()
                - control_outcomes.mean()
            )

        summary_rows.append(
            {
                "uplift_group": int(
                    uplift_group
                ),
                "observations": int(
                    len(group_data)
                ),
                "treatment_rate": float(
                    group_data[
                        treatment_column
                    ].mean()
                ),
                "average_predicted_uplift": float(
                    group_data[
                        "predicted_uplift"
                    ].mean()
                ),
                "observed_uplift": (
                    observed_uplift
                ),
                "average_observed_outcome": float(
                    group_data[
                        outcome_column
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    ).sort_values(
        "uplift_group"
    ).reset_index(drop=True)