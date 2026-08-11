"""Reusable statistical and machine-learning models."""

from .elasticity import (
    build_elasticity_design_matrix,
    calculate_arc_elasticity,
    classify_price_elasticity,
    estimate_category_heterogeneity,
    extract_price_elasticity,
    fit_group_elasticities,
    fit_log_log_elasticity,
    predict_quantity_from_elasticity,
    prepare_elasticity_data,
    shrink_product_elasticities,
    simulate_price_scenarios,
    summarize_elasticity_model,
)

from .promotion import (
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

from .forecasting import (
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

from .rag import (
    RAGCorpus,
    RAGFeatureBundle,
    aggregate_retrieval_features,
    build_rag_corpus,
    build_rag_features,
    build_retail_rag_query,
    count_keyword_signals,
    encode_rag_queries,
    extract_market_signal_features,
    normalize_document_text,
    prepare_document_chunks,
    retrieve_batch_evidence,
    retrieve_market_evidence,
    split_text_into_chunks,
)

from .evaluation import (
    summarize_segment_accuracy,
)


__all__ = [
    # Elasticity models
    "build_elasticity_design_matrix",
    "calculate_arc_elasticity",
    "classify_price_elasticity",
    "estimate_category_heterogeneity",
    "extract_price_elasticity",
    "fit_group_elasticities",
    "fit_log_log_elasticity",
    "predict_quantity_from_elasticity",
    "prepare_elasticity_data",
    "shrink_product_elasticities",
    "simulate_price_scenarios",
    "summarize_elasticity_model",

    # Promotion uplift models
    "PreparedPromotionData",
    "PromotionUpliftBundle",
    "calculate_ipw_weights",
    "calculate_propensity_scores",
    "estimate_ipw_promotion_effect",
    "estimate_naive_promotion_uplift",
    "evaluate_uplift_by_decile",
    "fit_promotion_t_learner",
    "fit_propensity_model",
    "normalize_promotion_indicator",
    "predict_promotion_uplift",
    "prepare_promotion_data",
    "rank_promotion_opportunities",
    "summarize_promotion_predictions",
    "trim_common_support",

    # Forecasting models
    "ForecastModelBundle",
    "PreparedForecastData",
    "calculate_forecast_metrics",
    "compare_forecast_with_baseline",
    "create_xgboost_regressor",
    "fit_xgboost_forecast",
    "get_feature_importance",
    "predict_future_demand",
    "prepare_forecasting_data",
    "prepare_future_features",
    "time_based_train_validation_split",

    # RAG models
    "RAGCorpus",
    "RAGFeatureBundle",
    "aggregate_retrieval_features",
    "build_rag_corpus",
    "build_rag_features",
    "build_retail_rag_query",
    "count_keyword_signals",
    "encode_rag_queries",
    "extract_market_signal_features",
    "normalize_document_text",
    "prepare_document_chunks",
    "retrieve_batch_evidence",
    "retrieve_market_evidence",
    "split_text_into_chunks",

    # Evaluation functions
    "summarize_segment_accuracy",
]
