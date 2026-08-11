"""
Pipeline Step 07: LLM-RAG Market Feature Extraction

This pipeline:
1. Loads market notes and retail data
2. Normalizes category keys for matching
3. Builds RAG corpus from market notes
4. Retrieves relevant market evidence by category
5. Extracts market signal features (sentiment, price sensitivity, etc.)
6. Merges RAG features with retail data using temporal constraints
7. Saves enriched dataset for optimization
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_MINIMUM_SIMILARITY,
    RAG_TOP_K,
)
from src.data import load_csv, save_csv, save_json, validate_dataframe
from src.features.text import normalize_category_key
from src.models.rag import (
    build_rag_corpus,
    build_rag_features,
    prepare_document_chunks,
)
from src.pipelines import check_pipeline_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    """Execute the RAG feature extraction pipeline."""

    print("=" * 72)
    print("PIPELINE STEP 07 — LLM-RAG MARKET FEATURE EXTRACTION")
    print("=" * 72)

    # ---------------------------------------------------------
    # Define input and output paths
    # ---------------------------------------------------------

    MARKET_NOTES_FILE = INTERIM_DIR / "market_notes_clean.csv"

    # Priority order for retail input
    RETAIL_INPUT_CANDIDATES = [
        # The optimizer sets a future price, so enrich the future forecast
        # before falling back to historical holdout rows.
        PROCESSED_DIR / "xgboost_next_period_forecast.csv",
        PROCESSED_DIR / "xgboost_forecast_test_predictions.csv",
        INTERIM_DIR / "retail_demand_clean.csv",
    ]

    RAG_LOOKBACK_DAYS = 120

    OUTPUT_FILES = {
        "rag_documents": PROCESSED_DIR / "rag_documents.csv",
        "rag_market_features": PROCESSED_DIR / "rag_market_features.csv",
        "retail_with_rag_features": PROCESSED_DIR / "retail_with_rag_features.csv",
        "rag_feature_coverage": PROCESSED_DIR / "rag_feature_coverage_summary.csv",
        "rag_configuration": PROCESSED_DIR / "rag_configuration.json",
    }

    # ---------------------------------------------------------
    # Check dependencies and select retail input
    # ---------------------------------------------------------

    check_pipeline_dependencies(
        {"market_notes": MARKET_NOTES_FILE},
        pipeline_name="step_07_rag_features",
    )

    # Find first available retail input
    RETAIL_INPUT_FILE = next(
        (f for f in RETAIL_INPUT_CANDIDATES if f.exists()),
        None,
    )

    if RETAIL_INPUT_FILE is None:
        raise FileNotFoundError(
            "No retail input file found. Please run pipeline steps 01 or 06 first."
        )

    print(f"\nInputs:")
    print(f"  - Market notes: {MARKET_NOTES_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  - Retail data:  {RETAIL_INPUT_FILE.relative_to(PROJECT_ROOT)}\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading market notes and retail data...")

    market_notes = load_csv(MARKET_NOTES_FILE, parse_dates=["date"], low_memory=False)
    retail_data = load_csv(RETAIL_INPUT_FILE, parse_dates=["date"], low_memory=False)

    validate_dataframe(market_notes, dataframe_name="market notes")
    validate_dataframe(retail_data, dataframe_name="retail data")

    print(f"Market notes: {len(market_notes):,} rows")
    print(f"Retail data: {len(retail_data):,} rows")
    print(f"Market note date range: {market_notes['date'].min().date()} to {market_notes['date'].max().date()}")
    print(f"Retail date range: {retail_data['date'].min().date()} to {retail_data['date'].max().date()}\n")

    # ---------------------------------------------------------
    # Normalize category keys
    # ---------------------------------------------------------

    print("Normalizing category keys for matching...")

    market_notes["category_key"] = market_notes["category"].map(
        normalize_category_key
    )

    retail_data["category_key"] = retail_data["category"].map(
        normalize_category_key
    )

    market_categories = set(market_notes["category_key"].dropna().unique())
    retail_categories = set(retail_data["category_key"].dropna().unique())

    print(f"Market categories: {len(market_categories)}")
    print(f"Retail categories: {len(retail_categories)}")
    print(f"Matching categories: {len(market_categories & retail_categories)}\n")

    # ---------------------------------------------------------
    # Prepare document chunks for RAG
    # ---------------------------------------------------------

    print("Preparing market document chunks...")

    # Build document text from market notes
    market_notes["document_text"] = (
        market_notes["category"].astype(str)
        + " in "
        + market_notes["region"].astype(str)
        + ". "
        + market_notes["market_note"].astype(str)
    )

    # Prepare chunks
    rag_documents = prepare_document_chunks(
        market_notes,
        text_column="document_text",
        metadata_columns=[
            "date",
            "category",
            "category_key",
            "region",
            "event_type",
            "expected_demand_effect",
            "impact_score",
        ],
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )

    print(f"Document chunks: {len(rag_documents):,}")
    print(f"Chunk size: {RAG_CHUNK_SIZE}")
    print(f"Chunk overlap: {RAG_CHUNK_OVERLAP}\n")

    # ---------------------------------------------------------
    # Build RAG corpus
    # ---------------------------------------------------------

    print("Building RAG corpus with embeddings...")

    # Use TF-IDF backend (simpler, no external dependencies)
    rag_corpus = build_rag_corpus(
        rag_documents,
        text_column="chunk_text",
        metadata_columns=[
            "date",
            "category_key",
            "region",
            "event_type",
            "expected_demand_effect",
            "impact_score",
        ],
        backend="tfidf",
    )

    print(f"Embedding backend: {rag_corpus.backend}")
    print(f"Indexed chunks: {len(rag_corpus.chunks):,}\n")

    # ---------------------------------------------------------
    # Retrieve evidence per retail record and extract real signals
    # ---------------------------------------------------------

    # Each retail row gets its own retrieval query (category + region),
    # and features are transparent keyword-signal counts over the
    # retrieved evidence text (extract_market_signal_features) rather
    # than fabricated constants.
    print("Retrieving market evidence per retail record...")

    rag_feature_bundle = build_rag_features(
        retail_data,
        rag_corpus,
        query_columns=["category", "region"],
        top_k=RAG_TOP_K,
        minimum_similarity=RAG_MINIMUM_SIMILARITY,
        metadata_filter_columns=["category_key", "region"],
        query_date_column="date",
        corpus_date_column="date",
        lookback_days=RAG_LOOKBACK_DAYS,
    )

    retail_with_rag_features = rag_feature_bundle.enriched_data
    rag_market_features = rag_feature_bundle.retrieved_evidence

    print(f"Retrieved evidence rows: {len(rag_market_features):,}")
    print(f"Merged dataset: {len(retail_with_rag_features):,} rows")

    # Calculate RAG coverage
    rag_coverage = (
        retail_with_rag_features["rag_evidence_count"] > 0
    ).mean() * 100

    print(f"RAG coverage: {rag_coverage:.1f}% of retail records\n")

    # ---------------------------------------------------------
    # Generate coverage summary
    # ---------------------------------------------------------

    rag_feature_coverage = (
        retail_with_rag_features.groupby("category")
        .agg(
            {
                "rag_evidence_count": ["mean", "sum"],
                "rag_mean_similarity": "mean",
                "net_demand_signal": "mean",
                "promotion_signal_count": "mean",
                "demand_growth_signal_count": "mean",
                "demand_decline_signal_count": "mean",
            }
        )
        .reset_index()
    )
    rag_feature_coverage.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in rag_feature_coverage.columns.values
    ]

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    print("Saving RAG feature outputs...")

    save_csv(rag_documents, OUTPUT_FILES["rag_documents"])
    save_csv(rag_market_features, OUTPUT_FILES["rag_market_features"])
    save_csv(retail_with_rag_features, OUTPUT_FILES["retail_with_rag_features"])
    save_csv(rag_feature_coverage, OUTPUT_FILES["rag_feature_coverage"])

    # Save configuration
    rag_configuration = {
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "top_k": RAG_TOP_K,
        "minimum_similarity": RAG_MINIMUM_SIMILARITY,
        "lookback_days": RAG_LOOKBACK_DAYS,
        "temporal_rule": "evidence_date <= decision_date",
        "metadata_filters": ["category_key", "region"],
        "embedding_backend": rag_corpus.backend,
        "n_documents": len(rag_documents),
        "n_retrieved_evidence_rows": len(rag_market_features),
        "rag_coverage_percent": float(rag_coverage),
    }

    save_json(rag_configuration, OUTPUT_FILES["rag_configuration"])

    print(f"\nOutputs saved to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {path.name}")

    # ---------------------------------------------------------
    # Feature variance guard
    # ---------------------------------------------------------
    #
    # A signal that takes the same value on every row cannot inform any
    # downstream model or decision. Nine of the fifteen features this step
    # emits were silently constant -- including `net_demand_signal`, the one
    # the step is named for -- and nothing reported it, so "100% coverage"
    # was announced over features carrying zero information.

    signal_columns = [
        column
        for column in retail_with_rag_features.columns
        if column.startswith("rag_") or column.endswith("_signal")
        or column.endswith("_signal_count")
    ]

    constant_columns = [
        column
        for column in signal_columns
        if retail_with_rag_features[column].nunique(dropna=False) <= 1
    ]

    print(
        f"\nFeature variance: {len(signal_columns) - len(constant_columns)} "
        f"of {len(signal_columns)} market features vary across rows"
    )

    if constant_columns:
        print(
            "  WARNING: the following features are constant and carry no "
            "information for any downstream model:"
        )

        for column in constant_columns:
            print(f"    - {column}")

        print(
            "  Either widen the market-note corpus, lower "
            "RAG_MINIMUM_SIMILARITY, or drop these features."
        )

    print("\n" + "=" * 72)
    print("PIPELINE STEP 07 COMPLETE")
    print("=" * 72)
    print(f"\nRAG Features:")
    print(f"  Chunks indexed: {len(rag_documents):,}")
    print(f"  Retrieved evidence rows: {len(rag_market_features):,}")
    print(f"  Coverage: {rag_coverage:.1f}%")
    print(f"\nPrimary output for optimization:")
    print(f"  {OUTPUT_FILES['retail_with_rag_features'].name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
