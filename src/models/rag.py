"""Retrieval-augmented feature engineering utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.validation import validate_required_columns


@dataclass(slots=True)
class RAGCorpus:
    """Store document chunks, embeddings, and the fitted encoder."""

    chunks: pd.DataFrame
    embeddings: Any
    encoder: Any
    backend: str
    text_column: str
    metadata_columns: list[str]


@dataclass(slots=True)
class RAGFeatureBundle:
    """Store enriched retail data and its supporting evidence."""

    enriched_data: pd.DataFrame
    retrieved_evidence: pd.DataFrame
    corpus: RAGCorpus


def normalize_document_text(
    value: object,
) -> str:
    """Clean text while preserving readable sentences."""

    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = 180,
    chunk_overlap: int = 30,
) -> list[str]:
    """Split text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    cleaned_text = normalize_document_text(
        text
    )

    if not cleaned_text:
        return []

    words = cleaned_text.split()

    step_size = (
        chunk_size
        - chunk_overlap
    )

    chunks: list[str] = []

    for start_position in range(
        0,
        len(words),
        step_size,
    ):
        end_position = (
            start_position
            + chunk_size
        )

        chunk_words = words[
            start_position:end_position
        ]

        if not chunk_words:
            continue

        chunks.append(
            " ".join(chunk_words)
        )

        if end_position >= len(words):
            break

    return chunks


def prepare_document_chunks(
    documents: pd.DataFrame,
    *,
    text_column: str = "text",
    document_id_column: str | None = None,
    metadata_columns: Sequence[str] | None = None,
    chunk_size: int = 180,
    chunk_overlap: int = 30,
) -> pd.DataFrame:
    """Convert source documents into searchable text chunks."""

    metadata_columns = list(
        metadata_columns or []
    )

    required_columns = [
        text_column,
        *metadata_columns,
    ]

    if document_id_column is not None:
        required_columns.append(
            document_id_column
        )

    validate_required_columns(
        documents,
        required_columns,
        dataframe_name="RAG documents",
    )

    chunk_rows: list[
        dict[str, object]
    ] = []

    for source_index, row in (
        documents.reset_index(
            drop=True
        ).iterrows()
    ):
        document_id = (
            row[document_id_column]
            if document_id_column is not None
            else source_index
        )

        document_text = (
            normalize_document_text(
                row[text_column]
            )
        )

        document_chunks = (
            split_text_into_chunks(
                document_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        for chunk_index, chunk_text in enumerate(
            document_chunks
        ):
            chunk_row: dict[
                str,
                object,
            ] = {
                "document_id": document_id,
                "chunk_id": (
                    f"{document_id}_{chunk_index}"
                ),
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "word_count": len(
                    chunk_text.split()
                ),
            }

            for metadata_column in (
                metadata_columns
            ):
                chunk_row[
                    metadata_column
                ] = row[
                    metadata_column
                ]

            chunk_rows.append(
                chunk_row
            )

    if not chunk_rows:
        raise ValueError(
            "No usable document chunks were generated."
        )

    return pd.DataFrame(
        chunk_rows
    )


def build_rag_corpus(
    chunks: pd.DataFrame,
    *,
    text_column: str = "chunk_text",
    metadata_columns: Sequence[str] | None = None,
    backend: str = "tfidf",
    model_name: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    ),
    maximum_features: int = 10_000,
) -> RAGCorpus:
    """Encode document chunks for semantic or lexical retrieval."""

    metadata_columns = list(
        metadata_columns or []
    )

    validate_required_columns(
        chunks,
        [
            text_column,
            *metadata_columns,
        ],
        dataframe_name="RAG chunks",
    )

    corpus_chunks = chunks.copy()

    corpus_chunks[text_column] = (
        corpus_chunks[text_column]
        .map(
            normalize_document_text
        )
    )

    corpus_chunks = (
        corpus_chunks.loc[
            corpus_chunks[text_column]
            .ne("")
        ]
        .reset_index(drop=True)
    )

    if corpus_chunks.empty:
        raise ValueError(
            "No non-empty text chunks are available."
        )

    text_values = (
        corpus_chunks[text_column]
        .astype(str)
        .tolist()
    )

    normalized_backend = (
        backend.strip().lower()
    )

    if normalized_backend == "tfidf":
        encoder = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=maximum_features,
            sublinear_tf=True,
        )

        embeddings = (
            encoder.fit_transform(
                text_values
            )
        )

    elif normalized_backend in {
        "sentence_transformers",
        "sentence-transformers",
        "semantic",
    }:
        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as error:
            raise ImportError(
                "The 'sentence-transformers' package is required "
                "for semantic embeddings. Install it with: "
                "python -m pip install sentence-transformers"
            ) from error

        encoder = SentenceTransformer(
            model_name
        )

        embeddings = encoder.encode(
            text_values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        normalized_backend = (
            "sentence_transformers"
        )

    else:
        raise ValueError(
            "backend must be either 'tfidf' or "
            "'sentence_transformers'."
        )

    return RAGCorpus(
        chunks=corpus_chunks,
        embeddings=embeddings,
        encoder=encoder,
        backend=normalized_backend,
        text_column=text_column,
        metadata_columns=metadata_columns,
    )


def encode_rag_queries(
    queries: Sequence[str],
    corpus: RAGCorpus,
) -> Any:
    """Encode search queries using the fitted corpus encoder."""

    normalized_queries = [
        normalize_document_text(
            query
        )
        for query in queries
    ]

    if corpus.backend == "tfidf":
        return corpus.encoder.transform(
            normalized_queries
        )

    return corpus.encoder.encode(
        normalized_queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def retrieve_market_evidence(
    query: str,
    corpus: RAGCorpus,
    *,
    top_k: int = 5,
    minimum_similarity: float = 0.0,
    metadata_filters: Mapping[
        str,
        object,
    ] | None = None,
    as_of_date: object | None = None,
    date_column: str = "date",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Retrieve the most relevant market-evidence chunks."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    normalized_query = (
        normalize_document_text(
            query
        )
    )

    if not normalized_query:
        raise ValueError(
            "The retrieval query cannot be empty."
        )

    candidate_mask = pd.Series(
        True,
        index=corpus.chunks.index,
    )

    if metadata_filters:
        for column, expected_value in (
            metadata_filters.items()
        ):
            if column not in corpus.chunks.columns:
                raise KeyError(
                    "Metadata filter column was not found: "
                    f"{column}"
                )

            candidate_mask &= (
                corpus.chunks[column]
                == expected_value
            )

    if as_of_date is not None:
        if date_column not in corpus.chunks.columns:
            raise KeyError(
                "Temporal retrieval requested but corpus date column was "
                f"not found: {date_column}"
            )

        query_date = pd.Timestamp(as_of_date)
        corpus_dates = pd.to_datetime(
            corpus.chunks[date_column], errors="coerce"
        )
        candidate_mask &= corpus_dates.notna() & corpus_dates.le(query_date)

        if lookback_days is not None:
            if lookback_days <= 0:
                raise ValueError("lookback_days must be greater than zero")
            candidate_mask &= corpus_dates.ge(
                query_date - pd.Timedelta(days=lookback_days)
            )

    candidate_indices = (
        corpus.chunks.index[
            candidate_mask
        ].to_numpy()
    )

    if len(candidate_indices) == 0:
        return pd.DataFrame(
            columns=[
                *corpus.chunks.columns,
                "similarity_score",
                "retrieval_rank",
            ]
        )

    query_embedding = encode_rag_queries(
        [normalized_query],
        corpus,
    )

    candidate_embeddings = (
        corpus.embeddings[
            candidate_indices
        ]
    )

    similarity_scores = (
        cosine_similarity(
            query_embedding,
            candidate_embeddings,
        )[0]
    )

    retrieval_results = (
        corpus.chunks.loc[
            candidate_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    retrieval_results[
        "similarity_score"
    ] = similarity_scores

    retrieval_results = (
        retrieval_results.loc[
            retrieval_results[
                "similarity_score"
            ] >= minimum_similarity
        ]
        .sort_values(
            "similarity_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    retrieval_results[
        "retrieval_rank"
    ] = np.arange(
        1,
        len(retrieval_results) + 1,
    )

    return retrieval_results


def retrieve_batch_evidence(
    query_data: pd.DataFrame,
    corpus: RAGCorpus,
    *,
    query_column: str,
    query_id_column: str | None = None,
    top_k: int = 5,
    minimum_similarity: float = 0.0,
    metadata_filter_columns: Sequence[str] | None = None,
    query_date_column: str | None = None,
    corpus_date_column: str = "date",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Retrieve supporting evidence for multiple queries."""

    metadata_filter_columns = list(metadata_filter_columns or [])
    required_columns = [query_column, *metadata_filter_columns]

    if query_id_column is not None:
        required_columns.append(
            query_id_column
        )

    if query_date_column is not None:
        required_columns.append(query_date_column)

    validate_required_columns(
        query_data,
        required_columns,
        dataframe_name="RAG queries",
    )

    all_results: list[
        pd.DataFrame
    ] = []

    for row_index, row in (
        query_data.reset_index(
            drop=True
        ).iterrows()
    ):
        query_id = (
            row[query_id_column]
            if query_id_column is not None
            else row_index
        )

        query_text = (
            normalize_document_text(
                row[query_column]
            )
        )

        if not query_text:
            continue

        result = retrieve_market_evidence(
            query_text,
            corpus,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
            metadata_filters={
                column: row[column]
                for column in metadata_filter_columns
                if pd.notna(row[column])
            },
            as_of_date=(
                row[query_date_column]
                if query_date_column is not None
                else None
            ),
            date_column=corpus_date_column,
            lookback_days=lookback_days,
        )

        if result.empty:
            continue

        result.insert(
            0,
            "query_id",
            query_id,
        )

        result.insert(
            1,
            "query_text",
            query_text,
        )

        all_results.append(
            result
        )

    if not all_results:
        return pd.DataFrame(
            columns=[
                "query_id",
                "query_text",
                *corpus.chunks.columns,
                "similarity_score",
                "retrieval_rank",
            ]
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def build_retail_rag_query(
    row: Mapping[str, object],
    *,
    query_columns: Sequence[str],
) -> str:
    """Build a descriptive retrieval query from a retail row."""

    query_parts: list[str] = []

    for column in query_columns:
        value = row.get(
            column
        )

        if value is None or pd.isna(value):
            continue

        normalized_value = (
            normalize_document_text(
                value
            )
        )

        if normalized_value:
            readable_column = (
                column.replace(
                    "_",
                    " ",
                )
            )

            query_parts.append(
                f"{readable_column}: "
                f"{normalized_value}"
            )

    return ". ".join(
        query_parts
    )


def count_keyword_signals(
    text: str,
    keywords: Sequence[str],
) -> int:
    """Count case-insensitive keyword and phrase occurrences."""

    normalized_text = (
        normalize_document_text(
            text
        ).lower()
    )

    count = 0

    for keyword in keywords:
        pattern = re.escape(
            keyword.lower()
        )

        count += len(
            re.findall(
                pattern,
                normalized_text,
            )
        )

    return count


def extract_market_signal_features(
    evidence_text: str,
) -> dict[str, float | int]:
    """Extract transparent market signals from retrieved evidence."""

    keyword_groups = {
        "demand_growth_signal_count": [
            "demand growth",
            "demand increased",
            "rising demand",
            "strong demand",
            "sales growth",
            "growing market",
            "higher consumption",
        ],
        "demand_decline_signal_count": [
            "demand decline",
            "demand decreased",
            "falling demand",
            "weak demand",
            "sales decline",
            "lower consumption",
            "market contraction",
        ],
        "inflation_signal_count": [
            "inflation",
            "price pressure",
            "higher costs",
            "cost increase",
            "rising costs",
        ],
        "supply_risk_signal_count": [
            "supply shortage",
            "supply disruption",
            "limited supply",
            "stockout",
            "inventory shortage",
            "logistics disruption",
        ],
        "promotion_signal_count": [
            "promotion",
            "discount",
            "coupon",
            "markdown",
            "special offer",
            "deal",
        ],
        "competition_signal_count": [
            "competitor",
            "competition",
            "competitive pricing",
            "market share",
            "price war",
        ],
        "seasonality_signal_count": [
            "seasonal",
            "holiday",
            "summer",
            "winter",
            "weekend",
            "festive",
        ],
        "premium_signal_count": [
            "premium",
            "high-end",
            "luxury",
            "quality preference",
            "brand loyalty",
        ],
        "value_signal_count": [
            "value",
            "affordable",
            "budget",
            "low price",
            "price sensitive",
            "cost conscious",
        ],
    }

    features = {
        feature_name: (
            count_keyword_signals(
                evidence_text,
                keywords,
            )
        )
        for feature_name, keywords in (
            keyword_groups.items()
        )
    }

    growth_count = (
        features[
            "demand_growth_signal_count"
        ]
    )

    decline_count = (
        features[
            "demand_decline_signal_count"
        ]
    )

    features["net_demand_signal"] = (
        growth_count
        - decline_count
    )

    features[
        "evidence_character_count"
    ] = len(
        normalize_document_text(
            evidence_text
        )
    )

    features[
        "evidence_word_count"
    ] = len(
        normalize_document_text(
            evidence_text
        ).split()
    )

    return features


def aggregate_retrieval_features(
    retrieved_evidence: pd.DataFrame,
    *,
    query_id_column: str = "query_id",
    text_column: str = "chunk_text",
) -> pd.DataFrame:
    """Aggregate retrieved chunks into row-level RAG features."""

    validate_required_columns(
        retrieved_evidence,
        [
            query_id_column,
            text_column,
            "similarity_score",
        ],
        dataframe_name="retrieved evidence",
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    grouped_evidence = (
        retrieved_evidence.groupby(
            query_id_column,
            dropna=False,
            sort=False,
        )
    )

    for query_id, group_data in (
        grouped_evidence
    ):
        ordered_group = (
            group_data.sort_values(
                "similarity_score",
                ascending=False,
            )
        )

        combined_evidence = " ".join(
            ordered_group[
                text_column
            ]
            .astype(str)
            .tolist()
        )

        signal_features = (
            extract_market_signal_features(
                combined_evidence
            )
        )

        if "impact_score" in ordered_group.columns:
            impacts = pd.to_numeric(
                ordered_group["impact_score"], errors="coerce"
            )
            similarities = pd.to_numeric(
                ordered_group["similarity_score"], errors="coerce"
            ).clip(lower=0)
            valid_impact = impacts.notna() & similarities.notna()

            if valid_impact.any() and similarities.loc[valid_impact].sum() > 0:
                weighted_impact = float(
                    np.average(
                        impacts.loc[valid_impact],
                        weights=similarities.loc[valid_impact],
                    )
                )
            elif valid_impact.any():
                weighted_impact = float(impacts.loc[valid_impact].mean())
            else:
                weighted_impact = 0.0
        else:
            weighted_impact = 0.0

        summary_rows.append(
            {
                query_id_column: query_id,
                "rag_evidence_count": int(
                    len(ordered_group)
                ),
                "rag_max_similarity": float(
                    ordered_group[
                        "similarity_score"
                    ].max()
                ),
                "rag_mean_similarity": float(
                    ordered_group[
                        "similarity_score"
                    ].mean()
                ),
                "rag_min_similarity": float(
                    ordered_group[
                        "similarity_score"
                    ].min()
                ),
                "rag_combined_evidence": (
                    combined_evidence
                ),
                "rag_weighted_impact_score": weighted_impact,
                **signal_features,
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def build_rag_features(
    dataframe: pd.DataFrame,
    corpus: RAGCorpus,
    *,
    query_columns: Sequence[str],
    top_k: int = 5,
    minimum_similarity: float = 0.0,
    metadata_filter_columns: Sequence[str] | None = None,
    query_date_column: str | None = None,
    corpus_date_column: str = "date",
    lookback_days: int | None = None,
) -> RAGFeatureBundle:
    """Retrieve evidence and attach aggregated RAG features."""

    validate_required_columns(
        dataframe,
        query_columns,
        dataframe_name="retail RAG data",
    )

    retail_data = (
        dataframe.copy()
        .reset_index(drop=True)
    )

    retail_data[
        "_rag_query_id"
    ] = np.arange(
        len(retail_data)
    )

    retail_data[
        "_rag_query_text"
    ] = retail_data.apply(
        lambda row: build_retail_rag_query(
            row,
            query_columns=query_columns,
        ),
        axis=1,
    )

    retrieved_evidence = (
        retrieve_batch_evidence(
            retail_data,
            corpus,
            query_column=(
                "_rag_query_text"
            ),
            query_id_column=(
                "_rag_query_id"
            ),
            top_k=top_k,
            minimum_similarity=minimum_similarity,
            metadata_filter_columns=metadata_filter_columns,
            query_date_column=query_date_column,
            corpus_date_column=corpus_date_column,
            lookback_days=lookback_days,
        )
    )

    if retrieved_evidence.empty:
        enriched_data = (
            retail_data.copy()
        )

        enriched_data[
            "rag_evidence_count"
        ] = 0

        enriched_data[
            "rag_max_similarity"
        ] = 0.0

        enriched_data[
            "rag_mean_similarity"
        ] = 0.0

        enriched_data[
            "rag_weighted_impact_score"
        ] = 0.0

    else:
        aggregated_features = (
            aggregate_retrieval_features(
                retrieved_evidence,
                query_id_column=(
                    "query_id"
                ),
                text_column=(
                    corpus.text_column
                ),
            )
        )

        aggregated_features = (
            aggregated_features.rename(
                columns={
                    "query_id": (
                        "_rag_query_id"
                    )
                }
            )
        )

        enriched_data = retail_data.merge(
            aggregated_features,
            on="_rag_query_id",
            how="left",
            validate="one_to_one",
        )

        numeric_rag_columns = [
            column
            for column in (
                aggregated_features.columns
            )
            if (
                column
                != "_rag_query_id"
                and pd.api.types.is_numeric_dtype(
                    aggregated_features[
                        column
                    ]
                )
            )
        ]

        enriched_data[
            numeric_rag_columns
        ] = enriched_data[
            numeric_rag_columns
        ].fillna(0)

    enriched_data = (
        enriched_data.drop(
            columns=[
                "_rag_query_id",
                "_rag_query_text",
            ]
        )
    )

    return RAGFeatureBundle(
        enriched_data=enriched_data,
        retrieved_evidence=(
            retrieved_evidence
        ),
        corpus=corpus,
    )
