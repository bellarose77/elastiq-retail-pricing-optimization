"""Unit tests for src/models/rag.py

Note: tests/test_regressions.py::TestTemporalIntegrity already exercises
retrieve_market_evidence (via prepare_document_chunks + build_rag_corpus) for
the specific scenario of future-dated and wrong-region documents being
excluded. This file intentionally avoids re-testing that exact scenario and
instead focuses on the functions that previously had zero dedicated
coverage, plus different edge cases (empty corpus, no matches, multiple
categories, boundary dates, top_k/threshold limits) for the functions that
already had partial indirect coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rag import (
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


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_chunks_frame():
    """A small, hand-authored set of pre-chunked market documents."""

    return pd.DataFrame(
        {
            "chunk_text": [
                "strong demand growth in the beverage category this quarter",
                "rising costs due to inflation pressure on suppliers",
                "competitor price war is affecting market share in grocery",
                "seasonal holiday shopping boosts grocery sales",
                "supply shortage delays household inventory restocking",
            ],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-10",
                    "2026-01-15",
                    "2026-01-20",
                ]
            ),
            "category_key": [
                "beverage",
                "beverage",
                "grocery",
                "grocery",
                "household",
            ],
            "region": ["East", "East", "West", "West", "East"],
        }
    )


@pytest.fixture(scope="module")
def sample_corpus(sample_chunks_frame):
    """A TF-IDF-backed corpus built from ``sample_chunks_frame``."""

    return build_rag_corpus(
        sample_chunks_frame,
        metadata_columns=["date", "category_key", "region"],
    )


# ---------------------------------------------------------------------------
# normalize_document_text
# ---------------------------------------------------------------------------


class TestNormalizeDocumentText:
    def test_none_returns_empty_string(self):
        assert normalize_document_text(None) == ""

    def test_nan_returns_empty_string(self):
        assert normalize_document_text(np.nan) == ""

    def test_collapses_internal_whitespace(self):
        result = normalize_document_text("hello    world\n\tfoo")
        assert result == "hello world foo"

    def test_strips_leading_and_trailing_whitespace(self):
        result = normalize_document_text("   padded text   ")
        assert result == "padded text"

    def test_non_string_value_is_stringified(self):
        assert normalize_document_text(42) == "42"

    def test_already_clean_text_is_unchanged(self):
        assert normalize_document_text("clean text") == "clean text"

    def test_empty_string_returns_empty_string(self):
        assert normalize_document_text("") == ""


# ---------------------------------------------------------------------------
# split_text_into_chunks
# ---------------------------------------------------------------------------


class TestSplitTextIntoChunks:
    def test_empty_text_returns_empty_list(self):
        assert split_text_into_chunks("") == []

    def test_whitespace_only_text_returns_empty_list(self):
        assert split_text_into_chunks("   \n\t  ") == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        chunks = split_text_into_chunks(
            "hello world", chunk_size=180, chunk_overlap=30
        )
        assert chunks == ["hello world"]

    def test_chunk_boundaries_and_overlap_are_correct(self):
        text = "a b c d e f g h i j"

        chunks = split_text_into_chunks(text, chunk_size=5, chunk_overlap=2)

        assert chunks == [
            "a b c d e",
            "d e f g h",
            "g h i j",
        ]

    def test_first_chunk_has_exactly_chunk_size_words(self):
        text = " ".join(f"w{i}" for i in range(20))

        chunks = split_text_into_chunks(text, chunk_size=6, chunk_overlap=1)

        assert len(chunks[0].split()) == 6

    def test_last_chunk_is_not_dropped_when_shorter_than_chunk_size(self):
        text = "a b c d e f g h i j"

        chunks = split_text_into_chunks(text, chunk_size=4, chunk_overlap=1)

        last_chunk_words = chunks[-1].split()
        assert len(last_chunk_words) <= 4
        assert last_chunk_words[-1] == "j"

    def test_consecutive_chunks_share_overlap_words(self):
        text = " ".join(f"w{i}" for i in range(30))

        chunks = split_text_into_chunks(text, chunk_size=10, chunk_overlap=3)

        first_words = chunks[0].split()
        second_words = chunks[1].split()

        assert first_words[-3:] == second_words[:3]

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            split_text_into_chunks("some text", chunk_size=0, chunk_overlap=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            split_text_into_chunks("some text", chunk_size=-5, chunk_overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=20)

    def test_zero_overlap_produces_non_overlapping_chunks(self):
        text = " ".join(f"w{i}" for i in range(9))

        chunks = split_text_into_chunks(text, chunk_size=3, chunk_overlap=0)

        assert chunks == ["w0 w1 w2", "w3 w4 w5", "w6 w7 w8"]


# ---------------------------------------------------------------------------
# prepare_document_chunks / build_rag_corpus -- edge cases only
# (temporal/region filtering itself is covered by test_regressions.py)
# ---------------------------------------------------------------------------


class TestPrepareDocumentChunksEdgeCases:
    def test_multiple_documents_get_independent_chunk_ids_and_metadata(self):
        documents = pd.DataFrame(
            {
                "doc_id": ["D1", "D2"],
                "text": [
                    "alpha beta gamma delta epsilon",
                    "zeta eta theta iota kappa",
                ],
                "category": ["grocery", "household"],
            }
        )

        chunks = prepare_document_chunks(
            documents,
            text_column="text",
            document_id_column="doc_id",
            metadata_columns=["category"],
            chunk_size=3,
            chunk_overlap=1,
        )

        d1_chunks = chunks.loc[chunks["document_id"] == "D1"]
        d2_chunks = chunks.loc[chunks["document_id"] == "D2"]

        assert set(d1_chunks["category"]) == {"grocery"}
        assert set(d2_chunks["category"]) == {"household"}
        assert list(d1_chunks["chunk_id"])[0] == "D1_0"
        assert list(d2_chunks["chunk_id"])[0] == "D2_0"

    def test_all_blank_documents_raise_value_error(self):
        documents = pd.DataFrame({"text": ["   ", ""]})

        with pytest.raises(ValueError, match="No usable document chunks"):
            prepare_document_chunks(documents, text_column="text")

    def test_missing_required_column_raises(self):
        documents = pd.DataFrame({"body": ["some text"]})

        with pytest.raises(ValueError, match="missing required columns"):
            prepare_document_chunks(documents, text_column="text")


class TestBuildRagCorpusEdgeCases:
    def test_all_empty_chunks_raise_value_error(self):
        chunks = pd.DataFrame({"chunk_text": ["", "   ", None]})

        with pytest.raises(ValueError, match="No non-empty text chunks"):
            build_rag_corpus(chunks)

    def test_invalid_backend_raises(self):
        chunks = pd.DataFrame({"chunk_text": ["some real text here"]})

        with pytest.raises(ValueError, match="backend"):
            build_rag_corpus(chunks, backend="not_a_real_backend")

    def test_rows_with_blank_text_are_dropped_from_the_corpus(self, sample_chunks_frame):
        chunks = pd.concat(
            [
                sample_chunks_frame,
                pd.DataFrame(
                    {
                        "chunk_text": [""],
                        "date": pd.to_datetime(["2026-02-01"]),
                        "category_key": ["grocery"],
                        "region": ["East"],
                    }
                ),
            ],
            ignore_index=True,
        )

        corpus = build_rag_corpus(
            chunks, metadata_columns=["date", "category_key", "region"]
        )

        assert len(corpus.chunks) == len(sample_chunks_frame)

    def test_tfidf_backend_is_recorded_on_corpus(self, sample_corpus):
        assert sample_corpus.backend == "tfidf"
        assert sample_corpus.text_column == "chunk_text"


# ---------------------------------------------------------------------------
# encode_rag_queries
# ---------------------------------------------------------------------------


class TestEncodeRagQueries:
    def test_output_has_one_row_per_query(self, sample_corpus):
        embeddings = encode_rag_queries(
            ["demand growth", "competitor pricing"], sample_corpus
        )
        assert embeddings.shape[0] == 2

    def test_single_query_shape(self, sample_corpus):
        embeddings = encode_rag_queries(["inflation pressure"], sample_corpus)
        assert embeddings.shape[0] == 1

    def test_empty_query_string_does_not_raise(self, sample_corpus):
        embeddings = encode_rag_queries([""], sample_corpus)
        assert embeddings.shape[0] == 1
        assert embeddings.toarray().sum() == 0

    def test_empty_list_of_queries_raises(self, sample_corpus):
        # The TF-IDF transformer requires at least one sample to transform.
        with pytest.raises(ValueError):
            encode_rag_queries([], sample_corpus)


# ---------------------------------------------------------------------------
# retrieve_market_evidence -- edge cases beyond the temporal-integrity test
# ---------------------------------------------------------------------------


class TestRetrieveMarketEvidenceEdgeCases:
    def test_top_k_limits_number_of_results(self, sample_corpus):
        result = retrieve_market_evidence(
            "demand growth costs shortage market", sample_corpus, top_k=2
        )
        assert len(result) <= 2

    def test_top_k_zero_raises(self, sample_corpus):
        with pytest.raises(ValueError, match="top_k"):
            retrieve_market_evidence("demand", sample_corpus, top_k=0)

    def test_blank_query_raises(self, sample_corpus):
        with pytest.raises(ValueError, match="query"):
            retrieve_market_evidence("   ", sample_corpus, top_k=5)

    def test_unknown_metadata_filter_column_raises_key_error(self, sample_corpus):
        with pytest.raises(KeyError):
            retrieve_market_evidence(
                "demand growth",
                sample_corpus,
                metadata_filters={"not_a_real_column": "x"},
            )

    def test_contradictory_filters_return_empty_dataframe_with_expected_columns(
        self, sample_corpus
    ):
        result = retrieve_market_evidence(
            "demand growth",
            sample_corpus,
            metadata_filters={"category_key": "beverage", "region": "West"},
        )

        assert result.empty
        assert list(result.columns) == [
            *sample_corpus.chunks.columns,
            "similarity_score",
            "retrieval_rank",
        ]

    def test_as_of_date_exactly_on_document_date_is_included(self, sample_corpus):
        result = retrieve_market_evidence(
            "beverage demand growth",
            sample_corpus,
            top_k=10,
            metadata_filters={"category_key": "beverage"},
            as_of_date="2026-01-05",
        )

        assert (pd.to_datetime(result["date"]) <= pd.Timestamp("2026-01-05")).all()
        # the document dated exactly 2026-01-05 must be retrievable (le, not lt)
        assert pd.Timestamp("2026-01-05") in pd.to_datetime(result["date"]).values

    def test_minimum_similarity_threshold_can_exclude_everything(self, sample_corpus):
        result = retrieve_market_evidence(
            "completely unrelated gibberish zzz qqq",
            sample_corpus,
            top_k=10,
            minimum_similarity=0.99,
        )
        assert result.empty

    def test_lookback_days_excludes_documents_before_the_window(self, sample_corpus):
        result = retrieve_market_evidence(
            "market evidence",
            sample_corpus,
            top_k=10,
            as_of_date="2026-01-20",
            lookback_days=5,
        )

        dates = pd.to_datetime(result["date"])
        assert (dates >= pd.Timestamp("2026-01-15")).all()
        assert (dates <= pd.Timestamp("2026-01-20")).all()

    def test_lookback_days_boundary_is_inclusive(self, sample_corpus):
        result = retrieve_market_evidence(
            "holiday grocery sales",
            sample_corpus,
            top_k=10,
            metadata_filters={"category_key": "grocery"},
            as_of_date="2026-01-20",
            lookback_days=5,
        )
        # 2026-01-15 is exactly 5 days before 2026-01-20 and must be included
        assert pd.Timestamp("2026-01-15") in pd.to_datetime(result["date"]).values

    def test_lookback_days_zero_or_negative_raises(self, sample_corpus):
        with pytest.raises(ValueError, match="lookback_days"):
            retrieve_market_evidence(
                "demand",
                sample_corpus,
                as_of_date="2026-01-20",
                lookback_days=0,
            )

    def test_retrieval_rank_is_one_indexed_and_ordered_by_similarity(
        self, sample_corpus
    ):
        result = retrieve_market_evidence(
            "beverage demand growth strong", sample_corpus, top_k=10
        )
        assert list(result["retrieval_rank"]) == list(range(1, len(result) + 1))
        assert result["similarity_score"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# build_retail_rag_query
# ---------------------------------------------------------------------------


class TestBuildRetailRagQuery:
    def test_builds_readable_query_from_row(self):
        row = {"category": "Beverage", "promo_notes": "BOGO deal"}
        result = build_retail_rag_query(row, query_columns=["category", "promo_notes"])
        assert result == "category: Beverage. promo notes: BOGO deal"

    def test_none_and_nan_values_are_skipped(self):
        row = {"category": "Beverage", "region": None, "notes": np.nan}
        result = build_retail_rag_query(
            row, query_columns=["category", "region", "notes"]
        )
        assert result == "category: Beverage"

    def test_missing_columns_are_skipped(self):
        row = {"category": "Grocery"}
        result = build_retail_rag_query(row, query_columns=["category", "missing_col"])
        assert result == "category: Grocery"

    def test_no_usable_columns_returns_empty_string(self):
        row = {"category": None}
        result = build_retail_rag_query(row, query_columns=["category"])
        assert result == ""

    def test_empty_query_columns_returns_empty_string(self):
        row = {"category": "Grocery"}
        result = build_retail_rag_query(row, query_columns=[])
        assert result == ""

    def test_works_with_pandas_series_rows(self):
        row = pd.Series({"category": "Household", "region": "East"})
        result = build_retail_rag_query(row, query_columns=["category", "region"])
        assert result == "category: Household. region: East"

    def test_underscore_column_names_become_readable(self):
        row = {"unit_cost_notes": "rising supplier costs"}
        result = build_retail_rag_query(row, query_columns=["unit_cost_notes"])
        assert result == "unit cost notes: rising supplier costs"


# ---------------------------------------------------------------------------
# count_keyword_signals
# ---------------------------------------------------------------------------


class TestCountKeywordSignals:
    def test_counts_case_insensitive_matches(self):
        text = "Demand is up. DEMAND keeps rising. demand demand."
        assert count_keyword_signals(text, ["demand"]) == 4

    def test_sums_multiple_keywords(self):
        text = "strong demand growth alongside rising costs"
        count = count_keyword_signals(text, ["demand growth", "rising costs"])
        assert count == 2

    def test_no_match_returns_zero(self):
        text = "nothing relevant is mentioned here"
        assert count_keyword_signals(text, ["inflation", "shortage"]) == 0

    def test_empty_keyword_list_returns_zero(self):
        text = "some text with demand growth in it"
        assert count_keyword_signals(text, []) == 0

    def test_empty_text_returns_zero(self):
        assert count_keyword_signals("", ["demand"]) == 0

    def test_matches_are_substrings_not_word_bounded(self):
        # "value" is found both as its own word and as a substring inside
        # "overvalued" -- count_keyword_signals does not use word boundaries.
        text = "The stock is overvalued despite value pricing"
        assert count_keyword_signals(text, ["value"]) == 2

    def test_phrase_keyword_requires_full_phrase(self):
        text = "there is demand, but no growth to speak of"
        # "demand" and "growth" both appear, but not as the phrase
        # "demand growth", so this must not count as a match.
        assert count_keyword_signals(text, ["demand growth"]) == 0


# ---------------------------------------------------------------------------
# extract_market_signal_features
# ---------------------------------------------------------------------------


class TestExtractMarketSignalFeatures:
    def test_growth_signals_are_detected(self):
        text = (
            "There was strong demand and demand growth was up, "
            "with rising demand across regions."
        )
        features = extract_market_signal_features(text)
        assert features["demand_growth_signal_count"] == 3
        assert features["demand_decline_signal_count"] == 0
        assert features["net_demand_signal"] == 3

    def test_decline_signals_are_detected(self):
        text = (
            "weak demand and demand decline hit the market alongside "
            "falling demand this month"
        )
        features = extract_market_signal_features(text)
        assert features["demand_decline_signal_count"] == 3
        assert features["demand_growth_signal_count"] == 0
        assert features["net_demand_signal"] == -3

    def test_net_demand_signal_is_growth_minus_decline(self):
        text = "demand growth was seen, but also demand decline elsewhere"
        features = extract_market_signal_features(text)
        assert features["net_demand_signal"] == (
            features["demand_growth_signal_count"]
            - features["demand_decline_signal_count"]
        )

    def test_character_and_word_counts_match_normalized_text(self):
        text = "  hello   world  "
        features = extract_market_signal_features(text)
        assert features["evidence_character_count"] == len("hello world")
        assert features["evidence_word_count"] == 2

    def test_empty_text_gives_all_zero_counts(self):
        features = extract_market_signal_features("")
        for key, value in features.items():
            assert value == 0, f"{key} expected 0, got {value}"

    def test_all_expected_signal_keys_are_present(self):
        features = extract_market_signal_features("some evidence text")
        expected_keys = {
            "demand_growth_signal_count",
            "demand_decline_signal_count",
            "inflation_signal_count",
            "supply_risk_signal_count",
            "promotion_signal_count",
            "competition_signal_count",
            "seasonality_signal_count",
            "premium_signal_count",
            "value_signal_count",
            "net_demand_signal",
            "evidence_character_count",
            "evidence_word_count",
        }
        assert expected_keys.issubset(features.keys())

    def test_promotion_and_supply_signals_counted_independently(self):
        text = "a big discount and coupon offer, plus a supply shortage warning"
        features = extract_market_signal_features(text)
        assert features["promotion_signal_count"] == 2
        assert features["supply_risk_signal_count"] == 1


# ---------------------------------------------------------------------------
# retrieve_batch_evidence
# ---------------------------------------------------------------------------


class TestRetrieveBatchEvidence:
    def test_matching_rows_each_produce_a_result_group(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["Q1", "Q2"],
                "query_text": [
                    "beverage demand growth",
                    "competitor price war grocery",
                ],
            }
        )

        result = retrieve_batch_evidence(
            query_data, sample_corpus, query_column="query_text", query_id_column="query_id"
        )

        assert set(result["query_id"]) == {"Q1", "Q2"}

    def test_row_with_blank_query_text_is_skipped(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["Q1", "Q2"],
                "query_text": ["beverage demand growth", "   "],
            }
        )

        result = retrieve_batch_evidence(
            query_data, sample_corpus, query_column="query_text", query_id_column="query_id"
        )

        assert "Q2" not in set(result["query_id"])
        assert "Q1" in set(result["query_id"])

    def test_row_whose_filters_match_nothing_is_absent_from_output(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["Q1", "Q2"],
                "query_text": ["beverage demand growth", "some grocery query"],
                "category_key": ["beverage", "not_a_real_category"],
            }
        )

        result = retrieve_batch_evidence(
            query_data,
            sample_corpus,
            query_column="query_text",
            query_id_column="query_id",
            metadata_filter_columns=["category_key"],
        )

        assert "Q1" in set(result["query_id"])
        assert "Q2" not in set(result["query_id"])

    def test_per_row_metadata_filters_do_not_leak_across_rows(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["Q_BEV", "Q_GRO"],
                "query_text": ["market evidence", "market evidence"],
                "category_key": ["beverage", "grocery"],
            }
        )

        result = retrieve_batch_evidence(
            query_data,
            sample_corpus,
            query_column="query_text",
            query_id_column="query_id",
            top_k=10,
            metadata_filter_columns=["category_key"],
        )

        bev_categories = set(
            result.loc[result["query_id"] == "Q_BEV", "category_key"]
        )
        gro_categories = set(
            result.loc[result["query_id"] == "Q_GRO", "category_key"]
        )

        assert bev_categories == {"beverage"}
        assert gro_categories == {"grocery"}

    def test_per_row_date_filters_are_independent(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["EARLY", "LATE"],
                "query_text": ["market evidence", "market evidence"],
                "as_of": pd.to_datetime(["2026-01-06", "2026-01-31"]),
            }
        )

        result = retrieve_batch_evidence(
            query_data,
            sample_corpus,
            query_column="query_text",
            query_id_column="query_id",
            top_k=10,
            query_date_column="as_of",
            corpus_date_column="date",
        )

        early_dates = pd.to_datetime(
            result.loc[result["query_id"] == "EARLY", "date"]
        )
        late_dates = pd.to_datetime(
            result.loc[result["query_id"] == "LATE", "date"]
        )

        assert (early_dates <= pd.Timestamp("2026-01-06")).all()
        # the later row can see documents the earlier row cannot
        assert late_dates.max() > early_dates.max()

    def test_default_query_id_uses_row_position(self, sample_corpus):
        query_data = pd.DataFrame(
            {"query_text": ["beverage demand growth", "competitor price war"]}
        )

        result = retrieve_batch_evidence(
            query_data, sample_corpus, query_column="query_text"
        )

        assert set(result["query_id"]) <= {0, 1}

    def test_all_rows_unmatched_returns_empty_dataframe_with_expected_columns(
        self, sample_corpus
    ):
        query_data = pd.DataFrame({"query_text": ["", "   "]})

        result = retrieve_batch_evidence(
            query_data, sample_corpus, query_column="query_text"
        )

        assert result.empty
        assert list(result.columns) == [
            "query_id",
            "query_text",
            *sample_corpus.chunks.columns,
            "similarity_score",
            "retrieval_rank",
        ]

    def test_missing_query_column_raises(self, sample_corpus):
        query_data = pd.DataFrame({"other_column": ["text"]})

        with pytest.raises(ValueError, match="missing required columns"):
            retrieve_batch_evidence(
                query_data, sample_corpus, query_column="query_text"
            )

    def test_top_k_applies_per_row(self, sample_corpus):
        query_data = pd.DataFrame(
            {
                "query_id": ["Q1", "Q2"],
                "query_text": [
                    "market evidence demand growth",
                    "market evidence competitor",
                ],
            }
        )

        result = retrieve_batch_evidence(
            query_data,
            sample_corpus,
            query_column="query_text",
            query_id_column="query_id",
            top_k=1,
        )

        counts = result.groupby("query_id").size()
        assert (counts <= 1).all()


# ---------------------------------------------------------------------------
# aggregate_retrieval_features
# ---------------------------------------------------------------------------


class TestAggregateRetrievalFeatures:
    def test_aggregates_one_row_per_query_id(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1", "Q2"],
                "chunk_text": ["alpha text", "beta text", "gamma text"],
                "similarity_score": [0.3, 0.9, 0.5],
            }
        )

        result = aggregate_retrieval_features(evidence)

        assert set(result["query_id"]) == {"Q1", "Q2"}
        assert len(result) == 2

    def test_similarity_summary_statistics_are_correct(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1"],
                "chunk_text": ["alpha", "beta"],
                "similarity_score": [0.3, 0.9],
            }
        )

        result = aggregate_retrieval_features(evidence)
        row = result.loc[result["query_id"] == "Q1"].iloc[0]

        assert row["rag_evidence_count"] == 2
        assert row["rag_max_similarity"] == pytest.approx(0.9)
        assert row["rag_min_similarity"] == pytest.approx(0.3)
        assert row["rag_mean_similarity"] == pytest.approx(0.6)

    def test_combined_evidence_is_ordered_by_similarity_descending(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1"],
                "chunk_text": ["alpha text", "beta text"],
                "similarity_score": [0.3, 0.9],
            }
        )

        result = aggregate_retrieval_features(evidence)
        row = result.loc[result["query_id"] == "Q1"].iloc[0]

        assert row["rag_combined_evidence"] == "beta text alpha text"

    def test_weighted_impact_score_uses_similarity_as_weight(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1"],
                "chunk_text": ["a", "b"],
                "similarity_score": [0.2, 0.8],
                "impact_score": [10.0, 20.0],
            }
        )

        result = aggregate_retrieval_features(evidence)
        row = result.loc[result["query_id"] == "Q1"].iloc[0]

        expected = (10.0 * 0.2 + 20.0 * 0.8) / (0.2 + 0.8)
        assert row["rag_weighted_impact_score"] == pytest.approx(expected)

    def test_missing_impact_score_column_defaults_to_zero(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1"],
                "chunk_text": ["alpha"],
                "similarity_score": [0.5],
            }
        )

        result = aggregate_retrieval_features(evidence)
        assert result.iloc[0]["rag_weighted_impact_score"] == 0.0

    def test_signal_features_are_attached_per_query(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1"],
                "chunk_text": ["strong demand growth reported"],
                "similarity_score": [0.7],
            }
        )

        result = aggregate_retrieval_features(evidence)
        # "strong demand" and "demand growth" are both growth keywords and
        # both appear (overlapping) in this text, so the count is 2.
        assert result.iloc[0]["demand_growth_signal_count"] == 2

    def test_missing_similarity_score_column_raises(self):
        evidence = pd.DataFrame({"query_id": ["Q1"], "chunk_text": ["alpha"]})

        with pytest.raises(ValueError, match="missing required columns"):
            aggregate_retrieval_features(evidence)

    def test_custom_text_column_is_respected(self):
        evidence = pd.DataFrame(
            {
                "query_id": ["Q1"],
                "custom_text": ["custom evidence"],
                "similarity_score": [0.4],
            }
        )

        result = aggregate_retrieval_features(evidence, text_column="custom_text")
        assert result.iloc[0]["rag_combined_evidence"] == "custom evidence"


# ---------------------------------------------------------------------------
# build_rag_features
# ---------------------------------------------------------------------------


class TestBuildRagFeatures:
    def test_enriched_data_preserves_row_count(self, sample_corpus):
        retail_data = pd.DataFrame(
            {
                "item_id": ["I1", "I2", "I3"],
                "notes": [
                    "demand growth beverage category",
                    "competitor price war grocery",
                    "totally unrelated placeholder text",
                ],
            }
        )

        bundle = build_rag_features(
            retail_data, sample_corpus, query_columns=["notes"], top_k=3
        )

        assert len(bundle.enriched_data) == 3

    def test_internal_helper_columns_are_dropped(self, sample_corpus):
        retail_data = pd.DataFrame(
            {"item_id": ["I1"], "notes": ["demand growth beverage category"]}
        )

        bundle = build_rag_features(
            retail_data, sample_corpus, query_columns=["notes"], top_k=3
        )

        assert "_rag_query_id" not in bundle.enriched_data.columns
        assert "_rag_query_text" not in bundle.enriched_data.columns

    def test_rag_feature_columns_are_attached(self, sample_corpus):
        retail_data = pd.DataFrame(
            {"item_id": ["I1"], "notes": ["demand growth beverage category"]}
        )

        bundle = build_rag_features(
            retail_data, sample_corpus, query_columns=["notes"], top_k=3
        )

        assert "rag_evidence_count" in bundle.enriched_data.columns
        assert bundle.enriched_data.loc[0, "rag_evidence_count"] > 0

    def test_row_without_any_matching_evidence_gets_zero_numeric_defaults(
        self, sample_corpus
    ):
        retail_data = pd.DataFrame(
            {
                "item_id": ["I1", "I2"],
                "notes": [
                    "demand growth beverage category",
                    "some text for a row with no filter match",
                ],
                "category_key": ["beverage", "not_a_real_category"],
            }
        )

        bundle = build_rag_features(
            retail_data,
            sample_corpus,
            query_columns=["notes"],
            top_k=3,
            metadata_filter_columns=["category_key"],
        )

        unmatched_row = bundle.enriched_data.loc[
            bundle.enriched_data["item_id"] == "I2"
        ].iloc[0]

        assert unmatched_row["rag_evidence_count"] == 0
        assert unmatched_row["rag_max_similarity"] == 0.0

    def test_all_rows_without_evidence_uses_empty_branch_defaults(self, sample_corpus):
        retail_data = pd.DataFrame(
            {
                "item_id": ["I1", "I2"],
                "notes": ["demand growth", "competitor pricing"],
                "category_key": ["not_a_real_category", "also_not_real"],
            }
        )

        bundle = build_rag_features(
            retail_data,
            sample_corpus,
            query_columns=["notes"],
            top_k=3,
            metadata_filter_columns=["category_key"],
        )

        assert bundle.retrieved_evidence.empty
        assert (bundle.enriched_data["rag_evidence_count"] == 0).all()
        assert (bundle.enriched_data["rag_max_similarity"] == 0.0).all()
        assert (bundle.enriched_data["rag_mean_similarity"] == 0.0).all()
        assert (bundle.enriched_data["rag_weighted_impact_score"] == 0.0).all()
        # the "all empty" short-circuit branch does not set this column,
        # unlike the partially-populated branch which does.
        assert "rag_min_similarity" not in bundle.enriched_data.columns

    def test_missing_query_column_raises(self, sample_corpus):
        retail_data = pd.DataFrame({"item_id": ["I1"]})

        with pytest.raises(ValueError, match="missing required columns"):
            build_rag_features(
                retail_data, sample_corpus, query_columns=["notes"], top_k=3
            )

    def test_corpus_is_carried_through_on_the_bundle(self, sample_corpus):
        retail_data = pd.DataFrame(
            {"item_id": ["I1"], "notes": ["demand growth beverage category"]}
        )

        bundle = build_rag_features(
            retail_data, sample_corpus, query_columns=["notes"], top_k=3
        )

        assert bundle.corpus is sample_corpus


# ---------------------------------------------------------------------------
# RAGCorpus / RAGFeatureBundle dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_rag_corpus_stores_all_fields(self):
        chunks = pd.DataFrame({"chunk_text": ["a"]})
        corpus = RAGCorpus(
            chunks=chunks,
            embeddings="embeddings-placeholder",
            encoder="encoder-placeholder",
            backend="tfidf",
            text_column="chunk_text",
            metadata_columns=["date"],
        )

        assert corpus.chunks is chunks
        assert corpus.embeddings == "embeddings-placeholder"
        assert corpus.encoder == "encoder-placeholder"
        assert corpus.backend == "tfidf"
        assert corpus.text_column == "chunk_text"
        assert corpus.metadata_columns == ["date"]

    def test_rag_corpus_is_slotted(self):
        chunks = pd.DataFrame({"chunk_text": ["a"]})
        corpus = RAGCorpus(
            chunks=chunks,
            embeddings=None,
            encoder=None,
            backend="tfidf",
            text_column="chunk_text",
            metadata_columns=[],
        )

        with pytest.raises(AttributeError):
            corpus.not_a_declared_field = 1

    def test_rag_feature_bundle_stores_all_fields(self):
        enriched = pd.DataFrame({"item_id": ["I1"]})
        evidence = pd.DataFrame({"query_id": ["I1"]})
        corpus = RAGCorpus(
            chunks=pd.DataFrame({"chunk_text": ["a"]}),
            embeddings=None,
            encoder=None,
            backend="tfidf",
            text_column="chunk_text",
            metadata_columns=[],
        )

        bundle = RAGFeatureBundle(
            enriched_data=enriched, retrieved_evidence=evidence, corpus=corpus
        )

        assert bundle.enriched_data is enriched
        assert bundle.retrieved_evidence is evidence
        assert bundle.corpus is corpus
