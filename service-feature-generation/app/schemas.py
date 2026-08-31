"""Request/response models for the feature generation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemFeatures(BaseModel):
    """One (product, store) item's structured market-intelligence features.

    These come from step 07 of the batch pipeline ("Market intelligence"),
    which converts unstructured market notes into structured pricing
    features via retrieval over embedded documents -- see
    src/features/text.py and src/pipelines/step_07_rag_features.py.
    """

    item_id: str
    product_id: str
    store_id: str
    category: str
    rag_evidence_count: int
    rag_max_similarity: float
    rag_mean_similarity: float
    rag_min_similarity: float
    rag_weighted_impact_score: float
    rag_combined_evidence: str
    demand_growth_signal_count: int
    demand_decline_signal_count: int
    inflation_signal_count: int
    supply_risk_signal_count: int
    promotion_signal_count: int
    competition_signal_count: int
    seasonality_signal_count: int
    premium_signal_count: int
    value_signal_count: int
    net_demand_signal: int
    evidence_character_count: int
    evidence_word_count: int


class FeatureRequest(BaseModel):
    """A batch request for item-level features."""

    item_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "item_id values from GET /items, e.g. 'P001_S001'. Unknown "
            "ids are reported back under 'not_found' rather than failing "
            "the whole request."
        ),
    )


class FeatureResponse(BaseModel):
    """Features for a requested batch, plus any unresolved ids."""

    features: list[ItemFeatures]
    not_found: list[str]
    generated_from: str = Field(
        description="Timestamp of the upstream feature artifact used.",
    )


class ProductFeatures(BaseModel):
    """One product's market-intelligence features, aggregated across stores."""

    itemId: str
    ragEvidenceCount: int
    ragWeightedImpactScore: float
    netDemandSignal: int
    storeCount: int
