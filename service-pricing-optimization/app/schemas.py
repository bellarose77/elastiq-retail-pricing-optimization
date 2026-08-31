"""Request/response models for the pricing API.

Pure I/O contracts -- no pricing logic lives here. See
``app/services/pricing_service.py`` for the business layer these
describe the shape of.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemSummary(BaseModel):
    """One priceable (product, store) decision unit, as listed by GET /items."""

    item_id: str
    product_id: str
    store_id: str
    category: str
    selling_price: float
    elasticity_source: str
    elasticity_is_causal: bool


class RecommendationRequest(BaseModel):
    """A batch request for (product, store)-level price recommendations."""

    item_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "item_id values from GET /items, e.g. 'P001_S001'. Unknown "
            "ids are reported back under 'not_found' rather than failing "
            "the whole request."
        ),
    )


class Recommendation(BaseModel):
    """One item's pricing recommendation, mirroring the batch pipeline's

    price_optimization_recommendations.csv row shape.
    """

    item_id: str
    status: str | None = None
    recommendation_action: str | None = None
    current_price: float | None = None
    recommended_price: float | None = None
    price_change: float | None = None
    price_change_rate: float | None = None
    expected_quantity: float | None = None
    elasticity: float | None = None
    elasticity_source: str | None = None
    elasticity_is_causal: bool | None = None
    error_message: str | None = None

    model_config = {"extra": "allow"}


class RecommendationResponse(BaseModel):
    """Recommendations for a requested batch, plus any unresolved ids."""

    recommendations: list[Recommendation]
    not_found: list[str]
    generated_from: str = Field(
        description="Timestamp of the upstream pipeline artifacts used.",
    )


class ProductSummary(BaseModel):
    """One product's aggregated catalog row, one row per productId.

    Mirrors the frontend's bundled ``DEMO_DATA`` row shape exactly (see
    app/frontend/src/lib/demoData.js) so the client can swap a live fetch
    in for the offline snapshot with no field mapping. This describes
    *input* data (current price, cost, elasticity...), not a
    recommendation -- see ``ProductRecommendation`` for that.
    """

    itemId: str
    productId: str
    category: str
    currentPrice: float
    unitCost: float
    competitorPrice: float
    inventory: int
    baselineQuantity: float
    elasticity: float
    elasticitySource: str
    elasticityIsCausal: bool
    baselineSource: str
    promotionUpliftRate: float
    promotionModelReliable: bool
    confidence: float
    ragSignal: float
    promotionFlag: int
    dataAsOfDate: str


class ProductRecommendation(BaseModel):
    """One product's aggregated pricing decision, one row per productId.

    Aggregates every (product, store) recommendation for a product (see
    ``build_product_recommendation_summary``), so a caller who only knows
    the product (not which store) still gets one price to act on. Price
    change, margin rate, and action label are intentionally left for the
    caller to derive from ``recommendedPrice`` against its own known
    current price/cost, rather than this endpoint re-deriving them.
    """

    itemId: str
    status: str
    recommendedPrice: float
    expectedQuantity: float
    expectedRevenue: float
    expectedProfit: float
    storeCount: int
