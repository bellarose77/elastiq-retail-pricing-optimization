/* Patches live pricing-service recommendations onto client-computed rows.

   Why patch rather than replace: engine.js's row shape carries the
   "what if we maximized revenue/quantity instead" comparison scenarios
   (_scenarios.revenueMax/profitMax/quantityMax) that the grid technique
   computes as part of its search -- exploratory context, not a decision.
   The pricing service (service-pricing-optimization) only computes ONE
   decision per item (the configured objective's grid search, matching
   the batch pipeline's production policy), so it has no equivalent for
   those comparison points. Rather than fabricate them or drop the
   comparison feature, only the fields that represent the actual decision
   (recommended price and its outcome) are overridden with the live
   service's numbers; the "what if" columns stay locally computed.

   This only ever runs for the "grid" technique -- see App.jsx, which is
   the one technique the service implements (src/optimization/pricing.py's
   optimize_price_portfolio). The other five techniques never call this;
   their rows pass through unpatched. */

import { classifyAction, isFin } from "./engine.js";

// Mirrors default_profit_configuration() in src/optimization/dataset.py --
// the actual policy the service enforces -- so the guardrail strip and
// "within implementation step" / "minimum margin" checks read correctly
// for live-sourced rows. This never decides a price; the real
// enforcement always happens server-side. It's deliberately NOT the same
// as this app's own DEFAULT_CONFIG (see engine.js's comment there) --
// DEFAULT_CONFIG mirrors PricingOptimizationConfig's looser bare class
// defaults, a separate, tested contract this must not disturb.
const LIVE_POLICY = {
  minPriceChangeRate: -0.10,
  maxPriceChangeRate: 0.20,
  minMarginRate: 0.15,
  competitorPriceTolerance: 0.15,
};
const LIVE_IMPLEMENTATION_STEP_RATE = Math.max(
  Math.abs(LIVE_POLICY.minPriceChangeRate),
  Math.abs(LIVE_POLICY.maxPriceChangeRate)
);

/* One product's live recommendation (from GET /products/recommendations)
   -> the subset of an engine.js row's fields it authoritatively decides.
   `row` is the client-computed row for the same item (already carries
   currentPrice, unitCost, competitorPrice, category, and the
   current/what-if scenarios). */
export function applyLiveRecommendation(row, live, cfg) {
  if (!live || !isFin(live.recommendedPrice)) return row;

  const currentPrice = row.currentPrice;
  const recommendedPrice = live.recommendedPrice;
  const priceChange = recommendedPrice - currentPrice;
  const priceChangeRate = currentPrice ? priceChange / currentPrice : 0;
  const isHeld = live.status !== "success";
  const recommendationAction = isHeld ? "review" : classifyAction(priceChangeRate, cfg.actionThreshold);

  const current = row._scenarios.current;
  const expectedDemand = isFin(live.expectedQuantity) ? live.expectedQuantity : current.demand;
  const expectedRevenue = isFin(live.expectedRevenue) ? live.expectedRevenue : current.revenue;
  const expectedProfit = isFin(live.expectedProfit) ? live.expectedProfit : current.profit;
  const pct = (a, b) => (b !== 0 && isFin(b) ? (a - b) / Math.abs(b) : NaN);

  const unitMargin = recommendedPrice - row.unitCost;
  const marginRate = recommendedPrice > 0 ? unitMargin / recommendedPrice : NaN;
  const inventoryBinding = isFin(row.inventory) && row.inventory > 0 && expectedDemand > row.inventory;

  // Not returned by the service (a product-level rollup has no single
  // "was this store promoted" answer); keep the client's own estimate --
  // a secondary/diagnostic field, not part of the price decision itself.
  const promoOn = current.promoOn;

  const recommended = {
    price: recommendedPrice,
    promoOn,
    demand: expectedDemand,
    revenue: expectedRevenue,
    profit: expectedProfit,
    promoCost: current.promoCost,
    unitMargin,
    marginRate,
    inventoryBinding,
  };

  const stepLo = currentPrice * (1 + LIVE_POLICY.minPriceChangeRate);
  const stepHi = currentPrice * (1 + LIVE_POLICY.maxPriceChangeRate);
  const compLo = isFin(row.competitorPrice) ? row.competitorPrice * (1 - LIVE_POLICY.competitorPriceTolerance) : -Infinity;
  const compHi = isFin(row.competitorPrice) ? row.competitorPrice * (1 + LIVE_POLICY.competitorPriceTolerance) : Infinity;
  const marginFloor = Math.max(row.unitCost / (1 - LIVE_POLICY.minMarginRate), row.unitCost);

  return {
    ...row,
    status: live.status,
    recommendedPrice,
    priceChange,
    priceChangeRate,
    priceChangePct: priceChangeRate * 100,
    recommendationAction,
    expectedDemand,
    expectedRevenue,
    expectedProfit,
    optimizedMarginRate: marginRate,
    inventoryBinding,
    demandChange: expectedDemand - current.demand,
    demandChangeRate: pct(expectedDemand, current.demand),
    revenueChange: expectedRevenue - current.revenue,
    revenueChangeRate: pct(expectedRevenue, current.revenue),
    profitChange: expectedProfit - current.profit,
    profitChangeRate: pct(expectedProfit, current.profit),
    implementationStepRate: LIVE_IMPLEMENTATION_STEP_RATE,
    // The service enforces its own competitor band (meets_competitor_constraint)
    // before returning a price, though a constrained fallback can rarely
    // still cross it; that per-store detail doesn't survive product-level
    // aggregation, so this is an approximation, not a guarantee.
    competitorConflict: recommendedPrice < compLo - 1e-6 || recommendedPrice > compHi + 1e-6,
    competitorBinding: false,
    stepBinding: Math.abs(priceChangeRate) >= LIVE_IMPLEMENTATION_STEP_RATE - 1e-9,
    holdReason: isHeld ? "The pricing service withheld this item (no causal elasticity)." : undefined,
    _scenarios: { ...row._scenarios, recommended },
    _guardrails: { stepLo, stepHi, compLo, compHi, marginFloor },
    _live: true,
  };
}

/* rows: engine.js rows (from optimizeItem, keyed by itemId).
   liveByItemId: Map<itemId, live recommendation row from /products/recommendations>. */
export function applyLiveRecommendations(rows, liveByItemId, cfg) {
  return rows.map((row) => applyLiveRecommendation(row, liveByItemId.get(row.itemId), cfg));
}
