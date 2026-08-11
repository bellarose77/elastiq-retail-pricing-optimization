/* ==========================================================================
   OPTIMIZATION ENGINE  (v2 — multi-technique)

   Demand model (shared by every technique):
       q(P) = q0 * (P / P_current)^elasticity
       q0   = baseline demand, adjusted by a bounded RAG market signal
   Economics:
       revenue = P * q
       profit  = (P - unit_cost) * q - promotion_cost
   Promotion multiplies demand by (1 + uplift) and costs promotionCostRate * revenue.
   Inventory CAPS realized demand (it does not make a price infeasible):
   selling out is an outcome, and because revenue is then price x stock the
   optimizer is correctly pushed to raise price on an item that stocks out.
   src/optimization/pricing.py now shares these semantics via
   PricingOptimizationConfig.inventory_mode = "cap".

   Six techniques share this demand model but differ in the objective /
   constraint structure they optimize (see techniques metadata + Help view).
   ========================================================================== */

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const isFin = (x) => Number.isFinite(x);

// Run-level instrumentation. The live runner resets these counters before a
// job and reads them at every stage, so the workload display reports executed
// mathematical operations instead of an estimate or an artificial delay.
const computationMetrics = {
  portfolioOptimizations: 0,
  candidateEvaluations: 0,
  demandEvaluations: 0,
  profitEvaluations: 0,
  capacitySolverIterations: 0,
};

export function resetComputationMetrics() {
  Object.keys(computationMetrics).forEach((key) => { computationMetrics[key] = 0; });
}

export function getComputationMetrics() {
  return { ...computationMetrics };
}

export const DEFAULT_CONFIG = {
  objective: "profit",
  // Kept in step with PricingOptimizationConfig in
  // src/optimization/pricing.py. These previously diverged silently
  // (max +0.25 vs +0.20, actionThreshold 0.01 vs 0.005), so the same SKU
  // produced different prices in the app and the pipeline.
  minPriceChangeRate: -0.2,
  maxPriceChangeRate: 0.2,
  priceChangeStep: 0.01,
  minMarginRate: 0.2,
  enforceNonnegativeProfit: true,
  actionThreshold: 0.005,
  maxImplementationStep: 0.1,
  lowConfidenceStep: 0.03,
  lowConfidenceThreshold: 0.5,
  competitorPriceTolerance: 0.2,
  promotionCostRate: 0.02,
  promotionUpliftThreshold: 0.2,
  ragDemandAdjustmentLimit: 0.1,
  requireCausalElasticity: true,
  priceEnding: null,             // e.g. 0.99 to round recommended prices to a psychological ending; null disables it
  crossPriceElasticity: 0,       // category-sibling cannibalization strength; 0 disables the effect
  categoryPriceChangeRates: {},  // { [category]: { min, max } } overrides of min/maxPriceChangeRate per category
};

export const DEFAULT_TECHNIQUE = {
  id: "grid",
  params: {
    profitWeight: 0.6,        // multi-objective: weight on profit vs revenue
    elasUncertainty: 0.25,    // robust: +/- fraction on elasticity
    costUncertainty: 0.1,     // robust: +/- fraction on unit cost
    priorStrength: 0.6,       // bayesian: elasticity prior spread multiplier
    capacityUtilization: 0.6, // lagrangian: sellable fraction of total inventory
  },
};

/* ---- primitives ---------------------------------------------------------- */
const ragAdjustedBaseline = (bq, rag, lim) => bq * (1 + clamp(rag || 0, -lim, lim));
const minPriceForMargin = (cost, m) => (m <= 0 ? cost : cost / (1 - m));

function demandAt(item, price, promoOn, cfg, elas, cannibAdj = 0) {
  computationMetrics.demandEvaluations += 1;
  const q0 = ragAdjustedBaseline(item.baselineQuantity, item.ragSignal, cfg.ragDemandAdjustmentLimit);
  let d = q0 * Math.pow(price / item.currentPrice, elas);
  if (promoOn) d *= 1 + (item.promotionUpliftRate || 0);
  d *= 1 + (cannibAdj || 0);
  const cap = isFin(item.inventory) && item.inventory > 0 ? item.inventory : Infinity;
  return Math.min(Math.max(d, 0), cap);
}

function profitWith(item, price, promoOn, cfg, elas, cost, cannibAdj = 0) {
  computationMetrics.profitEvaluations += 1;
  const d = demandAt(item, price, promoOn, cfg, elas, cannibAdj);
  const rev = price * d;
  const pc = promoOn ? cfg.promotionCostRate * rev : 0;
  return (price - cost) * d - pc;
}

// cannibAdj: category-sibling cannibalization demand multiplier (see optimizePortfolio's
// second pass). Left at 0 for the "current" scenario so that reference point always
// reflects the item's true observed baseline, never a cannibalization-adjusted one.
export function evaluateCandidate(item, price, promoOn, cfg, cannibAdj = 0) {
  computationMetrics.candidateEvaluations += 1;
  const q0 = ragAdjustedBaseline(item.baselineQuantity, item.ragSignal, cfg.ragDemandAdjustmentLimit);
  let demand = q0 * Math.pow(price / item.currentPrice, item.elasticity);
  if (promoOn) demand *= 1 + (item.promotionUpliftRate || 0);
  demand *= 1 + (cannibAdj || 0);
  const invCap = isFin(item.inventory) && item.inventory > 0 ? item.inventory : Infinity;
  const inventoryBinding = demand > invCap;
  demand = Math.min(Math.max(demand, 0), invCap);
  const revenue = price * demand;
  const promoCost = promoOn ? cfg.promotionCostRate * revenue : 0;
  const unitMargin = price - item.unitCost;
  const marginRate = price > 0 ? unitMargin / price : NaN;
  const profit = unitMargin * demand - promoCost;
  return { price, promoOn, demand, revenue, profit, promoCost, unitMargin, marginRate, inventoryBinding };
}

// Round to the nearest price with the given fractional ending (e.g. 0.99).
export function roundToPriceEnding(price, ending) {
  if (ending == null || !isFin(ending)) return price;
  const lower = Math.floor(price - ending) + ending;
  const upper = lower + 1;
  if (lower <= 0) return upper;
  return price - lower <= upper - price ? lower : upper;
}

export function priceChangeGrid(cfg) {
  const { minPriceChangeRate: lo, maxPriceChangeRate: hi, priceChangeStep: step } = cfg;
  const n = Math.floor((hi - lo) / step);
  const rates = [];
  for (let i = 0; i <= n; i++) rates.push(+(lo + i * step).toFixed(10));
  if (rates[rates.length - 1] < hi - 1e-10) rates.push(hi);
  // Offer "hold" only when holding is inside the permitted band. Pushing 0
  // unconditionally smuggled a no-change option into bands that exclude it
  // (e.g. a mandated minimum increase), so the engine could return
  // "hold_price" while claiming to honour the floor. Mirrors the same fix
  // in src/optimization/pricing.py::generate_price_change_rates.
  if (lo <= 0 && 0 <= hi) rates.push(0);
  return Array.from(new Set(rates.map((r) => +r.toFixed(10)))).sort((a, b) => a - b);
}

function feasible(c, item, cfg) {
  const minAllowed = minPriceForMargin(item.unitCost, cfg.minMarginRate);
  if (c.price < minAllowed - 1e-9) return false;
  if (c.price < item.unitCost - 1e-9) return false;
  if (cfg.enforceNonnegativeProfit && c.profit < -1e-9) return false;
  return true;
}

const objVal = (c, o) => (o === "revenue" ? c.revenue : o === "quantity" ? c.demand : c.profit);

/* Generic grid search maximizing an arbitrary score function. */
function searchBest(item, cfg, allowPromo, scoreFn, cannibAdj = 0) {
  const grid = priceChangeGrid(cfg);
  let best = null;
  for (const rate of grid) {
    const price = item.currentPrice * (1 + rate);
    if (price <= 0) continue;
    const modes = allowPromo ? [false, true] : [false];
    for (const promoOn of modes) {
      const c = evaluateCandidate(item, price, promoOn, cfg, cannibAdj);
      if (!feasible(c, item, cfg)) continue;
      const s = scoreFn(item, price, promoOn, c);
      if (!isFin(s)) continue;
      if (best === null || s > best._score) best = { ...c, priceChangeRate: rate, _score: s };
    }
  }
  return best;
}

/* ==========================================================================
   TECHNIQUE REGISTRY — each maps the shared demand model to a different
   optimization objective. Either `.analytic` (closed form) or `.score`
   (grid-searched). Portfolio-coupled techniques add `.solvePortfolio`.
   ========================================================================== */
const GH_NODES = [-2.020182870456086, -0.9585724646138185, 0, 0.9585724646138185, 2.020182870456086];
const GH_WTS = [0.019953242059046, 0.3936193231522, 0.9453087204829, 0.3936193231522, 0.019953242059046];

export const TECHNIQUES = {
  /* 1. Constant-elasticity grid search on the chosen objective (baseline). */
  grid: {
    score: (item, p, promo, c, cfg) => objVal(c, cfg.objective),
  },

  /* 2. Closed-form marginal optimum (Amoroso-Robinson / Lerner). */
  closedform: {
    analytic: (item, cfg, tp, objective, allowPromo, refs, cannibAdj = 0) => {
      const e = item.elasticity, cost = item.unitCost;
      let price;
      if (objective === "revenue") {
        price = Math.abs(e) > 1 ? item.currentPrice * (1 + cfg.minPriceChangeRate) : item.currentPrice * (1 + cfg.maxPriceChangeRate);
      } else if (objective === "quantity") {
        price = item.currentPrice * (1 + cfg.minPriceChangeRate);
      } else {
        price = Math.abs(e) > 1 + 1e-6 ? (cost * Math.abs(e)) / (Math.abs(e) - 1) : item.currentPrice * (1 + cfg.maxPriceChangeRate);
      }
      const modes = allowPromo ? [false, true] : [false];
      let best = null;
      for (const promoOn of modes) {
        const cand = evaluateCandidate(item, price, promoOn, cfg, cannibAdj);
        const v = objVal(cand, objective);
        if (best === null || v > best._score) best = { ...cand, priceChangeRate: (price - item.currentPrice) / item.currentPrice, _score: v };
      }
      return best;
    },
  },

  /* 3. Robust worst-case (min-max) over an elasticity x cost uncertainty box. */
  robust: {
    score: (item, p, promo, c, cfg, tp, ctx, cannibAdj) => {
      const d = tp.elasUncertainty ?? 0.25, g = tp.costUncertainty ?? 0.1;
      const elasCorners = [item.elasticity * (1 + d), item.elasticity * (1 - d)];
      const costCorners = [item.unitCost * (1 - g), item.unitCost * (1 + g)];
      let worst = Infinity;
      for (const e of elasCorners) for (const cc of costCorners) {
        const prof = profitWith(item, p, promo, cfg, e, cc, cannibAdj);
        if (prof < worst) worst = prof;
      }
      return worst;
    },
  },

  /* 4. Bayesian expected profit under an elasticity posterior (Gauss-Hermite). */
  bayesian: {
    score: (item, p, promo, c, cfg, tp, ctx, cannibAdj) => {
      const k = tp.priorStrength ?? 0.6;
      const sigma = Math.abs(item.elasticity) * k * (1 - (item.confidence ?? 0.8));
      if (sigma < 1e-6) return profitWith(item, p, promo, cfg, item.elasticity, item.unitCost, cannibAdj);
      let e = 0, wsum = 0;
      for (let i = 0; i < 5; i++) {
        const eps = item.elasticity + Math.SQRT2 * sigma * GH_NODES[i];
        e += GH_WTS[i] * profitWith(item, p, promo, cfg, eps, item.unitCost, cannibAdj);
        wsum += GH_WTS[i];
      }
      return e / wsum;
    },
  },

  /* 5. Multi-objective scalarization: weighted, current-normalized blend. */
  multiobjective: {
    score: (item, p, promo, c, cfg, tp, ctx) => {
      const w = tp.profitWeight ?? 0.6;
      const pr0 = Math.max(Math.abs(ctx.current.profit), 1e-6);
      const rv0 = Math.max(Math.abs(ctx.current.revenue), 1e-6);
      return w * (c.profit / pr0) + (1 - w) * (c.revenue / rv0);
    },
  },

  /* 6. Lagrangian capacity pricing (bid-price): profit - lambda * demand. */
  lagrangian: {
    score: (item, p, promo, c, cfg, tp, ctx) => c.profit - (ctx.lambda || 0) * c.demand,
    solvePortfolio: (items, cfg, tp) => {
      const totalInv = items.reduce((a, it) => a + (isFin(it.inventory) && it.inventory > 0 ? it.inventory : 0), 0);
      const promoAllowed = (it) => (it.promotionUpliftRate || 0) >= cfg.promotionUpliftThreshold;
      const totalDemand = (lambda) =>
        items.reduce((a, it) => {
          const best = searchBest(it, cfg, promoAllowed(it), (i, pp, pr, cc) => cc.profit - lambda * cc.demand);
          return a + (best ? best.demand : 0);
        }, 0);
      const base = totalDemand(0);
      // Capacity budget = the sellable fraction of ACTUAL on-hand stock.
      //
      // This previously read `capacityUtilization * base`, i.e. a fraction of
      // unconstrained demand, which made the "capacity constraint" circular:
      // it always bound at a fixed fraction of whatever demand happened to
      // be, and `totalInv` was collected, reported, and then ignored. The
      // technique is a bid-price method; the shadow price only means
      // anything if the budget is a real resource limit. Falls back to the
      // demand-fraction behaviour only when no inventory is supplied at all.
      const utilization = tp.capacityUtilization ?? DEFAULT_TECHNIQUE.params.capacityUtilization;
      const budget = totalInv > 0 ? utilization * totalInv : utilization * base;
      if (base <= budget || base === 0) return { lambda: 0, budget, totalInventory: totalInv, baselineDemand: base, boundDemand: base, binding: false };
      let hi = 1;
      while (totalDemand(hi) > budget && hi < 1e6) hi *= 2;
      let a = 0, b = hi;
      for (let k = 0; k < 42; k++) {
        computationMetrics.capacitySolverIterations += 1;
        const m = (a + b) / 2;
        if (totalDemand(m) > budget) a = m;
        else b = m;
      }
      return { lambda: b, budget, totalInventory: totalInv, baselineDemand: base, boundDemand: totalDemand(b), binding: true };
    },
  },
};

const classifyAction = (r, t) => (!isFin(r) ? "review" : r > t ? "increase_price" : r < -t ? "decrease_price" : "hold_price");

/* ==========================================================================
   PER-ITEM OPTIMIZATION
   ========================================================================== */
export function optimizeItem(raw, config = {}, techArg, cannibAdj = 0) {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const tech = { id: (techArg && techArg.id) || "grid", params: { ...DEFAULT_TECHNIQUE.params, ...((techArg && techArg.params) || {}) }, _lambda: techArg && techArg._lambda };
  const item = { ...raw };
  const objective = cfg.objective;

  // Per-category price-change-rate band overrides the global one for every
  // search below; all other config (margin, competitor tolerance, etc.)
  // stays global.
  const catRate = cfg.categoryPriceChangeRates && cfg.categoryPriceChangeRates[item.category];
  const searchCfg = catRate ? { ...cfg, minPriceChangeRate: catRate.min, maxPriceChangeRate: catRate.max } : cfg;

  // "Current" always reflects the item's true observed state -- no
  // cannibalization adjustment here, global cfg (not category-scoped,
  // since price is fixed at currentPrice regardless of search bounds).
  const current = evaluateCandidate(item, item.currentPrice, !!item.promotionFlag, cfg);
  const causalSource = ["product_iv", "pooled_iv"].includes(item.elasticitySource);
  const evidenceDeclared = item.elasticityIsCausal !== undefined
    || item.elasticitySource !== undefined;
  const elasticityIsCausal = item.elasticityIsCausal === true
    || causalSource
    || !evidenceDeclared;
  const promoAllowed = item.promotionModelReliable !== false
    && (item.promotionUpliftRate || 0) >= cfg.promotionUpliftThreshold;

  if (cfg.requireCausalElasticity && !elasticityIsCausal) {
    const step = (item.confidence ?? 1) < cfg.lowConfidenceThreshold
      ? cfg.lowConfidenceStep
      : cfg.maxImplementationStep;
    const pct = () => 0;
    return {
      itemId: item.itemId, productId: item.productId, category: item.category,
      status: "held_no_causal_elasticity",
      holdReason: "No causally identified elasticity was supplied.",
      currentPrice: item.currentPrice, unitCost: item.unitCost,
      competitorPrice: item.competitorPrice, inventory: item.inventory,
      elasticity: item.elasticity, elasticitySource: item.elasticitySource,
      elasticityIsCausal: false, confidence: item.confidence,
      ragSignal: item.ragSignal, promotionUpliftRate: item.promotionUpliftRate,
      baselineQuantity: item.baselineQuantity, baselineSource: item.baselineSource,
      promotionModelReliable: item.promotionModelReliable,
      revenueMaxPrice: item.currentPrice, profitMaxPrice: item.currentPrice,
      techniqueOptimumPrice: item.currentPrice, recommendedPrice: item.currentPrice,
      priceChange: 0, priceChangeRate: 0, priceChangePct: 0,
      recommendedPromotionFlag: item.promotionFlag ? 1 : 0,
      recommendationAction: "review",
      currentDemand: current.demand, currentRevenue: current.revenue,
      currentProfit: current.profit, currentMarginRate: current.marginRate,
      expectedDemand: current.demand, expectedRevenue: current.revenue,
      expectedProfit: current.profit, optimizedMarginRate: current.marginRate,
      inventoryBinding: current.inventoryBinding,
      demandChange: 0, demandChangeRate: pct(), revenueChange: 0,
      revenueChangeRate: pct(), profitChange: 0, profitChangeRate: pct(),
      implementationStepRate: step, competitorConflict: false,
      competitorBinding: false, stepBinding: false, priceEndingApplied: false,
      noHarmFallbackApplied: false,
      cannibalizationAdjustment: 0, categoryPriceChangeOverride: !!catRate,
      _scenarios: {
        current, revenueMax: current, profitMax: current,
        recommended: current, quantityMax: current,
      },
      _guardrails: {
        stepLo: item.currentPrice * (1 - step),
        stepHi: item.currentPrice * (1 + step),
        compLo: -Infinity, compHi: Infinity,
        marginFloor: minPriceForMargin(item.unitCost, cfg.minMarginRate),
      },
    };
  }

  // reference scenarios (always grid) for the four scenario markers / cards
  const revenueMax = searchBest(item, searchCfg, promoAllowed, (i, p, pr, c) => c.revenue, cannibAdj) || current;
  const profitMax = searchBest(item, searchCfg, promoAllowed, (i, p, pr, c) => c.profit, cannibAdj) || current;
  const quantityMax = searchBest(item, searchCfg, promoAllowed, (i, p, pr, c) => c.demand, cannibAdj) || current;
  const refs = { current, revenueMax, profitMax, quantityMax, lambda: tech._lambda || 0 };

  // technique optimum
  const T = TECHNIQUES[tech.id] || TECHNIQUES.grid;
  let techOpt;
  if (T.analytic) techOpt = T.analytic(item, searchCfg, tech.params, objective, promoAllowed, refs, cannibAdj);
  else techOpt = searchBest(item, searchCfg, promoAllowed, (i, p, pr, c) => T.score(i, p, pr, c, cfg, tech.params, refs, cannibAdj), cannibAdj);
  techOpt = techOpt || profitMax || current;

  const step = (item.confidence ?? 1) < cfg.lowConfidenceThreshold ? cfg.lowConfidenceStep : cfg.maxImplementationStep;
  const stepLo = item.currentPrice * (1 - step);
  const stepHi = item.currentPrice * (1 + step);
  const compLo = isFin(item.competitorPrice) ? item.competitorPrice * (1 - cfg.competitorPriceTolerance) : -Infinity;
  const compHi = isFin(item.competitorPrice) ? item.competitorPrice * (1 + cfg.competitorPriceTolerance) : Infinity;
  const marginFloor = Math.max(minPriceForMargin(item.unitCost, cfg.minMarginRate), item.unitCost);
  const clampToGuardrails = (price) => Math.max(clamp(price, Math.max(stepLo, compLo, marginFloor), Math.max(Math.min(stepHi, compHi), marginFloor)), marginFloor);

  let recPrice = techOpt.price;
  let competitorBinding = isFin(item.competitorPrice) && (recPrice > compHi + 1e-9 || recPrice < compLo - 1e-9);
  let stepBinding = recPrice > stepHi + 1e-9 || recPrice < stepLo - 1e-9;
  recPrice = clampToGuardrails(recPrice);

  // Psychological-price rounding can push the price back outside the
  // guardrails above (e.g. past the competitor band); re-clamp after
  // rounding rather than trusting the rounded value blindly.
  let priceEndingApplied = false;
  if (cfg.priceEnding != null) {
    const rounded = roundToPriceEnding(recPrice, cfg.priceEnding);
    if (rounded !== recPrice) {
      priceEndingApplied = true;
      recPrice = clampToGuardrails(rounded);
    }
  }

  let compConflict = isFin(item.competitorPrice) && (recPrice > compHi + 1e-6 || recPrice < compLo - 1e-6);
  let recPromo = promoAllowed && (techOpt.promoOn ?? promoAllowed);

  let recommended = evaluateCandidate(item, recPrice, recPromo, cfg, cannibAdj);
  let noHarmFallbackApplied = false;
  if (objVal(recommended, objective) < objVal(current, objective) - 1e-9) {
    recPrice = item.currentPrice;
    recPromo = !!item.promotionFlag;
    recommended = current;
    noHarmFallbackApplied = true;
    competitorBinding = false;
    stepBinding = false;
    compConflict = isFin(item.competitorPrice)
      && (recPrice > compHi + 1e-6 || recPrice < compLo - 1e-6);
  }
  recommended.priceChangeRate = (recPrice - item.currentPrice) / item.currentPrice;
  const pcr = recommended.priceChangeRate;
  const pct = (a, b) => (b !== 0 && isFin(b) ? (a - b) / Math.abs(b) : NaN);

  return {
    itemId: item.itemId, productId: item.productId, category: item.category, status: "success",
    currentPrice: item.currentPrice, unitCost: item.unitCost, competitorPrice: item.competitorPrice,
    inventory: item.inventory, elasticity: item.elasticity, elasticitySource: item.elasticitySource,
    elasticityIsCausal,
    confidence: item.confidence, ragSignal: item.ragSignal, promotionUpliftRate: item.promotionUpliftRate,
    baselineQuantity: item.baselineQuantity, baselineSource: item.baselineSource,
    promotionModelReliable: item.promotionModelReliable,
    revenueMaxPrice: revenueMax.price, profitMaxPrice: profitMax.price, techniqueOptimumPrice: techOpt.price, recommendedPrice: recPrice,
    priceChange: recPrice - item.currentPrice, priceChangeRate: pcr, priceChangePct: pcr * 100,
    recommendedPromotionFlag: recPromo ? 1 : 0, recommendationAction: classifyAction(pcr, cfg.actionThreshold),
    currentDemand: current.demand, currentRevenue: current.revenue, currentProfit: current.profit, currentMarginRate: current.marginRate,
    expectedDemand: recommended.demand, expectedRevenue: recommended.revenue, expectedProfit: recommended.profit, optimizedMarginRate: recommended.marginRate,
    inventoryBinding: recommended.inventoryBinding,
    demandChange: recommended.demand - current.demand, demandChangeRate: pct(recommended.demand, current.demand),
    revenueChange: recommended.revenue - current.revenue, revenueChangeRate: pct(recommended.revenue, current.revenue),
    profitChange: recommended.profit - current.profit, profitChangeRate: pct(recommended.profit, current.profit),
    implementationStepRate: step, competitorConflict: compConflict, competitorBinding, stepBinding,
    priceEndingApplied, cannibalizationAdjustment: cannibAdj, categoryPriceChangeOverride: !!catRate,
    noHarmFallbackApplied,
    _scenarios: { current, revenueMax, profitMax, recommended, quantityMax },
    _guardrails: { stepLo, stepHi, compLo, compHi, marginFloor },
  };
}

/* ==========================================================================
   PORTFOLIO OPTIMIZATION
   ========================================================================== */
export function optimizePortfolio(items, config = {}, techArg) {
  computationMetrics.portfolioOptimizations += 1;
  const cfg = { ...DEFAULT_CONFIG, ...config };
  let tech = { id: (techArg && techArg.id) || "grid", params: { ...DEFAULT_TECHNIQUE.params, ...((techArg && techArg.params) || {}) } };
  const T = TECHNIQUES[tech.id] || TECHNIQUES.grid;

  let portfolioInfo = null;
  if (T.solvePortfolio) {
    portfolioInfo = T.solvePortfolio(items, cfg, tech.params);
    tech = { ...tech, _lambda: portfolioInfo.lambda };
  }

  let rows = items.map((it) => optimizeItem(it, cfg, tech));

  // Category cannibalization: a two-pass heuristic, not a joint solver.
  // Pass 1 optimizes every item independently (above); pass 2 nudges each
  // item's demand by cfg.crossPriceElasticity times its category siblings'
  // average price-change rate from pass 1 (excluding itself), then
  // re-optimizes. Sibling averages are always computed from pass 1, so
  // this doesn't converge to a joint equilibrium when several siblings
  // move together -- same limitation as the Python pricing engine's
  // apply_category_cannibalization, by design (see its docstring).
  if (cfg.crossPriceElasticity) {
    const catRates = {};
    rows.forEach((r) => {
      const c = r.category || "Uncategorized";
      if (!catRates[c]) catRates[c] = { sum: 0, count: 0 };
      catRates[c].sum += r.priceChangeRate;
      catRates[c].count += 1;
    });
    rows = items.map((it, i) => {
      const c = rows[i].category || "Uncategorized";
      const { sum, count } = catRates[c];
      const siblingAvg = count > 1 ? (sum - rows[i].priceChangeRate) / (count - 1) : 0;
      return optimizeItem(it, cfg, tech, cfg.crossPriceElasticity * siblingAvg);
    });
  }

  const scen = (name, label) => {
    const demand = rows.reduce((a, r) => a + r._scenarios[name].demand, 0);
    const revenue = rows.reduce((a, r) => a + r._scenarios[name].revenue, 0);
    const profit = rows.reduce((a, r) => a + r._scenarios[name].profit, 0);
    return { name: label, key: name, demand, revenue, profit };
  };
  const current = scen("current", "Current Strategy");
  const withDeltas = (s) => ({
    ...s,
    demandChangeRate: current.demand ? (s.demand - current.demand) / current.demand : 0,
    revenueChangeRate: current.revenue ? (s.revenue - current.revenue) / current.revenue : 0,
    profitChangeRate: current.profit ? (s.profit - current.profit) / current.profit : 0,
  });
  const portfolio = [current, scen("revenueMax", "Revenue Maximizing"), scen("profitMax", "Profit Maximizing"), scen("recommended", "Recommended Strategy")].map(withDeltas);

  const actions = { increase_price: 0, decrease_price: 0, hold_price: 0, review: 0 };
  rows.forEach((r) => { actions[r.recommendationAction] = (actions[r.recommendationAction] || 0) + 1; });

  const catMap = {};
  rows.forEach((r) => {
    const c = r.category || "Uncategorized";
    if (!catMap[c]) catMap[c] = { category: c, items: 0, currentRevenue: 0, expectedRevenue: 0, currentProfit: 0, expectedProfit: 0, priceChangeRateSum: 0, currentPriceSum: 0, recPriceSum: 0 };
    const m = catMap[c];
    m.items += 1;
    m.currentRevenue += r._scenarios.current.revenue;
    m.expectedRevenue += r._scenarios.recommended.revenue;
    m.currentProfit += r._scenarios.current.profit;
    m.expectedProfit += r._scenarios.recommended.profit;
    m.priceChangeRateSum += r.priceChangeRate;
    m.currentPriceSum += r.currentPrice;
    m.recPriceSum += r.recommendedPrice;
  });
  const categories = Object.values(catMap).map((m) => ({
    category: m.category, items: m.items,
    avgCurrentPrice: m.currentPriceSum / m.items, avgRecommendedPrice: m.recPriceSum / m.items,
    avgPriceChangeRate: m.priceChangeRateSum / m.items,
    currentRevenue: m.currentRevenue, expectedRevenue: m.expectedRevenue, revenueChange: m.expectedRevenue - m.currentRevenue,
    revenueChangeRate: m.currentRevenue ? (m.expectedRevenue - m.currentRevenue) / m.currentRevenue : 0,
    currentProfit: m.currentProfit, expectedProfit: m.expectedProfit, profitChange: m.expectedProfit - m.currentProfit,
    profitChangeRate: m.currentProfit ? (m.expectedProfit - m.currentProfit) / m.currentProfit : 0,
  })).sort((a, b) => b.expectedProfit - a.expectedProfit);

  const checks = [
    ["Recommended price is positive", (r) => r.recommendedPrice > 0],
    ["Recommended price covers cost", (r) => r.recommendedPrice >= r.unitCost - 1e-9],
    ["Minimum margin satisfied", (r) => r.optimizedMarginRate >= cfg.minMarginRate - 1e-9],
    ["Within competitor range", (r) => !r.competitorConflict],
    ["Within implementation step", (r) => Math.abs(r.priceChangeRate) <= r.implementationStepRate + 1e-9],
    ["Inventory limit respected", (r) => !r.inventoryBinding || r.expectedDemand <= r.inventory + 1e-6],
    ["Objective is not worse than current", (r) => {
      if (cfg.objective === "revenue") return r.expectedRevenue >= r.currentRevenue - 1e-8;
      if (cfg.objective === "quantity") return r.expectedDemand >= r.currentDemand - 1e-8;
      return r.expectedProfit >= r.currentProfit - 1e-8;
    }],
  ];
  const validation = checks.map(([check, fn]) => {
    const passed = rows.filter(fn).length;
    return { check, total: rows.length, passed, failed: rows.length - passed, passRate: rows.length ? passed / rows.length : 1 };
  });
  const allPass = rows.filter((r) => checks.every(([, fn]) => fn(r))).length;
  validation.push({ check: "All checks passed", total: rows.length, passed: allPass, failed: rows.length - allPass, passRate: rows.length ? allPass / rows.length : 1 });

  const rec = portfolio[3];
  const exec = {
    totalItems: rows.length, increase: actions.increase_price, decrease: actions.decrease_price, hold: actions.hold_price, review: actions.review,
    totalCurrentRevenue: current.revenue, totalExpectedRevenue: rec.revenue, totalRevenueChange: rec.revenue - current.revenue, revenueChangeRate: rec.revenueChangeRate,
    totalCurrentProfit: current.profit, totalExpectedProfit: rec.profit, totalProfitChange: rec.profit - current.profit, profitChangeRate: rec.profitChangeRate,
    totalCurrentDemand: current.demand, totalExpectedDemand: rec.demand,
    avgPriceChangeRate: rows.length ? rows.reduce((a, r) => a + r.priceChangeRate, 0) / rows.length : 0,
    promoRecommended: rows.filter((r) => r.recommendedPromotionFlag).length,
  };
  return { rows, portfolio, actions, categories, validation, exec, config: cfg, technique: tech.id, portfolioInfo };
}

export function sensitivityAnalysis(items, config, techArg, elasMult = [0.8, 1.0, 1.2], costMult = [0.9, 1.0, 1.1]) {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const grid = [];
  for (const em of elasMult)
    for (const cm of costMult) {
      const si = items.map((it) => ({ ...it, elasticity: it.elasticity * em, unitCost: it.unitCost * cm }));
      const res = optimizePortfolio(si, cfg, techArg);
      const cur = res.portfolio[0], rec = res.portfolio[3];
      grid.push({ elasticityMult: em, costMult: cm, avgPriceChangeRate: res.exec.avgPriceChangeRate, currentRevenue: cur.revenue, expectedRevenue: rec.revenue, revenueChangeRate: rec.revenueChangeRate, currentProfit: cur.profit, expectedProfit: rec.profit, profitChangeRate: rec.profitChangeRate });
    }
  return grid;
}

export function responseCurve(item, cfg = {}, promoOn = false, points = 81) {
  const c = { ...DEFAULT_CONFIG, ...cfg };
  const lo = 1 + c.minPriceChangeRate, hi = 1 + c.maxPriceChangeRate, out = [];
  for (let i = 0; i < points; i++) {
    const f = lo + ((hi - lo) * i) / (points - 1);
    const price = item.currentPrice * f;
    const cand = evaluateCandidate(item, price, promoOn, c);
    out.push({ price, demand: cand.demand, revenue: cand.revenue, profit: cand.profit, marginRate: cand.marginRate });
  }
  return out;
}
