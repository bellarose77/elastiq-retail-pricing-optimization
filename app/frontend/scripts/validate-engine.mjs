import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { evaluateCandidate, priceChangeGrid } from "../src/lib/engine.js";
import { optimizePortfolio, optimizeItem, summarizePortfolio, DEFAULT_CONFIG, DEFAULT_TECHNIQUE, roundToPriceEnding } from "../src/lib/engine.js";
import { applyLiveRecommendation, applyLiveRecommendations } from "../src/lib/liveRecommendations.js";
import { DEMO_DATA } from "../src/lib/demoData.js";
import { TECHNIQUE_LIST } from "../src/lib/techniques.js";
import { parseCSV } from "../src/lib/report.js";
import { generateScenario } from "../src/lib/scenarioGenerator.js";
import { analyzeInput } from "../src/lib/inputAnalysis.js";
import {
  buildHybrid,
  choosePortfolioChampion,
  runTechnique,
  validateExperimentResult,
} from "../src/lib/experiment.js";

for (const method of TECHNIQUE_LIST) {
  const result = optimizePortfolio(DEMO_DATA, DEFAULT_CONFIG, { ...DEFAULT_TECHNIQUE, id: method.id });
  assert.equal(result.rows.length, DEMO_DATA.length, `${method.id}: row count`);
  assert.ok(Number.isFinite(result.exec.totalExpectedProfit), `${method.id}: profit finite`);
  assert.ok(Number.isFinite(result.exec.totalExpectedRevenue), `${method.id}: revenue finite`);
  for (const row of result.rows) {
    assert.ok(row.recommendedPrice > 0, `${method.id}/${row.itemId}: positive price`);
    assert.ok(row.recommendedPrice + 1e-8 >= row.unitCost, `${method.id}/${row.itemId}: covers cost`);
    assert.ok(row.optimizedMarginRate + 1e-8 >= DEFAULT_CONFIG.minMarginRate, `${method.id}/${row.itemId}: margin`);
    assert.ok(Math.abs(row.priceChangeRate) <= row.implementationStepRate + 1e-8, `${method.id}/${row.itemId}: step`);
  }
  const overall = result.validation.find(v => v.check === "All checks passed");
  assert.equal(overall.passed, overall.total, `${method.id}: validation`);
  console.log(`${method.id.padEnd(14)} profit ${result.exec.profitChangeRate.toFixed(4)} revenue ${result.exec.revenueChangeRate.toFixed(4)} checks ${overall.passed}/${overall.total}`);
}
console.log("All optimization methods passed engine and guardrail validation.");

/* ---- advanced features (parity with the Python pricing model) ---------- */

assert.equal(roundToPriceEnding(12.34, 0.99), 11.99, "roundToPriceEnding: rounds down when closer");
assert.equal(roundToPriceEnding(12.90, 0.99), 12.99, "roundToPriceEnding: rounds up when closer");
assert.ok(roundToPriceEnding(0.5, 0.99) > 0, "roundToPriceEnding: never returns a non-positive price");

const priceEndingResult = optimizePortfolio(DEMO_DATA, { ...DEFAULT_CONFIG, priceEnding: 0.99 }, DEFAULT_TECHNIQUE);
assert.ok(priceEndingResult.rows.some((r) => r.priceEndingApplied), "priceEnding: applied to at least one row");
const priceEndingChecks = priceEndingResult.validation.find((v) => v.check === "All checks passed");
assert.equal(priceEndingChecks.passed, priceEndingChecks.total, "priceEnding: guardrails still hold after rounding");

const cannibResult = optimizePortfolio(DEMO_DATA, { ...DEFAULT_CONFIG, objective: "revenue", crossPriceElasticity: 0.3 }, DEFAULT_TECHNIQUE);
assert.ok(cannibResult.rows.some((r) => r.cannibalizationAdjustment !== 0), "crossPriceElasticity: adjustment engages for at least one row");
const cannibChecks = cannibResult.validation.find((v) => v.check === "All checks passed");
assert.equal(cannibChecks.passed, cannibChecks.total, "crossPriceElasticity: guardrails still hold");

const targetCategory = DEMO_DATA[0].category;
const categoryCapResult = optimizePortfolio(
  DEMO_DATA,
  { ...DEFAULT_CONFIG, objective: "revenue", categoryPriceChangeRates: { [targetCategory]: { min: -0.001, max: 0.001 } } },
  DEFAULT_TECHNIQUE
);
const cappedRows = categoryCapResult.rows.filter((r) => r.category === targetCategory);
assert.ok(cappedRows.length > 0, "categoryPriceChangeRates: at least one row in the overridden category");
assert.ok(
  cappedRows.every((r) => r.recommendationAction === "hold_price"),
  "categoryPriceChangeRates: near-zero band forces a hold"
);

console.log("Advanced pricing features (price ending, cannibalization, category caps) validated.");

const imported = parseCSV(
  'itemId,category,currentPrice,unitCost,baselineQuantity,elasticity,elasticitySource,elasticityIsCausal\n' +
  'A1,"Food, chilled",10,4,100,-1.8,product_iv,true\n'
);
assert.equal(imported[0].category, "Food, chilled", "CSV parser preserves quoted commas");
assert.equal(imported[0].elasticityIsCausal, true, "CSV parser preserves causal provenance");
assert.throws(
  () => parseCSV('itemId,currentPrice,unitCost,baselineQuantity,elasticity\nA1,10,4,100,0.4\n'),
  /elasticity must be a finite negative value/,
  "CSV parser rejects upward-sloping demand"
);
console.log("CSV import quoting, aliases, provenance and validation checks passed.");


/* ---- technique differentiation ------------------------------------------ *
 * The app's headline feature is six distinct optimization techniques. When
 * the demo data carried a constant fallback elasticity and low confidence,
 * the implementation-step clamp bound on every item and five of six
 * techniques collapsed to byte-identical prices -- and nothing here noticed,
 * because every assertion above tests invariants rather than behaviour.
 * ------------------------------------------------------------------------ */

const priceVectors = new Map();

for (const method of TECHNIQUE_LIST) {
  const result = optimizePortfolio(DEMO_DATA, DEFAULT_CONFIG, { ...DEFAULT_TECHNIQUE, id: method.id });
  priceVectors.set(method.id, result.rows.map((r) => r.recommendedPrice.toFixed(4)).join(","));
  const uplift = result.exec.profitChangeRate;
  assert.ok(
    Number.isFinite(uplift) && uplift > -0.5 && uplift < 0.5,
    `${method.id}: profit uplift ${uplift} is outside a plausible range -- a repricing-only ` +
      `lift beyond +/-50% indicates a broken elasticity or baseline, not an opportunity`
  );
}

const distinctVectors = new Set(priceVectors.values()).size;
assert.ok(
  distinctVectors >= 4,
  `only ${distinctVectors} of ${TECHNIQUE_LIST.length} techniques produce distinct prices -- ` +
    `a guardrail is clamping them all to the same answer, making the technique choice cosmetic`
);
console.log(`Technique differentiation: ${distinctVectors}/${TECHNIQUE_LIST.length} distinct price vectors.`);

/* ---- live experiment lab ------------------------------------------------ */

for (const size of [50, 250, 1000]) {
  const generated = generateScenario({ size, profile: "balanced", seed: 2407 });
  assert.equal(generated.length, size, `live lab: generated ${size} rows`);
  assert.deepEqual(
    generated[0],
    generateScenario({ size, profile: "balanced", seed: 2407 })[0],
    `live lab: seed ${size} is reproducible`
  );
}

const liveRows = generateScenario({ size: 50, profile: "balanced", seed: 20260731 });
const inputProfile = analyzeInput(liveRows);
assert.equal(inputProfile.rows, 50, "input profile: row count");
assert.equal(inputProfile.completenessRate, 1, "input profile: complete generated rows");
assert.equal(inputProfile.causalRate, 1, "input profile: causal coverage");
assert.ok(inputProfile.categoryCount > 1, "input profile: multiple categories");

const liveMethods = ["grid", "robust", "bayesian", "multiobjective"];
const completed = liveMethods.map((id) => runTechnique(liveRows, DEFAULT_CONFIG, id));
for (const method of completed) {
  assert.equal(method.result.rows.length, liveRows.length, `${method.techniqueId}: live result row count`);
  assert.ok(Number.isFinite(method.summary.expectedProfit), `${method.techniqueId}: live profit finite`);
  assert.equal(method.summary.passRate, 1, `${method.techniqueId}: live guardrails`);
  assert.ok(method.summary.elapsedMs >= 0, `${method.techniqueId}: real elapsed time recorded`);
}

const hybrid = buildHybrid(completed, DEFAULT_CONFIG);
assert.equal(hybrid.rows.length, liveRows.length, "hybrid: one selected decision per row");
assert.equal(
  Object.values(hybrid.summary.methodMix).reduce((total, count) => total + count, 0),
  liveRows.length,
  "hybrid: method contribution counts cover the scenario"
);
assert.equal(hybrid.summary.passRate, 1, "hybrid: all selected decisions pass guardrails");
assert.ok(Object.keys(hybrid.summary.methodMix).length >= 2, "hybrid: context routing uses more than one method");
assert.ok(choosePortfolioChampion(completed, DEFAULT_CONFIG), "live lab: portfolio champion selected");
assert.equal(
  validateExperimentResult({ methodSummaries: completed.map(({ summary }) => summary), hybrid: hybrid.summary }),
  true,
  "live lab: completed experiment validates"
);
console.log("Live scenario generation, input profiling, measured execution and hybrid selection validated.");

/* ---- cross-language parity with src/optimization/pricing.py ------------- *
 * Replays a fixture generated by the Python engine. These are two
 * independent implementations of one demand model; without this they drift.
 * Regenerate with: PYTHONPATH=. python scripts/generate_parity_fixture.py
 * ------------------------------------------------------------------------ */

const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/engine_parity.json", import.meta.url), "utf-8"));
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

for (const key of ["minPriceChangeRate", "maxPriceChangeRate", "priceChangeStep", "actionThreshold"]) {
  assert.ok(
    near(DEFAULT_CONFIG[key], fixture.python_defaults[key]),
    `default drift on ${key}: JS ${DEFAULT_CONFIG[key]} vs Python ${fixture.python_defaults[key]}`
  );
}

for (const { input, expected } of fixture.grids) {
  const grid = priceChangeGrid({
    minPriceChangeRate: input.minimum,
    maxPriceChangeRate: input.maximum,
    priceChangeStep: input.step,
  });
  assert.equal(grid.length, expected.length, `grid length for ${input.minimum}..${input.maximum}`);
  grid.forEach((rate, i) =>
    assert.ok(near(rate, expected[i]), `grid rate ${i} for ${input.minimum}..${input.maximum}: ${rate} vs ${expected[i]}`)
  );
}

for (const { input, expected } of fixture.roundings) {
  assert.ok(
    near(roundToPriceEnding(input.price, input.ending), expected),
    `roundToPriceEnding(${input.price}, ${input.ending}) diverges from Python`
  );
}

for (const { input, expected } of fixture.candidates) {
  const item = {
    itemId: input.label,
    currentPrice: input.current_price,
    baselineQuantity: input.baseline_quantity,
    elasticity: input.elasticity,
    unitCost: input.unit_cost,
    promotionUpliftRate: input.promotion_uplift_rate,
    inventory: input.available_inventory ?? NaN,
    ragSignal: 0,
  };
  const cfg = { ...DEFAULT_CONFIG, promotionCostRate: input.promotion_cost_rate, ragDemandAdjustmentLimit: 0 };
  const got = evaluateCandidate(item, input.candidate_price, input.promotion_active, cfg);
  assert.ok(near(got.demand, expected.demand, 1e-6), `${input.label}: demand ${got.demand} vs Python ${expected.demand}`);
  assert.ok(near(got.revenue, expected.revenue, 1e-6), `${input.label}: revenue ${got.revenue} vs Python ${expected.revenue}`);
  assert.ok(near(got.profit, expected.profit, 1e-6), `${input.label}: profit ${got.profit} vs Python ${expected.profit}`);
  assert.equal(got.inventoryBinding, expected.inventoryBinding, `${input.label}: inventoryBinding`);
}

console.log(`Cross-language parity: ${fixture.candidates.length} candidates, ${fixture.grids.length} grids, ${fixture.roundings.length} roundings match the Python engine.`);

/* ---- live pricing-service integration (summarizePortfolio extraction, --- *
 * liveRecommendations.js's patching) --------------------------------------
 * ------------------------------------------------------------------------ */

{
  // summarizePortfolio must be a lossless extraction: computing rows and
  // aggregating them separately has to match optimizePortfolio doing both.
  const rows = DEMO_DATA.map((it) => optimizeItem(it, DEFAULT_CONFIG, DEFAULT_TECHNIQUE));
  const viaSummarize = summarizePortfolio(rows, DEFAULT_CONFIG, DEFAULT_TECHNIQUE, null);
  const viaPortfolio = optimizePortfolio(DEMO_DATA, DEFAULT_CONFIG, DEFAULT_TECHNIQUE);
  assert.deepEqual(viaSummarize.exec, viaPortfolio.exec, "summarizePortfolio: exec matches optimizePortfolio");
  assert.deepEqual(viaSummarize.portfolio, viaPortfolio.portfolio, "summarizePortfolio: portfolio matches optimizePortfolio");
  assert.deepEqual(viaSummarize.validation, viaPortfolio.validation, "summarizePortfolio: validation matches optimizePortfolio");
  console.log("summarizePortfolio extraction is lossless against optimizePortfolio.");
}

{
  const source = DEMO_DATA[0];
  const row = optimizeItem(source, DEFAULT_CONFIG, DEFAULT_TECHNIQUE);

  // A row with no matching live recommendation passes through unchanged.
  const unmatched = applyLiveRecommendation(row, undefined, DEFAULT_CONFIG);
  assert.strictEqual(unmatched, row, "applyLiveRecommendation: no live match returns the original row");

  // A successful live recommendation overrides the decision fields.
  const live = {
    itemId: source.itemId,
    status: "success",
    recommendedPrice: row.currentPrice * 1.15,
    expectedQuantity: 100,
    expectedRevenue: 500,
    expectedProfit: 200,
  };
  const patched = applyLiveRecommendation(row, live, DEFAULT_CONFIG);
  assert.equal(patched.recommendedPrice, live.recommendedPrice, "live patch: recommendedPrice overridden");
  assert.ok(Math.abs(patched.priceChangeRate - 0.15) < 1e-9, "live patch: priceChangeRate derived from live price");
  assert.equal(patched.recommendationAction, "increase_price", "live patch: action reclassified from the live price");
  assert.equal(patched.expectedDemand, 100, "live patch: expectedDemand from the service's expectedQuantity");
  assert.equal(patched._scenarios.recommended.price, live.recommendedPrice, "live patch: recommended scenario price updated");
  assert.ok(
    Math.abs(patched._scenarios.recommended.marginRate - (live.recommendedPrice - row.unitCost) / live.recommendedPrice) < 1e-9,
    "live patch: recommended scenario margin recomputed at the live price"
  );
  assert.ok(Math.abs(patched._guardrails.stepHi - row.currentPrice * 1.20) < 1e-6, "live patch: guardrail step-hi uses the service's +20% band");
  assert.ok(Math.abs(patched._guardrails.stepLo - row.currentPrice * 0.90) < 1e-6, "live patch: guardrail step-lo uses the service's -10% band");
  assert.equal(patched._live, true, "live patch: row flagged as live-sourced");
  // "What if" comparison scenarios stay locally computed, untouched.
  assert.equal(patched._scenarios.revenueMax, row._scenarios.revenueMax, "live patch: revenueMax scenario left to the client engine");

  // A held item (no causal elasticity) is classified as a review, not a move.
  const heldLive = { ...live, status: "held_no_causal_elasticity", recommendedPrice: row.currentPrice };
  const heldPatched = applyLiveRecommendation(row, heldLive, DEFAULT_CONFIG);
  assert.equal(heldPatched.recommendationAction, "review", "live patch: held item classified as review");
  assert.ok(typeof heldPatched.holdReason === "string" && heldPatched.holdReason.length > 0, "live patch: held item carries a hold reason");

  console.log("Live recommendation patching (lib/liveRecommendations.js) validated.");
}

{
  // Full round-trip: patch every DEMO_DATA row with a live recommendation
  // and re-summarize -- the shape App.jsx renders for the "grid" technique.
  const rows = DEMO_DATA.map((it) => optimizeItem(it, DEFAULT_CONFIG, DEFAULT_TECHNIQUE));
  const liveByItemId = new Map(
    DEMO_DATA.map((it) => {
      const recommendedPrice = it.currentPrice * 1.05;
      return [
        it.itemId,
        {
          itemId: it.itemId,
          status: "success",
          recommendedPrice,
          expectedQuantity: it.baselineQuantity,
          expectedRevenue: recommendedPrice * it.baselineQuantity,
          expectedProfit: (recommendedPrice - it.unitCost) * it.baselineQuantity,
        },
      ];
    })
  );
  const patchedRows = applyLiveRecommendations(rows, liveByItemId, DEFAULT_CONFIG);
  assert.ok(patchedRows.every((r) => r._live === true), "live round-trip: every row matched a live recommendation");
  const liveResult = summarizePortfolio(patchedRows, DEFAULT_CONFIG, DEFAULT_TECHNIQUE, null);
  assert.equal(liveResult.rows.length, DEMO_DATA.length, "live round-trip: row count preserved");
  assert.ok(Number.isFinite(liveResult.exec.totalExpectedProfit), "live round-trip: total profit finite");
  assert.ok(Number.isFinite(liveResult.exec.totalExpectedRevenue), "live round-trip: total revenue finite");
  const overall = liveResult.validation.find((v) => v.check === "All checks passed");
  assert.ok(Number.isFinite(overall.passRate), "live round-trip: validation pass rate finite");
  console.log("Live-sourced portfolio summary (App.jsx's grid-technique path) validated end to end.");
}
