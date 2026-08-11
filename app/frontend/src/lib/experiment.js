import { DEFAULT_TECHNIQUE, isFin, optimizePortfolio } from "./engine.js";
import { TECHNIQUE_MAP } from "./techniques.js";

const objectiveValue = (row, objective) => objective === "revenue"
  ? row.expectedRevenue
  : objective === "quantity"
    ? row.expectedDemand
    : row.expectedProfit;

const currentObjective = (row, objective) => objective === "revenue"
  ? row.currentRevenue
  : objective === "quantity"
    ? row.currentDemand
    : row.currentProfit;

export function rowPasses(row, cfg) {
  if (row.status !== "success") return false;
  if (!(row.recommendedPrice > 0) || row.recommendedPrice < row.unitCost - 1e-9) return false;
  if (row.optimizedMarginRate < cfg.minMarginRate - 1e-9) return false;
  if (row.competitorConflict) return false;
  if (Math.abs(row.priceChangeRate) > row.implementationStepRate + 1e-9) return false;
  if (row.inventoryBinding && row.expectedDemand > row.inventory + 1e-6) return false;
  return objectiveValue(row, cfg.objective) >= currentObjective(row, cfg.objective) - 1e-8;
}

export function summarizeRows(rows, cfg, { id, name, elapsedMs = 0, kind = "method" } = {}) {
  const sums = rows.reduce((acc, row) => {
    acc.currentRevenue += row.currentRevenue || 0;
    acc.expectedRevenue += row.expectedRevenue || 0;
    acc.currentProfit += row.currentProfit || 0;
    acc.expectedProfit += row.expectedProfit || 0;
    acc.currentDemand += row.currentDemand || 0;
    acc.expectedDemand += row.expectedDemand || 0;
    acc.priceChange += row.priceChangeRate || 0;
    acc.actions[row.recommendationAction] = (acc.actions[row.recommendationAction] || 0) + 1;
    if (rowPasses(row, cfg)) acc.passed += 1;
    return acc;
  }, { currentRevenue: 0, expectedRevenue: 0, currentProfit: 0, expectedProfit: 0, currentDemand: 0, expectedDemand: 0, priceChange: 0, passed: 0, actions: {} });
  const delta = (next, current) => current ? (next - current) / Math.abs(current) : 0;
  const methodMix = {};
  rows.forEach((row) => {
    const method = row._hybridTechnique || id;
    methodMix[method] = (methodMix[method] || 0) + 1;
  });
  return {
    id, name, kind, elapsedMs, itemCount: rows.length,
    passed: sums.passed, passRate: rows.length ? sums.passed / rows.length : 1,
    currentRevenue: sums.currentRevenue, expectedRevenue: sums.expectedRevenue,
    revenueChange: sums.expectedRevenue - sums.currentRevenue,
    revenueChangeRate: delta(sums.expectedRevenue, sums.currentRevenue),
    currentProfit: sums.currentProfit, expectedProfit: sums.expectedProfit,
    profitChange: sums.expectedProfit - sums.currentProfit,
    profitChangeRate: delta(sums.expectedProfit, sums.currentProfit),
    currentDemand: sums.currentDemand, expectedDemand: sums.expectedDemand,
    demandChangeRate: delta(sums.expectedDemand, sums.currentDemand),
    avgPriceChangeRate: rows.length ? sums.priceChange / rows.length : 0,
    actions: sums.actions, methodMix,
  };
}

export function runTechnique(items, cfg, techniqueId, params = {}) {
  const started = performance.now();
  const result = optimizePortfolio(items, cfg, { id: techniqueId, params: { ...DEFAULT_TECHNIQUE.params, ...params } });
  const elapsedMs = performance.now() - started;
  const meta = TECHNIQUE_MAP[techniqueId];
  return {
    techniqueId,
    result,
    summary: summarizeRows(result.rows, cfg, { id: techniqueId, name: meta?.short || techniqueId, elapsedMs }),
  };
}

export function buildHybrid(runs, cfg) {
  if (!runs.length) throw new Error("Hybrid execution requires at least one completed technique.");
  const started = performance.now();
  const rowCount = runs[0].result.rows.length;
  const rows = [];
  for (let index = 0; index < rowCount; index += 1) {
    const candidates = runs.map((run) => ({ row: run.result.rows[index], techniqueId: run.techniqueId }))
      .filter(({ row }) => rowPasses(row, cfg));
    const pool = candidates.length ? candidates : runs.map((run) => ({ row: run.result.rows[index], techniqueId: run.techniqueId }));
    const sample = pool[0]?.row || {};
    const preferred =
      (sample.inventory <= sample.baselineQuantity * 1.05 && pool.find((candidate) => candidate.techniqueId === "multiobjective")) ||
      (((sample.confidence ?? 1) < .7 || Math.abs(sample.ragSignal || 0) >= .045) && pool.find((candidate) => candidate.techniqueId === "robust")) ||
      ((sample.promotionModelReliable !== false && (sample.promotionUpliftRate || 0) >= cfg.promotionUpliftThreshold) && pool.find((candidate) => candidate.techniqueId === "bayesian"));
    pool.sort((a, b) => {
      const objectiveDelta = objectiveValue(b.row, cfg.objective) - objectiveValue(a.row, cfg.objective);
      if (Math.abs(objectiveDelta) > 1e-8) return objectiveDelta;
      return Math.abs(a.row.priceChangeRate || 0) - Math.abs(b.row.priceChangeRate || 0);
    });
    const selected = preferred || pool[0];
    const selectionReason = preferred
      ? selected.techniqueId === "multiobjective" ? "Inventory-pressure routing"
        : selected.techniqueId === "robust" ? "Uncertainty-aware routing"
          : "Promotion-evidence routing"
      : "Best feasible configured objective";
    rows.push({ ...selected.row, _hybridTechnique: selected.techniqueId, _hybridReason: selectionReason });
  }
  const elapsedMs = performance.now() - started + runs.reduce((total, run) => total + run.summary.elapsedMs, 0);
  return {
    rows,
    summary: summarizeRows(rows, cfg, { id: "hybrid", name: "Item-level hybrid", elapsedMs, kind: "hybrid" }),
  };
}

export function choosePortfolioChampion(runs, cfg) {
  const valid = runs.filter((run) => run.summary.passRate === 1);
  const pool = valid.length ? valid : runs;
  return [...pool].sort((a, b) => {
    const aValue = cfg.objective === "revenue" ? a.summary.expectedRevenue : cfg.objective === "quantity" ? a.summary.expectedDemand : a.summary.expectedProfit;
    const bValue = cfg.objective === "revenue" ? b.summary.expectedRevenue : cfg.objective === "quantity" ? b.summary.expectedDemand : b.summary.expectedProfit;
    return bValue - aValue || a.summary.elapsedMs - b.summary.elapsedMs;
  })[0] || null;
}

export function validateExperimentResult(result) {
  if (!result || !Array.isArray(result.methodSummaries) || !result.methodSummaries.length) return false;
  return result.methodSummaries.every((summary) => isFin(summary.elapsedMs) && summary.elapsedMs >= 0 && summary.itemCount > 0 && summary.passRate >= 0 && summary.passRate <= 1);
}
