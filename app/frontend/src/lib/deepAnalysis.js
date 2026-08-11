import { evaluateCandidate } from "./engine.js";
import { runTechnique } from "./experiment.js";

export const ANALYSIS_DEPTHS = [
  { id: "standard", label: "Standard", note: "Baseline optimization with a light verification sample." },
  { id: "deep", label: "Deep analysis", note: "Multi-resolution search, stress re-optimization and 1,000 uncertainty draws." },
  { id: "research", label: "Research", note: "The heaviest audit: finer search, more shocks and 3,000 uncertainty draws." },
];

const DEPTH = {
  standard: { steps: [0.01], draws: 100, stress: (n) => n <= 75 ? 4 : n <= 300 ? 2 : 1 },
  deep: { steps: [0.01, 0.005, 0.0025], draws: 1000, stress: (n) => n <= 75 ? 90 : n <= 300 ? 24 : 7 },
  research: { steps: [0.01, 0.005, 0.0025, 0.00125], draws: 3000, stress: (n) => n <= 75 ? 180 : n <= 300 ? 55 : 15 },
};

export function analysisPlan(depthId = "deep", itemCount = 50, methodCount = 6) {
  const cfg = DEPTH[depthId] || DEPTH.deep;
  const stressScenarios = cfg.stress(itemCount);
  return {
    id: depthId,
    label: ANALYSIS_DEPTHS.find((item) => item.id === depthId)?.label || "Deep analysis",
    refinementSteps: cfg.steps,
    refinementPasses: cfg.steps.length,
    stressScenarios,
    monteCarloDraws: cfg.draws,
    plannedOptimizerRuns: methodCount * (cfg.steps.length + stressScenarios),
    plannedPolicySimulations: methodCount * cfg.draws * itemCount,
  };
}

const seeded = (seed = 1) => {
  let state = (Number(seed) || 1) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const normal = (random) => {
  const u = Math.max(random(), 1e-12);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const percentile = (values, p) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * p;
  const lo = Math.floor(index), hi = Math.ceil(index);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
};

const stats = (values) => {
  const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(values.length, 1);
  return {
    mean,
    p05: percentile(values, 0.05),
    median: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    standardDeviation: Math.sqrt(variance),
    positiveRate: values.filter((value) => value >= 0).length / Math.max(values.length, 1),
  };
};

function shockedItems(items, scenarioIndex, seed) {
  const random = seeded((seed || 1) + scenarioIndex * 104729);
  const demandShock = Math.max(0.65, 1 + normal(random) * 0.09);
  const costShock = Math.max(0.82, 1 + normal(random) * 0.045);
  const elasticityShock = Math.max(0.72, 1 + normal(random) * 0.10);
  const inventoryShock = Math.max(0.65, 1 + normal(random) * 0.08);
  const competitorShock = Math.max(0.8, 1 + normal(random) * 0.055);
  const shocks = { scenario: scenarioIndex + 1, demandShock, costShock, elasticityShock, inventoryShock, competitorShock };
  const rows = items.map((item) => ({
    ...item,
    baselineQuantity: Math.max(0, item.baselineQuantity * demandShock * Math.max(0.82, 1 + normal(random) * 0.035)),
    unitCost: Math.max(0.01, item.unitCost * costShock * Math.max(0.94, 1 + normal(random) * 0.012)),
    elasticity: item.elasticity * elasticityShock * Math.max(0.9, 1 + normal(random) * 0.025),
    inventory: Math.max(1, item.inventory * inventoryShock),
    competitorPrice: Number.isFinite(item.competitorPrice) ? item.competitorPrice * competitorShock : item.competitorPrice,
    ragSignal: Math.max(-0.1, Math.min(0.1, (item.ragSignal || 0) + normal(random) * 0.018)),
  }));
  return { rows, shocks };
}

export async function runRefinement({ items, cfg, techniqueIds, params, plan, notify, yieldWork }) {
  const passes = [];
  let finalRuns = [];
  for (let passIndex = 0; passIndex < plan.refinementSteps.length; passIndex += 1) {
    const step = plan.refinementSteps[passIndex];
    const passRuns = [];
    for (let methodIndex = 0; methodIndex < techniqueIds.length; methodIndex += 1) {
      const techniqueId = techniqueIds[methodIndex];
      notify({ type: "refinementProgress", passIndex, passes: plan.refinementPasses, methodIndex, totalMethods: techniqueIds.length, techniqueId, step });
      passRuns.push(runTechnique(items, { ...cfg, priceChangeStep: step }, techniqueId, params));
      await yieldWork();
    }
    const prior = passes[passes.length - 1];
    const convergence = passRuns.map((run) => {
      const previous = prior?.runs.find((item) => item.techniqueId === run.techniqueId);
      if (!previous) return { techniqueId: run.techniqueId, meanAbsolutePriceShift: null, stableRate: null };
      let totalShift = 0, stable = 0;
      run.result.rows.forEach((row, index) => {
        const shift = Math.abs(row.recommendedPrice - previous.result.rows[index].recommendedPrice) / Math.max(row.currentPrice, 1e-9);
        totalShift += shift;
        if (shift <= step + 1e-9) stable += 1;
      });
      return { techniqueId: run.techniqueId, meanAbsolutePriceShift: totalShift / run.result.rows.length, stableRate: stable / run.result.rows.length };
    });
    passes.push({ pass: passIndex + 1, priceStep: step, convergence, runs: passRuns });
    finalRuns = passRuns;
  }
  return {
    finalRuns,
    history: passes.map(({ pass, priceStep, convergence, runs }) => ({ pass, priceStep, convergence, summaries: runs.map((run) => run.summary) })),
  };
}

export async function runStressAnalysis({ items, cfg, techniqueIds, params, plan, seed, notify, yieldWork }) {
  const byMethod = Object.fromEntries(techniqueIds.map((id) => [id, []]));
  const scenarios = [];
  for (let index = 0; index < plan.stressScenarios; index += 1) {
    const scenario = shockedItems(items, index, seed);
    const outcomes = {};
    for (const techniqueId of techniqueIds) {
      const run = runTechnique(scenario.rows, cfg, techniqueId, params);
      const outcome = { profitChangeRate: run.summary.profitChangeRate, revenueChangeRate: run.summary.revenueChangeRate, passRate: run.summary.passRate };
      byMethod[techniqueId].push(outcome.profitChangeRate);
      outcomes[techniqueId] = outcome;
    }
    scenarios.push({ ...scenario.shocks, outcomes });
    notify({ type: "stressProgress", completed: index + 1, total: plan.stressScenarios });
    await yieldWork();
  }
  const summary = Object.fromEntries(techniqueIds.map((id) => [id, stats(byMethod[id])]));
  return { scenarios, summary };
}

export async function runMonteCarlo({ items, runs, cfg, plan, seed, notify, yieldWork }) {
  const random = seeded((seed || 1) ^ 0xA5A5A5A5);
  const outcomes = Object.fromEntries(runs.map((run) => [run.techniqueId, []]));
  const chunk = Math.max(10, Math.floor(plan.monteCarloDraws / 40));
  for (let draw = 0; draw < plan.monteCarloDraws; draw += 1) {
    const portfolioShocks = {
      demand: Math.max(0.6, 1 + normal(random) * 0.10),
      cost: Math.max(0.8, 1 + normal(random) * 0.05),
      elasticity: Math.max(0.65, 1 + normal(random) * 0.12),
    };
    for (const run of runs) {
      let currentProfit = 0, recommendedProfit = 0;
      for (let index = 0; index < items.length; index += 1) {
        const item = items[index], decision = run.result.rows[index];
        const simulated = {
          ...item,
          baselineQuantity: Math.max(0, item.baselineQuantity * portfolioShocks.demand * Math.max(0.78, 1 + normal(random) * 0.05)),
          unitCost: Math.max(0.01, item.unitCost * portfolioShocks.cost * Math.max(0.92, 1 + normal(random) * 0.02)),
          elasticity: item.elasticity * portfolioShocks.elasticity * Math.max(0.85, 1 + normal(random) * 0.04),
        };
        currentProfit += evaluateCandidate(simulated, item.currentPrice, !!item.promotionFlag, cfg).profit;
        recommendedProfit += evaluateCandidate(simulated, decision.recommendedPrice, !!decision.recommendedPromotionFlag, cfg).profit;
      }
      outcomes[run.techniqueId].push(currentProfit ? (recommendedProfit - currentProfit) / Math.abs(currentProfit) : 0);
    }
    if ((draw + 1) % chunk === 0 || draw + 1 === plan.monteCarloDraws) {
      notify({ type: "monteCarloProgress", completed: draw + 1, total: plan.monteCarloDraws });
      await yieldWork();
    }
  }
  return Object.fromEntries(runs.map((run) => [run.techniqueId, stats(outcomes[run.techniqueId])]));
}

export function analyzeConsensus(runs) {
  if (!runs.length) return { meanAgreementRate: 1, unanimousRate: 1, disputedItems: 0 };
  let agreement = 0, unanimous = 0, disputed = 0;
  const rows = runs[0].result.rows;
  rows.forEach((_, index) => {
    const actions = runs.map((run) => run.result.rows[index].recommendationAction);
    const counts = actions.reduce((acc, action) => ({ ...acc, [action]: (acc[action] || 0) + 1 }), {});
    const best = Math.max(...Object.values(counts));
    agreement += best / actions.length;
    if (best === actions.length) unanimous += 1;
    else disputed += 1;
  });
  return { meanAgreementRate: agreement / rows.length, unanimousRate: unanimous / rows.length, disputedItems: disputed };
}
