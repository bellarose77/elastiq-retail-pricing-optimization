import { buildHybrid, choosePortfolioChampion } from "../lib/experiment.js";
import { getComputationMetrics, resetComputationMetrics } from "../lib/engine.js";
import { analysisPlan, analyzeConsensus, runMonteCarlo, runRefinement, runStressAnalysis } from "../lib/deepAnalysis.js";

self.onmessage = async (event) => {
  const request = event.data;
  if (!request || request.type !== "run") return;
  const { runId, items, cfg, techniqueIds, params, mode, depth = "deep", seed = 1 } = request;
  try {
    const startedAt = performance.now();
    resetComputationMetrics();
    const plan = analysisPlan(depth, items.length, techniqueIds.length);
    const notify = (message) => self.postMessage({ ...message, runId, metrics: getComputationMetrics() });
    const yieldWork = () => new Promise((resolve) => setTimeout(resolve, 0));
    const total = techniqueIds.length;
    notify({ type: "started", itemCount: items.length, totalMethods: total, plan });
    notify({ type: "refinementStarted", plan });
    const refinement = await runRefinement({ items, cfg, techniqueIds, params, plan, notify, yieldWork });
    const runs = refinement.finalRuns;
    runs.forEach((run, index) => notify({ type: "methodCompleted", techniqueId: run.techniqueId, index, total, summary: run.summary }));
    notify({ type: "refinementCompleted", history: refinement.history });

    notify({ type: "stressStarted", total: plan.stressScenarios });
    const stress = await runStressAnalysis({ items, cfg, techniqueIds, params, plan, seed, notify, yieldWork });
    notify({ type: "stressCompleted", summary: stress.summary });

    notify({ type: "monteCarloStarted", total: plan.monteCarloDraws });
    const risk = await runMonteCarlo({ items, runs, cfg, plan, seed, notify, yieldWork });
    notify({ type: "monteCarloCompleted", risk });

    const champion = choosePortfolioChampion(runs, cfg);
    let hybrid = null;
    if (mode === "hybrid" && runs.length > 1) {
      notify({ type: "hybridStarted" });
      hybrid = buildHybrid(runs, cfg);
      notify({ type: "hybridCompleted", summary: hybrid.summary });
    }
    notify({ type: "validationStarted" });
    const finalRows = hybrid?.rows || champion?.result.rows || runs[0].result.rows;
    const finalSummary = hybrid?.summary || champion?.summary || runs[0].summary;
    const topRows = [...finalRows]
      .sort((a, b) => (b.profitChange || 0) - (a.profitChange || 0))
      .slice(0, 40)
      .map((row) => ({
        itemId: row.itemId, productId: row.productId, category: row.category,
        currentPrice: row.currentPrice, recommendedPrice: row.recommendedPrice,
        priceChangeRate: row.priceChangeRate, recommendationAction: row.recommendationAction,
        profitChange: row.profitChange, revenueChange: row.revenueChange,
        selectedTechnique: row._hybridTechnique || champion?.techniqueId || runs[0].techniqueId,
        selectionReason: row._hybridReason || "Portfolio method",
      }));
    const consensus = analyzeConsensus(runs);
    notify({ type: "validationCompleted", passed: finalSummary.passed, total: finalSummary.itemCount, passRate: finalSummary.passRate, consensus });
    self.postMessage({
      type: "completed", runId,
      result: {
        methodSummaries: runs.map((run) => run.summary),
        championId: champion?.techniqueId || runs[0].techniqueId,
        hybridSummary: hybrid?.summary || null,
        finalSummary, topRows,
        analysis: {
          depth: plan,
          refinement: refinement.history,
          stress,
          risk,
          consensus,
          metrics: getComputationMetrics(),
          computeElapsedMs: performance.now() - startedAt,
          validationChecks: finalSummary.itemCount * 7,
        },
      },
    });
  } catch (error) {
    self.postMessage({ type: "failed", runId, error: error instanceof Error ? error.message : String(error) });
  }
};
