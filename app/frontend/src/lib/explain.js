import { fmtMoney, fmtPrice, fmtPct, fmtPctPlain, fmtNum } from "./format.js";
import { TECHNIQUE_MAP } from "./techniques.js";

/* Produces a live, data-driven read-out for whichever page the user is on.
   Everything here is computed from the current result — no static text. */
function techNote(techId, result, cfg) {
  const t = TECHNIQUE_MAP[techId];
  if (!t) return null;
  if (techId === "lagrangian" && result.portfolioInfo) {
    const p = result.portfolioInfo;
    return p.binding
      ? `Capacity is binding: the sellable budget of ${fmtNum(p.budget)} units is below unconstrained demand (${fmtNum(p.baselineDemand)} units). The solved bid price is ${fmtPrice(p.lambda)} per unit — the marginal profit value of one more unit of capacity.`
      : `Capacity is currently slack (budget ${fmtNum(p.budget)} units ≥ demand ${fmtNum(p.baselineDemand)}), so the bid price is $0.00 and this reduces to per-unit pricing. Lower the sellable-capacity setting to make it bind.`;
  }
  if (techId === "robust") return `Prices are set for the worst case inside the elasticity ±${fmtPctPlain(cfg && 0.25)} and cost uncertainty box, so moves are more defensive than the point-estimate methods.`;
  if (techId === "bayesian") return `Elasticity is treated as a distribution scaled by each unit's confidence; low-confidence units are automatically priced more cautiously.`;
  if (techId === "multiobjective") return `The recommendation blends profit and revenue; shift the weight in the rail to trace the trade-off between margin and top-line growth.`;
  if (techId === "closedform") return `Recommendations come from the analytic markup P = c·|e|/(|e|-1), then bounded by the guardrails.`;
  return `Each unit is priced independently by searching the price grid for the objective maximum.`;
}

const tone = (v) => (Math.abs(v) < 1e-9 ? "neutral" : v > 0 ? "pos" : "neg");

export function explain(view, ctx) {
  const { result, cfg, techId, selected } = ctx;
  const e = result.exec;
  const t = TECHNIQUE_MAP[techId];
  const base = { technique: { name: t ? t.name : techId, note: techNote(techId, result, cfg) } };

  if (view === "portfolio") {
    const cats = [...result.categories].sort((a, b) => b.profitChange - a.profitChange);
    const topCat = cats[0], worstCat = cats[cats.length - 1];
    const invBound = result.rows.filter((r) => r.inventoryBinding).length;
    const promos = e.promoRecommended;
    return {
      ...base,
      title: "Portfolio overview",
      summary: `Under the ${t ? t.short : techId} method, the recommended strategy moves expected profit from ${fmtMoney(e.totalCurrentProfit)} to ${fmtMoney(e.totalExpectedProfit)} (${fmtPct(e.profitChangeRate)}) and revenue by ${fmtPct(e.revenueChangeRate)}, across ${e.totalItems} decision units.`,
      readings: [
        { label: "Profit uplift", text: `${fmtPct(e.profitChangeRate)} (${fmtMoney(e.totalProfitChange)}) versus current pricing.`, tone: tone(e.profitChangeRate) },
        { label: "Action mix", text: `${e.increase} increases, ${e.decrease} decreases, ${e.hold} holds; average move ${fmtPct(e.avgPriceChangeRate)}.`, tone: "neutral" },
        { label: "Strongest category", text: `${topCat.category} contributes ${fmtMoney(topCat.profitChange)} (${fmtPct(topCat.profitChangeRate)}) from ${topCat.items} units.`, tone: tone(topCat.profitChange) },
        { label: "Weakest category", text: `${worstCat.category} at ${fmtMoney(worstCat.profitChange)} (${fmtPct(worstCat.profitChangeRate)}) — the least accretive segment.`, tone: tone(worstCat.profitChange) },
        { label: "Constraints", text: `${invBound} unit(s) inventory-bound; ${promos} promotion(s) recommended.`, tone: invBound ? "warn" : "neutral" },
      ],
      guidance: [
        "The scenario bars compare all four strategies; the Recommended bar is the conservative, guardrail-bounded plan you would actually implement.",
        "Revenue-max and Profit-max bars show the unconstrained ceilings — the gap to Recommended is the cost of the safety guardrails.",
        "Category bars rank where profit is won or lost; start rollout with the strongest segment.",
      ],
    };
  }

  if (view === "sku") {
    const r = result.rows.find((x) => x.itemId === (selected && selected.itemId)) || result.rows[0];
    const flags = [];
    if (r.inventoryBinding) flags.push("inventory-bound");
    if (r.competitorBinding) flags.push("competitor-capped");
    if (r.stepBinding) flags.push("implementation-step-capped");
    const elasticTxt = Math.abs(r.elasticity) > 1 ? "elastic (demand reacts strongly to price)" : "inelastic (demand is relatively price-insensitive)";
    return {
      ...base,
      title: `Unit ${r.itemId} — ${r.category}`,
      summary: `${r.itemId} is ${elasticTxt} at elasticity ${r.elasticity.toFixed(2)}. The recommendation moves price ${fmtPrice(r.currentPrice)} → ${fmtPrice(r.recommendedPrice)} (${fmtPct(r.priceChangeRate)}) for ${fmtPct(r.profitChangeRate)} expected profit.`,
      readings: [
        { label: "Recommended price", text: `${fmtPrice(r.recommendedPrice)} (${fmtPct(r.priceChangeRate)}); margin ${fmtPctPlain(r.optimizedMarginRate)}.`, tone: tone(r.priceChangeRate) },
        { label: "Profit impact", text: `${fmtMoney(r.profitChange)} (${fmtPct(r.profitChangeRate)}); revenue ${fmtPct(r.revenueChangeRate)}, demand ${fmtPct(r.demandChangeRate)}.`, tone: tone(r.profitChange) },
        { label: "Technique optimum", text: `Unclamped optimum ${fmtPrice(r.techniqueOptimumPrice)}; guardrails ${r.recommendedPrice === r.techniqueOptimumPrice ? "did not bind" : "pulled it to " + fmtPrice(r.recommendedPrice)}.`, tone: "neutral" },
        { label: "Binding constraints", text: flags.length ? "This unit is " + flags.join(", ") + "." : "No guardrail is actively binding — the optimum sits inside all limits.", tone: flags.length ? "warn" : "pos" },
      ],
      guidance: [
        "The curve plots profit (bold), revenue, and demand (faint) against price; the four markers are Current, Revenue-max, Profit-max and Recommended.",
        "Shaded bands show the competitor-price window and the margin-infeasible region — the recommendation is always kept out of the red.",
        "Drag anywhere on the curve to test an arbitrary price and read demand, revenue, profit and margin instantly.",
      ],
    };
  }

  if (view === "sensitivity") {
    const rows = result.rows;
    const meanElas = rows.reduce((a, r) => a + r.elasticity, 0) / rows.length;
    return {
      ...base,
      title: "Sensitivity analysis",
      summary: `The grid re-optimizes the entire portfolio under the ${t ? t.short : techId} method for every combination of scaled elasticity (rows) and unit cost (columns), showing how the recommended uplift holds up when assumptions move.`,
      readings: [
        { label: "Base case", text: `At ×1.00 / ×1.00 the recommended plan delivers ${fmtPct(e.profitChangeRate)} profit and ${fmtPct(e.revenueChangeRate)} revenue.`, tone: tone(e.profitChangeRate) },
        { label: "Elasticity risk", text: `More elastic assumptions (×1.20) shrink price headroom; more inelastic (×0.80) expand it. Read down a column to isolate elasticity.`, tone: "neutral" },
        { label: "Cost risk", text: `Higher unit cost (×1.10) lifts the margin floor and can raise recommended prices. Read across a row to isolate cost.`, tone: "neutral" },
        { label: "Portfolio elasticity", text: `Mean elasticity across the catalog is ${meanElas.toFixed(2)}.`, tone: "neutral" },
      ],
      guidance: [
        "Green cells are profit-accretive, rose cells dilutive; darker means larger magnitude.",
        "A plan that stays green across the whole grid is robust to assumption error — a good sign before rollout.",
        "If one corner turns rose, that assumption combination is the risk to validate before pricing.",
      ],
    };
  }

  if (view === "validation") {
    const failing = result.validation.filter((v) => v.check !== "All checks passed" && v.passRate < 1);
    const all = result.validation.find((v) => v.check === "All checks passed");
    return {
      ...base,
      title: "Constraint validation",
      summary: `Every recommendation is checked against the pricing, margin, competitor, implementation-step and inventory guardrails. ${all.passed}/${all.total} units pass all checks simultaneously.`,
      readings: [
        { label: "Overall", text: `${all.passed}/${all.total} units clear every guardrail (${fmtPctPlain(all.passRate)}).`, tone: all.passRate >= 1 ? "pos" : "warn" },
        ...(failing.length
          ? failing.map((v) => ({ label: v.check, text: `${v.passed}/${v.total} pass (${fmtPctPlain(v.passRate)}) — ${v.failed} unit(s) need review.`, tone: "warn" }))
          : [{ label: "Guardrails", text: "No individual constraint is violated by any unit.", tone: "pos" }]),
        { label: "Binding flags", text: `${result.rows.filter((r) => r.inventoryBinding).length} inventory-bound, ${result.rows.filter((r) => r.competitorBinding).length} competitor-capped, ${result.rows.filter((r) => r.stepBinding).length} step-capped.`, tone: "neutral" },
      ],
      guidance: [
        "A binding flag is not a failure — it means a guardrail is actively shaping the recommendation to keep it safe.",
        "The flags table shows exactly which limit constrains each unit, so you can relax a guardrail deliberately if the business allows.",
        "Aim for 100% on 'All checks passed' before exporting a plan for execution.",
      ],
    };
  }

  if (view === "report") {
    const ready = result.validation.find((v) => v.check === "All checks passed");
    const priority = [...result.rows].sort((a, b) => b.profitChange - a.profitChange)[0];
    return {
      ...base,
      title: "Decision package",
      summary: `The report packages the ${t ? t.short : techId} recommendation into a stakeholder-ready decision record: ${fmtMoney(e.totalProfitChange)} expected profit uplift, ${fmtPct(e.revenueChangeRate)} revenue change, and ${ready ? ready.passed + "/" + ready.total : "all"} units clearing the complete control set.`,
      readings: [
        { label: "Financial case", text: `${fmtMoney(e.totalExpectedProfit)} expected profit after optimization, up ${fmtPct(e.profitChangeRate)} from current pricing.`, tone: tone(e.profitChangeRate) },
        { label: "Highest-value action", text: `${priority.itemId} contributes ${fmtMoney(priority.profitChange)} expected profit at a ${fmtPct(priority.priceChangeRate)} price move.`, tone: tone(priority.profitChange) },
        { label: "Control readiness", text: `${ready ? ready.passed + "/" + ready.total : e.totalItems + "/" + e.totalItems} units pass all guardrails; ${result.rows.filter((r) => r.inventoryBinding || r.competitorBinding || r.stepBinding).length} are actively shaped by a binding control.`, tone: ready && ready.passRate < 1 ? "warn" : "pos" },
        { label: "Export scope", text: "PDF provides the executive narrative and evidence; CSV provides the unit-level actions and binding-control fields.", tone: "neutral" },
      ],
      guidance: [
        "Confirm the method, configuration timestamp and data owner before circulating the PDF.",
        "Use the CSV for approval, implementation tracking and downstream workflow integration.",
        "Treat the package as decision support: commercial ownership and execution approval remain with the pricing team.",
      ],
    };
  }

  if (view === "data") {
    const complete = result.rows.filter((r) => Number.isFinite(r.elasticity) && Number.isFinite(r.currentPrice) && Number.isFinite(r.unitCost)).length;
    return {
      ...base,
      title: "Input readiness",
      summary: `${complete}/${result.rows.length} catalog rows contain the core price, cost and elasticity inputs used by the optimizer. Changes made here immediately recalculate every analytical and reporting page.`,
      readings: [
        { label: "Catalog coverage", text: `${result.rows.length} decision units across ${result.categories.length} categories.`, tone: "neutral" },
        { label: "Core completeness", text: `${complete}/${result.rows.length} rows have finite current price, unit cost and elasticity values.`, tone: complete === result.rows.length ? "pos" : "warn" },
        { label: "Data lineage", text: "CSV imports remain browser-local; the demo does not transmit catalog data to a server.", tone: "neutral" },
      ],
      guidance: [
        "Validate elasticity sign and confidence before relying on the recommended action.",
        "Keep units, time period and demand scale consistent across every row.",
        "Use stable item identifiers so exported actions can be reconciled with source systems.",
      ],
    };
  }

  if (view === "help") {
    return {
      ...base,
      title: "Operating guidance",
      summary: "The guide explains the decision workflow from catalog preparation through method selection, stress testing, control review and stakeholder export.",
      readings: [
        { label: "Recommended sequence", text: "Inputs → configuration → overview → recommendations → stress test → controls → method comparison → report.", tone: "neutral" },
        { label: "Approval principle", text: "A positive modeled uplift is not sufficient by itself; confirm data quality, downside behavior, controls and action ownership.", tone: "warn" },
      ],
      guidance: [
        "Start with the Quick Start for a first run, then use the User Guide for interpretation and governance.",
        "Engineering and analytics teams should use the Technical Reference for architecture, formulas and extension points.",
      ],
    };
  }

  if (view === "techniques") {
    return {
      ...base,
      title: "Technique comparison",
      summary: `Every optimization method is run on the current catalog and settings so you can compare the resulting profit and revenue uplift, average price move, and guardrail compliance side by side before committing to one.`,
      readings: [
        { label: "Reading the table", text: "Each row is a full portfolio re-optimization under that method. Profit and revenue columns are versus current pricing.", tone: "neutral" },
        { label: "Point vs uncertainty", text: "Grid and Closed-form price the point estimate; Robust and Bayesian trade some upside for protection against elasticity error.", tone: "neutral" },
        { label: "Coupling", text: "Lagrangian is the only method that couples the portfolio through a shared capacity limit and reports a bid price.", tone: "neutral" },
      ],
      guidance: [
        "Pick the method whose assumptions match your situation, not simply the highest number — the highest uplift often carries the most estimation risk.",
        "Use Apply to switch the whole app to a method and inspect it in detail on the other tabs.",
        "If methods disagree sharply, that divergence is itself information about how uncertain the pricing decision is.",
      ],
    };
  }

  return {
    ...base,
    title: "This page",
    summary: "Live analysis is available on the analytical pages (Portfolio, SKU detail, Sensitivity, Validation, Techniques).",
    readings: [],
    guidance: [],
  };
}
