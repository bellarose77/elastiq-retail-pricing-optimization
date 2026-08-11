import { fmtPctPlain } from "./format.js";

/* Metadata + documentation for every optimization technique.
   `params` drives the dynamic controls shown in the config rail. */
export const TECHNIQUE_LIST = [
  {
    id: "grid",
    name: "Constant-Elasticity Grid Search",
    short: "Grid search",
    tagline: "Enumerate the price grid; pick the feasible price that maximizes the chosen objective.",
    objectiveAware: true,
    portfolioCoupled: false,
    params: [],
    family: "Point estimate",
    summary:
      "The baseline method. For each unit it evaluates every price on a discrete grid (default 1% steps within the allowed range), applies the constant-elasticity demand model, and selects the feasible price maximizing the chosen objective (profit, revenue, or units). Simple, transparent, and exact on the grid.",
    when: "Default choice. Best when elasticity estimates are trusted and each unit can be priced independently.",
    math: "q(P)=q0*(P/P0)^e ; choose argmax over the grid of the selected objective, subject to margin, competitor, step and inventory guardrails.",
    strengths: ["Fully transparent and auditable", "Respects the exact objective", "No distributional assumptions"],
    limits: ["Ignores estimation uncertainty", "Prices each unit in isolation (no capacity coupling)", "Resolution limited by grid step"],
  },
  {
    id: "closedform",
    name: "Closed-Form Marginal (Amoroso–Robinson)",
    short: "Closed-form",
    tagline: "Analytic profit optimum from the elasticity — marginal revenue equals marginal cost.",
    objectiveAware: true,
    portfolioCoupled: false,
    params: [],
    family: "Point estimate",
    summary:
      "Uses the analytic optimality condition for constant-elasticity demand instead of a grid. For profit, the optimal price is a fixed markup over unit cost determined solely by elasticity: P* = c·|e|/(|e|-1) for elastic items. Inelastic items (|e|<1) have no interior profit maximum, so the method pushes toward the ceiling. The result is then bounded by the same guardrails.",
    when: "When you want the exact economic optimum and a clean markup interpretation, free of grid resolution.",
    math: "Profit-max: P* = c*|e|/(|e|-1), |e|>1. Revenue-max: toward the floor if elastic, ceiling if inelastic. Then clamp to guardrails.",
    strengths: ["Exact, grid-free optimum", "Clear markup / Lerner-index interpretation", "Fast"],
    limits: ["Assumes the point elasticity is correct", "Undefined interior optimum when inelastic", "No uncertainty or coupling"],
  },
  {
    id: "robust",
    name: "Robust Worst-Case (Min–Max)",
    short: "Robust",
    tagline: "Maximize the worst-case profit across an elasticity × cost uncertainty box.",
    objectiveAware: false,
    portfolioCoupled: false,
    params: [
      { key: "elasUncertainty", label: "Elasticity uncertainty", min: 0, max: 0.5, step: 0.01, fmt: (x) => "±" + fmtPctPlain(x) },
      { key: "costUncertainty", label: "Cost uncertainty", min: 0, max: 0.3, step: 0.01, fmt: (x) => "±" + fmtPctPlain(x) },
    ],
    family: "Uncertainty-aware",
    summary:
      "Treats elasticity and unit cost as uncertain within a bounded box and prices for the worst case inside it. For each candidate price it computes profit at the corners of the uncertainty set and keeps the minimum, then chooses the price that maximizes that guaranteed floor. Produces more defensive moves as uncertainty widens.",
    when: "When elasticity is noisy, costs may drift, or a downside guarantee matters more than the expected best case.",
    math: "max_P min over e in [e(1±d)], c in [c(1±g)] of profit(P; e, c).",
    strengths: ["Protects against estimation error", "Tunable conservatism", "No probability distribution required"],
    limits: ["Can leave upside on the table", "Corner-based set is an approximation", "Sensitive to the chosen box width"],
  },
  {
    id: "bayesian",
    name: "Bayesian Expected-Profit",
    short: "Bayesian",
    tagline: "Maximize expected profit integrating over an elasticity posterior scaled by confidence.",
    objectiveAware: false,
    portfolioCoupled: false,
    params: [{ key: "priorStrength", label: "Prior spread", min: 0, max: 1.5, step: 0.05, fmt: (x) => x.toFixed(2) + "×" }],
    family: "Uncertainty-aware",
    summary:
      "Places a Gaussian posterior on each unit's elasticity whose spread grows as confidence falls, then maximizes expected profit integrated over that posterior using 5-point Gauss–Hermite quadrature. Low-confidence units are automatically priced more cautiously; high-confidence units behave like the point estimate.",
    when: "When elasticity estimates carry per-unit confidence and you want risk folded in probabilistically rather than as a hard box.",
    math: "e ~ Normal(e_hat, s^2), s = |e_hat|·k·(1-confidence). max_P E[profit(P; e)] via Gauss–Hermite.",
    strengths: ["Uses per-unit confidence directly", "Smoothly regularizes noisy items", "Principled expectation, not a corner"],
    limits: ["Assumes a Gaussian posterior", "Prior spread must be chosen", "Expected value ignores tail risk"],
  },
  {
    id: "multiobjective",
    name: "Multi-Objective Scalarization",
    short: "Multi-objective",
    tagline: "Optimize a weighted, current-normalized blend of profit and revenue.",
    objectiveAware: false,
    portfolioCoupled: false,
    params: [{ key: "profitWeight", label: "Profit ↔ revenue weight", min: 0, max: 1, step: 0.05, fmt: (x) => fmtPctPlain(x) + " profit" }],
    family: "Trade-off",
    summary:
      "Traces the profit–revenue trade-off by maximizing a weighted sum of the two, each normalized by its current value so the weight is meaningful. Weight 1 reproduces profit-max; weight 0 reproduces revenue-max; intermediate weights yield balanced interior prices that grow the top line while protecting margin.",
    when: "When the business wants both margin and growth and needs an explicit, tunable balance between them.",
    math: "max_P [ w · profit(P)/profit_0 + (1-w) · revenue(P)/revenue_0 ].",
    strengths: ["Explicit, interpretable trade-off", "Spans the Pareto frontier via one weight", "Balances growth and margin"],
    limits: ["Weight choice is a judgment call", "Linear scalarization misses non-convex frontier points", "Point-estimate elasticity"],
  },
  {
    id: "lagrangian",
    name: "Lagrangian Capacity Pricing (Bid-Price)",
    short: "Lagrangian",
    tagline: "Portfolio-coupled pricing under a shared capacity limit, priced via a shadow value.",
    objectiveAware: false,
    portfolioCoupled: true,
    params: [{ key: "capacityUtilization", label: "Sellable capacity", min: 0.3, max: 1, step: 0.05, fmt: (x) => fmtPctPlain(x) + " of demand" }],
    family: "Constrained portfolio",
    summary:
      "Prices the whole portfolio under a shared sellable-capacity limit rather than unit by unit. It relaxes a shared sellable-capacity limit (expressed as a fraction of unconstrained demand) with a Lagrange multiplier (a bid price) and solves for the multiplier that makes total expected demand meet the budget, subtracting lambda × demand from each unit's profit. The multiplier is the marginal value of one unit of capacity — a classic revenue-management shadow price.",
    when: "When supply, shelf space, or fulfilment capacity is shared and limited, so selling more of one unit competes with others.",
    math: "max sum_i [ profit_i(P_i) - lambda · demand_i(P_i) ] ; choose lambda so sum_i demand_i ≤ capacity. lambda = bid price.",
    strengths: ["Couples the portfolio through a real constraint", "Yields an interpretable shadow price", "Grounded in RM theory (bid prices)"],
    limits: ["Needs a meaningful capacity figure", "Multiplier solved to tolerance", "Reduces to per-unit pricing when capacity is slack"],
  },
];

export const TECHNIQUE_MAP = Object.fromEntries(TECHNIQUE_LIST.map((t) => [t.id, t]));
