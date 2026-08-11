# ELASTIQ Pricing Optimization - Technical reference

## System boundary

The repository has two complementary surfaces:

- A Python eight-stage analytical pipeline under `src/pipeline`, with model,
  retrieval, optimization, validation, and reporting modules under `src`.
- A React/Vite decision-workbench under `app/frontend`, with an independent
  JavaScript pricing engine and CSV/PDF exports.

The browser surface is suitable for scenario review and demonstration. The
Python pipeline is the source of modelled evidence. A manual browser row is not
treated as causal evidence and therefore cannot become an executable price by
default.

## Demand and profit model

Candidate demand uses a constant-elasticity response:

```text
Q(P) = Q0 * (P / P0) ^ elasticity
```

`Q0` is the next-period baseline forecast at the current price `P0`. The engine
applies only bounded, provenance-bearing promotion and market adjustments.
Realized sales are `min(unconstrained demand, available inventory)` in inventory
cap mode. Profit is:

```text
profit(P) = (P - unit_cost) * realized_sales - promotion_cost
```

Current and candidate profits use identical inventory and promotion semantics.
This is essential: comparing an uncapped current case with a capped candidate
creates artificial negative lift.

## Causal elasticity

The descriptive elasticity stage fits log demand on log price and controls. It
is useful for diagnosis but can be biased because prices often respond to
anticipated demand. The causal stage uses two-stage least squares by product:

1. Predict log price using observed cost-side instruments and controls.
2. Regress log quantity on instrumented log price and controls.

No instrument is synthesized when required columns are absent. Reliability
requires an economically plausible sign, sufficient observations, and a strong
first stage. Step 08 uses causal estimates by default and emits an explicit hold
otherwise.

## Forecasting and leakage control

Forecast evaluation and tuning split on unique calendar dates, not row
positions. Every store-product row for a date is assigned to the same partition.
Lag and rolling features are backward looking. Next-period rows are explicitly
constructed for one day ahead, with stale promotions cleared and known or
trailing context identified. The evaluation report includes MAE, MAPE, WAPE,
and naive lag/seasonal baselines.

## Promotion readiness

The promotion model excludes variables affected by treatment, including
selling price, competitor price, inventory, and stockout state. It evaluates on
a complete-date holdout and reports propensity overlap, stockout censoring, and
ranking quality. Uplift enters pricing only if the study passes every readiness
gate. Model scores can still be saved for analysis when the action gate is
closed; they are labelled non-actionable.

## Time-safe market retrieval

For each decision row, retrieval is restricted to notes satisfying all of:

- note date is on or before the decision date;
- note date is inside the configured 120-day lookback;
- category and region metadata match exactly when supplied.

Retrieved notes are converted into structured features and a bounded market
adjustment. Evidence count, date range, filters, and score travel with the
recommendation. Free text never directly selects a price.

## Optimization controls

The engine searches only within configured price-change bounds and checks
minimum margin, demand, profit, competitor, inventory, and implementation-step
rules. An independent validator recomputes the selected recommendation. If the
current price is feasible, the no-harm rule prevents returning a worse value of
the configured objective. Cross-price effects are excluded from the final
decision path until a genuine joint portfolio solver is implemented.

The browser exposes six optimization techniques. Technique tests require each
to satisfy the same controls, cross-language fixtures compare Python and
JavaScript demand calculations, and distinctness assertions prevent all six
methods from silently collapsing to identical output.

## Live experiment architecture

`scenarioGenerator.js` creates 5-2,000 deterministic retail decision rows from
a numeric seed and scenario profile. Generated rows include identity, price,
cost, competitor, forecast demand, inventory, causal elasticity provenance,
promotion readiness, confidence, and bounded market signal.

`inputAnalysis.js` profiles these rows before execution. `experimentWorker.js`
runs selected techniques in a module Web Worker and posts stage events back to
React. `deepAnalysis.js` controls multi-resolution refinement, shocked-market
re-optimization, fixed-policy Monte Carlo simulation, and action consensus.
Each method receives the same input rows and pricing policy.

The opening screen exposes Small (50), Medium (250), and Large (1,000) cards,
plus a 5-2,000 custom size and the current workspace dataset. Fresh random data
is generated for each run by default. All six methods are initially selected.
Worker events drive a six-stage visual pipeline: Start and profile, Optimize and
refine, Stress test, Simulate risk, Validate or hybrid-route, and Finish. The
default Deep mode uses 1%, 0.5%, and 0.25% price-grid resolutions, 1,000 draws
per method, and a size-adjusted number of complete shocked-market reruns. The
Research mode adds a 0.125% grid, 3,000 draws, and more stress states. Standard
is the lightweight mode.

Instrumentation inside `engine.js` counts portfolio optimizations, candidate
price/promotion evaluations, uncertainty-aware demand and profit evaluations,
and Lagrangian capacity-solver iterations. These exact counters are displayed
live and stored in the run record. The worker yields between completed units of
real work to keep React responsive; it does not add artificial waiting.

Hybrid execution considers all completed method results for each row and filters
to candidates passing row-level controls. When available, inventory-pressure
items route to Multi-objective, low-confidence or high-market-variation items to
Robust, and eligible high-uplift items to Bayesian. Standard cases maximize the
configured objective; ties within tolerance use the smaller absolute price move.
The completed record retains the frozen configuration, refinement convergence,
every stress scenario, method risk percentiles, consensus, exact operation
counters, method summaries, hybrid mix, top decisions, input profile, and timing.
It can be exported as JSON or as a two-page Deep Analysis Run PDF with embedded
fonts and computation evidence.

## Core outputs

| Output | Purpose |
| --- | --- |
| `causal_elasticity_by_product.csv` | Per-item causal estimates and diagnostics |
| `promotion_model_readiness.csv` | Promotion action gate and reasons |
| `xgboost_forecast_metrics.csv` | Forecast accuracy and evaluation dates |
| `next_period_demand_forecast.csv` | Quantity baseline for optimization |
| `rag_market_features.csv` | Time-safe structured market evidence |
| `pricing_recommendations.csv` | Final item-level decisions and provenance |
| `pricing_validation_report.csv` | Independently recomputed constraint audit |

## Run and validate

From the repository root:

```bash
python -m pip install -r requirements.txt
make pipeline
make test-python
make test-js
```

For the workbench:

```bash
cd app/frontend
npm ci
npm run dev
```

Use `npm run build` for a production bundle and `npm run validate` for the
browser engine suite.

## Production-readiness boundary

Before execution on live prices, add an authenticated data and approval layer,
schema/version contracts, secrets management, monitoring, alert thresholds,
model registry, reproducible releases, audit retention, user roles, and a
rollback mechanism. Validate causal assumptions on the retailer's actual price
process and run a controlled pilot. The included retrieval layer is deterministic
and local; a production embedding or vector service must preserve the same
as-of-time and metadata rules.
