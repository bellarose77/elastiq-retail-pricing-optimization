# ELASTIQ Pricing Optimization - Plain-language guide

## What this application does

ELASTIQ helps a retailer decide whether to raise, hold, or lower the price of
each item for the next selling period. It does not simply choose the price that
made the most money last time. It combines expected demand, price sensitivity,
unit cost, available inventory, promotion evidence, competitor context, and
commercial guardrails.

The output is a decision-support file. A recommendation can be reviewed and
approved by a pricing or category manager; it is not an instruction to publish
prices automatically.

## First understand the input

The Data page analyzes the rows before optimization. It reports numeric
completeness, invalid and duplicate records, price and demand ranges, current
margin, competitor position, inventory pressure, causal-evidence coverage,
promotion readiness, and category-level patterns. This profile updates whenever
data is edited or imported.

## Run live examples

The Test Case Runner generates real, reproducible workloads with Small (50),
Medium (250), Large (1,000), or custom 5-2,000 decision units. A seed recreates
the same rows. Scenario profiles
cover balanced, inventory-constrained, promotion-intensive, and volatile cases.

- **Single** executes one selected optimization method.
- **Compare** runs selected methods against the same rows and policy.
- **Hybrid** runs several methods and routes each feasible item by context:
  Multi-objective for inventory pressure, Robust for uncertainty, Bayesian for
  strong promotion evidence, and the objective leader for standard cases.

All six approaches are selected by default. Choose an analysis depth:

- **Standard** runs the baseline optimizer and a light verification sample.
- **Deep analysis** (default) uses three progressively finer price grids,
  size-adjusted market-shock re-optimization, 1,000 Monte Carlo draws per
  method, method consensus, and independent controls.
- **Research** uses four price-grid resolutions, more shocked market states,
  and 3,000 Monte Carlo draws per method.

The run happens in a dedicated browser worker. A six-stage visual pipeline
shows Start and profile, Optimize and refine, Stress test, Simulate risk,
Validate, and Finish. Exact engine counters report portfolio optimizations,
candidate price/promotion evaluations, demand and profit evaluations, and
capacity-solver iterations. These are real computations; the runner adds no
artificial waiting. Stress distributions, risk percentiles, method agreement,
control pass rates, top decisions, and the complete run record remain available
after the run.

## The idea in one example

Suppose an item sells for $10, costs $6, has 100 units available, and is
expected to sell 70 units tomorrow. If reliable evidence says a 1% price rise
reduces demand by 1.5%, the application evaluates many allowed prices around
$10. At every price it estimates demand, caps sales at inventory, calculates
revenue and profit, and checks margin and change limits. It recommends the best
price that passes every check. If the evidence is not reliable, it holds the
current price.

## What happens in the eight stages

| Stage | Plain-language purpose | Main result |
| --- | --- | --- |
| 01 Validate | Check whether the input can be trusted | Accepted rows and issue report |
| 02 Explore | Find demand, margin, discount, and stock patterns | Commercial KPI tables |
| 03 Estimate | Measure the observed price-demand relationship | Diagnostic elasticity |
| 04 Verify causality | Separate price effect from price reacting to demand | Causal elasticity and reliability gate |
| 05 Evaluate promotions | Ask whether promotions add demand | Uplift estimates or an explicit hold |
| 06 Forecast | Estimate next-period demand if the current plan continues | Item-level demand baseline |
| 07 Add market context | Convert recent relevant market notes into bounded signals | Evidence-linked market adjustment |
| 08 Optimize | Compare feasible prices and audit the selected decision | Raise, hold, lower, or not scored |

## How to read a recommendation

- `recommended_price` is the proposed next-period price.
- `expected_quantity` is modelled demand after price response, limited by
  available inventory.
- `expected_profit` and `profit_change` use the same assumptions for both the
  current and recommended prices.
- `elasticity_source` says where price sensitivity came from.
- `elasticity_is_causal` says whether the evidence passed the causal gate.
- `forecast_source` identifies the next-period demand baseline.
- `market_evidence_count` shows how many time-safe notes supported the context.
- `promotion_readiness` says whether promotion uplift was allowed into pricing.
- `status` explains whether the item was priced or held.
- `meets_all_constraints` is the final independent control result.

## Why an item may be held

Holding is a valid decision. The application holds or declines to score an item
when it lacks reliable causal elasticity, a usable next-period forecast, cost
or inventory data, or a feasible price under the configured constraints. It
also suppresses promotion uplift when the promotion study is not decision
ready. The status and reason remain in the output so that missing items are not
silently dropped.

## What the application does not prove

The included dataset is synthetic. Passing tests demonstrates that the code
follows its stated rules on this sample; it does not prove that a live retailer
will earn the modelled lift. A production rollout still needs source-system
contracts, monitored data quality, price experiments or defensible natural
experiments, approval workflow, legal and policy review, and a measured pilot
with a holdout group.

## Recommended operating process

1. Refresh and validate source data.
2. Review the causal, forecast, promotion, and market-readiness reports.
3. Run optimization only for items whose evidence gates pass.
4. Review large changes, binding constraints, and low-confidence items.
5. Approve prices through the retailer's normal workflow.
6. Measure realized demand and margin against a holdout.
7. Recalibrate, document overrides, and stop the rollout if guardrails degrade.
