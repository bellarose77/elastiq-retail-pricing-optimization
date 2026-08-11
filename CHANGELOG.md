# Changelog

## 2026-07-31 - Independent hardening and decision-safety review

This version preserves the revised application's causal-elasticity correction,
then closes the remaining gaps between model evaluation and a price that can be
defended in production. All reported results below are from the included
synthetic data and are validation evidence, not a forecast of live commercial
performance.

### Decision-safety changes

- Split forecasting and causal evaluation on complete calendar dates, so the
  same date cannot appear in both training and evaluation sets.
- Made the next-period forecast the optimizer's quantity baseline. Historical
  realized sales are no longer reused as if they were a future forecast.
- Calculated both current-price and candidate-price profit under the same
  inventory cap. This removes false negative lift when current demand also
  exceeds stock on hand.
- Added a no-harm fallback: if the current price is feasible, an optimizer may
  not return a candidate with a worse configured objective.
- Disabled the one-pass cross-price heuristic in the final decision pipeline;
  it is not a joint portfolio solver and therefore is not suitable evidence for
  an executable recommendation.
- Continued the revised version's default rule that non-causal elasticity is a
  hold, and made the browser engine apply the same rule.

### Promotion and market-evidence changes

- Removed post-treatment fields such as selling price, competitor price,
  inventory, and stockout status from the promotion uplift feature set.
- Added overlap, stockout-censoring, ranking-quality, and date-holdout readiness
  gates. On the sample data the promotion model is correctly marked `HOLD`
  because 22.0% of evaluation rows are stockout-censored; no promotion is passed into
  pricing as an actionable uplift.
- Retrieval is now as-of-time safe, has a 120-day lookback, and filters exactly
  on category and region. Future market notes cannot inform an earlier price.
- Market text produces bounded context adjustments rather than unconstrained
  price effects, and each recommendation carries evidence provenance.

### Interface, reporting, and supply-chain changes

- Replaced comma splitting with a quote-aware CSV parser and added aliases for
  the Python pipeline's exported column names.
- Manual browser rows are explicitly non-causal and remain on hold until causal
  evidence is supplied.
- Added provenance and governance fields to CSV and PDF exports.
- Updated `jspdf` and `jspdf-autotable`, reducing the production main bundle
  from 648.41 kB to 256.56 kB by loading PDF code only when requested.
- Current `npm audit` result: 0 known vulnerabilities.

### Validation evidence

| Check | Result |
| --- | --- |
| Python tests | 358 passed |
| Browser technique checks | 6 of 6 methods passed all controls |
| Cross-language parity | 4 candidate cases, 3 price grids, 4 rounding cases passed |
| Production frontend build | passed |
| End-to-end sample recommendations | 50 rows accounted for; 46 priced; 4 explicit holds |
| Independent recommendation audit | 46 of 46 priced rows passed all constraints |
| Sample expected profit delta | +$443, after consistent inventory treatment |
| Sample expected revenue delta | +$307 |

The sample action mix is 37 increases, 5 holds, and 4 decreases among priced
rows, plus 4 explicit unpriced holds. These counts are illustrative only.

## 2026-07-31 — Review remediation

A code review of the July 2026 snapshot found that the pipeline's pricing
recommendations were built on elasticities with the **wrong sign** for 8 of 10
products, and that the step designed to detect exactly that bias (step 04,
IV/2SLS) computed the correct answer and was then read by nothing. This release
fixes that and the defects found alongside it.

Every number below was measured on the repository's own synthetic dataset,
which carries ground-truth elasticities in `true_price_elasticity`.

---

### Headline: elasticity is now causally identified

| Metric | Before | After |
| --- | --- | --- |
| Elasticity MAE vs ground truth | 1.147 | **0.185** |
| Correct sign | 2 / 10 | **10 / 10** |
| Correctly classed elastic vs inelastic | 0 / 10 | **9 / 10** |
| Ground truth inside the 95% CI | not calibrated | **10 / 10** |
| First-stage F (per product) | n/a | **311 – 378** |
| Items with no recommendation | 7 (14%, silent) | **1** (explicit, with reason) |
| Action mix | 41 raise / 2 cut | **31 raise / 15 cut / 3 hold** |
| Reported profit lift | $884 | **$144** |

The reported lift fell by a factor of six. That is the fix working, not a
regression: because every estimated elasticity came back inelastic (|ε| < 1)
when the truth was uniformly elastic (|ε| > 1), constant-elasticity profit rose
monotonically with price and the grid search pinned itself to the +20% band
ceiling on nearly every item. With causal elasticities the optimizer finds
interior optima and cuts price on the 15 items where that is correct.

---

### Added

- **`src/models/causal.py` — `fit_group_iv_elasticities()`.** Per-product 2SLS
  price elasticity. Instruments price with the same cost-side shifters step 04
  already used, adds store fixed effects, drops within-group constant controls
  to avoid singular design matrices, and returns one row per group *always* —
  including failures — carrying `first_stage_f_statistic`,
  `is_economically_plausible`, and `is_reliable`. Per-product instruments prove
  far stronger than pooled (F 311–378 vs 37) because within a product the cost
  indices explain price much better.

- **`PricingOptimizationConfig.inventory_mode`** (`"cap"` | `"constraint"`,
  default `"cap"`). Under `"cap"` realized demand truncates at stock on hand;
  selling out is an outcome, and since revenue is then price × stock the
  optimizer is correctly pushed to *raise* price on an item that stocks out.
  `"constraint"` preserves the old reject-the-scenario behaviour.

- **Elasticity provenance on every recommendation.** `elasticity_source`
  (`product_iv` / `pooled_iv` / `shrunk_ols` / `default_fallback`),
  `elasticity_is_causal`, and `baseline_is_censored` travel with each row, so
  the output file is self-documenting about the evidence behind each price.

- **Full guardrail status on every recommendation.**
  `meets_margin_constraint`, `meets_quantity_constraint`,
  `meets_profit_constraint`, `meets_all_constraints`, `inventory_binding`,
  `unconstrained_quantity`, `minimum_allowed_price`.

- **Cross-language parity testing.**
  `scripts/generate_parity_fixture.py` writes `tests/fixtures/engine_parity.json`
  from the Python engine; `app/frontend/scripts/validate-engine.mjs` replays it
  through the JavaScript engine and fails on any disagreement. The two engines
  are independent implementations of one demand model and nothing previously
  asserted they agree.

- **Technique-differentiation and plausibility assertions** in the JS suite. The
  app's headline feature is six distinct optimization techniques; five of six had
  silently collapsed to byte-identical prices and no assertion noticed, because
  every existing check tested invariants rather than behaviour.

- **`tests/test_regressions.py`** — 19 tests, one per defect, plus ground-truth
  recovery tests for elasticity (sign, MAE, CI calibration, elastic/inelastic
  classification). The absence of this last class of test is what let the sign
  error ship.

- **Naive baselines for the forecaster** (`lag-1`, seasonal `lag-7`,
  `rolling mean 7`), written to `xgboost_naive_baseline_comparison.csv`.

- **`Makefile`** encoding the eight-step order, plus `make validate` to run the
  pipeline and both test suites.

- **Feature-variance guard in step 07**, which now names any market feature that
  is constant across all rows.

### Changed

- **Step 08 sources elasticity through an explicit precedence chain**:
  `product_iv` → `pooled_iv` → `shrunk_ols` (only if economically plausible) →
  flat default. Step 04's outputs are now hard dependencies.

- **Step 08 will not price on non-causal evidence.** With
  `ALLOW_NON_CAUSAL_PRICING = False` (default), items lacking a causal
  elasticity get `status="held_no_causal_elasticity"` and a machine-readable
  reason rather than a price. All 50 items are still accounted for.

- **Step 08 baseline selection is stockout-aware.** It now prefers the most
  recent *non-stockout* day per item. On a stockout day `units_sold` is
  censored — it records what the shelf could supply, not what customers wanted
  — and the old code used such rows as both the demand baseline *and* the stock
  ceiling. Step 05 already excluded stockout rows for exactly this reason; the
  knowledge existed one module away and did not propagate. 598 censored rows are
  now excluded.

- **`validate_price_recommendations()` audits the real guardrails.** Given the
  configuration it independently re-derives the permitted price-change band, the
  margin-rate floor, the quantity floor, non-negative profit, and the competitor
  band. It previously checked three arithmetic properties (price > 0,
  price ≥ cost, change-rate self-consistency) and none of the constraints it
  implied.

- **Validation distinguishes `not_scored` from `failed`,** and the pass rate is
  computed over items actually priced. The old headline `0.86` was a coverage
  number wearing a quality label.

- **Forecaster: same-day `inventory_level` removed, `inventory_level_lag_1`
  added.** The contemporaneous value carried 61.5% of feature importance,
  correlated 0.933 with the target, and equalled it exactly on 25.9% of rows
  (every stockout day). It is also not knowable at forecast time, and the
  next-period path was copying a stale value forward into the model's single
  most important input.

  | | Before | After |
  | --- | --- | --- |
  | MAE | 4.84 | 6.36 |
  | MAPE | 15.63% | 19.84% |
  | R² | 0.901 | 0.811 |
  | Improvement vs baseline | 68% (vs *mean*) | **40.1%** (vs best naive) |

  Accuracy is lower and now honest. The model still beats the best naive
  baseline (MAE 10.62) by a wide margin.

- **JS `DEFAULT_CONFIG` aligned with Python** — `maxPriceChangeRate` 0.25 → 0.20,
  `actionThreshold` 0.01 → 0.005. These had diverged silently, so the same SKU
  produced different prices in the app and the pipeline.

- **`demoData.js` regenerated.** All ten products now carry `product_iv`
  elasticities matching ground truth, with confidence 0.47–0.72. The committed
  version had `elasticity: -1.5, elasticitySource: "default_fallback"` on 9 of
  10 — the deployed demo was running on a hard-coded constant. With real
  confidence values the low-confidence step clamp no longer binds on everything
  and all six techniques differentiate again (2 → 6 distinct price vectors).

- **`ENGINE_VALIDATION.txt` regenerated from actual output.** The committed
  record claimed 71.43% profit uplift for grid search; the code at that commit
  produced 1.45%, and −1.16% on revenue — off by ~49× with the wrong sign on
  revenue.

- **`export_demo_data.py` reads step 08's provenance** instead of re-deriving a
  weaker answer from step 03's `is_reliable` flag.

- **Dependencies pinned with upper bounds.** `pandas>=2.0.0,<2.3` in particular:
  pandas 2.2 deprecated and 3.0 removed passing grouping columns into
  `groupby(...).apply`, which broke step 06 outright on any current install.

- **`.gitignore` now excludes derived data and model binaries.** 42 MB was
  tracked, including the same XGBoost model as both `.pkl` (8.5 MB) and
  `.joblib` (26 MB). Committed artefacts had drifted from the code: the RAG
  metadata claimed a `sentence-transformers` backend that `requirements.txt`
  cannot install, and carried Windows paths from a different machine than the
  committed CSVs.

### Fixed

- **`generate_price_change_rates` injected `0.0` unconditionally.** A mandated
  increase-only band (+5% to +20%) produced the grid
  `[0.0, 0.05, 0.10, 0.15, 0.20]` and could return `hold_price` while reporting
  that it honoured the floor. Now injected only when zero lies inside the band.
  The identical bug in `priceChangeGrid` in `engine.js` is fixed the same way.

- **Post-processing bypassed two guardrails.** Confidence dampening, `.99`
  rounding, and the margin-floor clamp each move the price off the grid pick,
  but only the competitor band and inventory cap were re-checked — despite a
  comment claiming full re-validation. A reproducer with
  `minimum_expected_quantity=95` returned a recommendation expecting 90.99
  units; it now returns 95.238 with `meets_quantity_constraint=True`. New
  `_check_guardrails()` re-tests margin, quantity, profit, competitor, and
  inventory together.

- **Cannibalization sibling-average inflated by n/(n−1)** for every item whose
  own first-pass rate was NaN, because the denominator subtracted 1
  unconditionally. With 7 of 50 items failing pass 1 in the real run, this
  fired in practice.

- **`except Exception` in `_run_portfolio_pass` narrowed to `except ValueError`.**
  A mistyped column name previously produced a portfolio of silent "review" rows
  instead of a stack trace.

- **Step 06 crashed on any current pandas** at the next-period forecast. Group
  keys are now reattached explicitly rather than relying on removed
  `groupby.apply` behaviour, and `inventory_level_lag_1` is populated from the
  last observed stock level.

- **`eda.py` relied on the deprecated `observed=False` groupby default**, which
  failed one test on pandas 3.

- **Lagrangian technique ignored the capacity it collected.** `budget` was
  `capacityUtilization × unconstrained_demand`, making the "capacity
  constraint" circular — always binding at a fixed fraction of whatever demand
  happened to be — while `totalInv` was summed, reported, and discarded. Now
  `utilization × totalInv` when inventory is supplied. Its default also
  disagreed with `DEFAULT_TECHNIQUE` (0.8 vs 0.6).

### Test suite

355 passing, up from 333 passing + 1 failing. The JS suite additionally
verifies technique differentiation (6/6 distinct) and cross-language parity
(4 candidates, 3 grids, 4 roundings).

### Known limitations, unchanged

- The dataset is synthetic: 18,300 rows, 10 products × 5 stores × 366 days, with
  elasticities the repository generated itself. Nothing here has met real data.
- Nine of fifteen step 07 market features remain constant; step 07 now warns
  loudly rather than fixing the underlying corpus.
- `_run_portfolio_pass` still iterates rows and builds a DataFrame per item.
  Fine at 50 items; hours at 100k. The maths is vectorizable.
- The cannibalization pass remains a one-shot heuristic, not a joint demand
  system. The existing `TODO` at `pricing.py` documents this honestly.
- `pricing.py` is still ~2,400 lines mixing optimization, orchestration,
  summarization, and validation. It should be four modules.
- Nothing here writes a price to a till. Every recommendation remains a proposal
  requiring financial, legal, and operational review.
