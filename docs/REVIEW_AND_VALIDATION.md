# Independent review and validation record

## Scope

This review compared the uploaded original repository, the uploaded revised
repository, and its changelog. It reproduced the revised test and pipeline
results, challenged the assumptions connecting model outputs to final prices,
implemented additional safeguards, and reran unit, integration, parity, build,
and document-render checks.

## What the revised version fixed well

The revision corrected the most serious original defect: descriptive
elasticities had the wrong sign for 8 of 10 synthetic products, while the causal
analysis was not used by the optimizer. The revised version wired product-level
IV estimates into pricing, strengthened first-stage diagnostics, accounted for
every item, added engine parity fixtures, and added regression tests against
known synthetic ground truth. Those changes were retained.

## Additional findings and resolution

| Finding after revision | Risk | Resolution in this version |
| --- | --- | --- |
| Row-position time splits shared dates between train and evaluation | Optimistic accuracy | Split on unique complete dates |
| Optimizer reused realized historical sales | Wrong future baseline | Wire explicit next-period forecast |
| Current profit was uncapped while candidates were inventory capped | Artificial lift or loss | Apply identical semantics to both |
| Promotion features included post-treatment data | Leakage and biased uplift | Pretreatment-only feature set |
| Promotion scores remained actionable despite weak readiness | Margin-destructive promotions | Overlap, censoring, ranking, and hold gates |
| Retrieval could use future or mismatched notes | Look-ahead bias | As-of date, lookback, exact metadata filters |
| Missing instruments could be fabricated | False causal claim | Fail clearly unless observed instruments exist |
| Browser accepted manual elasticity as actionable | Unsafe interface divergence | Causal provenance gate and explicit hold |
| CSV parsing used comma splitting | Corrupted quoted fields | Quote-aware parser and strict validation |
| Cross-price heuristic was not a joint solver | Misleading portfolio effect | Disabled in decision pipeline |
| Candidate could be worse than a feasible hold | Objective regression | No-harm fallback and audit check |
| PDF libraries had known advisories and inflated initial bundle | Supply-chain and performance risk | Updated libraries and lazy loading |

## Reproduced results on the included synthetic data

| Evidence | Result |
| --- | --- |
| Python suite | 358 tests passed |
| Browser methods | 6 of 6 passed, 10 of 10 controls each |
| Python/JavaScript parity | 4 candidate, 3 grid, and 4 rounding cases passed |
| Frontend production build | passed |
| Dependency audit | 0 known npm vulnerabilities |
| Causal recovery | 10 of 10 reliable product estimates |
| First-stage strength | F-statistic range 311 to 378 |
| Forecast MAE | 6.39 units |
| Forecast MAPE / WAPE | 19.84% / 17.92% |
| Best naive MAE | 10.67 units |
| Model improvement over best naive MAE | 40.1% |
| Next-period recommendations | 50 accounted for; 46 priced; 4 explicit holds |
| Independent decision validation | 46 of 46 priced rows passed |
| Expected sample profit / revenue delta | +$443 / +$307 |

The profit and revenue deltas are outputs of the synthetic scenario. They are
not an estimate of real-world value.

## Remaining limitations

1. The source data and known elasticities are synthetic.
2. The IV exclusion restriction still needs business and statistical validation
   on live data; a high first-stage F-statistic is necessary but not sufficient.
3. The sample promotion study is deliberately held because evaluation stockout
   censoring is 22.0%. It should not be actioned until identification and
   overlap improve.
4. Half of the current structured market features have no cross-row variance in
   the sample, limiting their decision value.
5. The final optimization is item-wise. Cannibalization and substitution require
   a jointly constrained portfolio model and controlled validation.
6. Point forecasts and point elasticities understate uncertainty. Production
   decisions should incorporate intervals, downside risk, and drift monitoring.
7. The application is decision support, not an authenticated price-publishing
   system. Approval, permissions, audit, and rollback remain external.

## Release recommendation

The repository is suitable as a substantially improved analytical prototype and
demonstration workbench. It is not yet suitable for unattended live execution.
Proceed to a limited, reviewed pilot only after replacing synthetic inputs,
validating causal assumptions, closing the promotion readiness gate, and adding
operational governance.
