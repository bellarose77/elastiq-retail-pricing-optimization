# ELASTIQ final application release

## Included deliverables

- One-click Windows launcher: `start.bat`
- One-click Linux/macOS launcher: `start.sh`
- Optional full analytical refresh: add `--refresh-data`
- React/Vite pricing decision workbench
- Primary Deep Analysis Runner with random Small, Medium and Large datasets,
  all-six comparison, six-stage live pipeline, multi-resolution optimization,
  stress re-optimization, Monte Carlo risk, consensus, exact workload counters,
  result analysis, cross-test history, and JSON run records
- Single-method, multi-method comparison, and per-item hybrid execution
- Initial input-data profile with portfolio and category analysis
- Eight-stage Python analytical pipeline
- Packaged synthetic demonstration data and generated model evidence
- High-level team overview PDF
- Detailed layered technical system guide PDF
- CSV and PDF decision exports
- Automated Python, browser, parity, build, and dependency checks

## Quick start

Windows: double-click `start.bat`.

Linux/macOS:

```sh
chmod +x start.sh
./start.sh
```

The first launch installs the exact web dependencies. The application then
opens at `http://127.0.0.1:5173`.

## Validation status

Regenerate this table with `python scripts/generate_release_status.py` —
it runs the Python suite, the frontend engine-validation suite, the
production build, and `npm audit`, and rewrites the rows below from what
those commands actually reported. Do not hand-edit the table; a hand-typed
number here has already drifted from reality once before (see `CHANGELOG.md`,
2026-07-31, on `ENGINE_VALIDATION.txt`).

<!-- VALIDATION_TABLE_START -->
| Check | Result |
| --- | --- |
| Python tests | 629 passed |
| Browser optimization methods | 6 of 6 passed all decision controls |
| Python/JavaScript parity | Passed (4 candidates, 3 grids, 4 roundings) |
| Frontend production build | Passed (main bundle 314.89 kB, 93.07 kB gzip) |
| Dependency audit | 0 known vulnerabilities |
| Live scenario and hybrid browser checks | Passed |
| High-level PDF | 9 pages present (page count verified automatically; visual review is still manual) |
| Technical PDF | 22 pages present (page count verified automatically; visual review is still manual) |
<!-- VALIDATION_TABLE_END -->

## Intended use

The packaged application is ready for demonstration, team training, reviewed
scenario analysis, and controlled pilot preparation. The included data is
synthetic. Live price publication additionally requires authenticated data
connections, retailer-specific causal validation, user approvals, monitoring,
audit retention, and rollback controls.
