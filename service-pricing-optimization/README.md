# ELASTIQ Pricing API

A standalone FastAPI microservice that serves price recommendations
computed by the ELASTIQ pipeline's price-optimization step. It does not
re-run the batch pipeline or reimplement its algorithm -- it reuses
`src/optimization` from the monorepo (see [Architecture](#architecture))
and scores requests against the pipeline's most recent output.

## Quick start

From this directory:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.

The service reads pipeline artifacts from `../data/processed/` by
default (a sibling of this directory in the monorepo checkout). Those
files are populated by running the batch pipeline first:

```bash
cd ..
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

The repository ships with pre-computed demo artifacts, so this is only
needed if you want fresh numbers.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check. |
| GET | `/items` | Every priceable `(product, store)` decision unit. |
| POST | `/recommendations` | Score requested `item_id`s (e.g. `P001_S001`); body `{"item_ids": [...]}`. Unknown ids come back under `not_found` instead of failing the request. |
| GET | `/products` | Live version of the frontend's bundled demo catalog -- one row per product (input data: price, cost, elasticity...). |
| GET | `/products/recommendations` | One pricing decision per product, aggregated across its stores. |
| POST | `/refresh` | Reload artifacts from disk, e.g. after a new pipeline run. |

Items with no causally identified elasticity are returned with
`status="held_no_causal_elasticity"` and an unchanged price -- the same
guardrail the batch pipeline enforces against pricing on correlational
evidence (see `src/optimization/dataset.py`).

## Architecture

```
app/
  main.py              FastAPI app: creates the app, configures CORS, mounts routes.
  config.py            Environment-driven settings (see Configuration below).
  schemas.py           Pydantic request/response models -- the API's I/O contract.
  api/
    routes.py          HTTP handlers. Thin: validate via schemas.py, delegate to
                        services/pricing_service.py, map results to responses.
  services/
    artifact_store.py  Loads and caches the batch pipeline's output files.
    pricing_service.py Business layer: composes src.optimization functions
                        (elasticity reconciliation, guardrails, grid search)
                        with the cached artifacts. The ONLY module here that
                        imports from src/ -- everything above it only knows
                        about plain dicts/lists and this module's functions.
```

`routes.py` never touches `src.optimization` or a DataFrame directly, and
`pricing_service.py` never touches FastAPI or Pydantic -- that boundary is
deliberate, so the actual pricing algorithm stays owned by
`src/optimization` (single source of truth, shared with the batch
pipeline) and this service stays a thin, testable HTTP wrapper around it.

Because of that reuse, this service is coupled to the monorepo layout: it
needs `src/` (and, at runtime, `data/processed/`) available as siblings
of this directory. `app/config.py` adds the repository root to
`sys.path` automatically so `from src.optimization import ...` resolves
regardless of the current working directory, as long as this service is
invoked from *within* `service-pricing-optimization/` (so its own `app`
package resolves too) -- see the Dockerfile for how the container
preserves this layout.

## Configuration

All settings are environment variables prefixed `PRICING_API_`:

| Variable | Default | Purpose |
|---|---|---|
| `PRICING_API_CORS_ALLOW_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Comma-separated list of allowed frontend origins. |
| `PRICING_API_PROCESSED_DIR` | `<repo>/data/processed` | Where to read pipeline artifacts from. |

## Tests

```bash
pip install -r requirements.txt
pytest
```

Tests run against the real demo artifacts checked into the repository
(`data/processed/*.csv`) rather than mocks -- this service's entire job
is scoring real pipeline output, so a schema drift between it and
`src/optimization` is exactly what a mock would hide.

## Docker

Build context must be the **repository root** (this service copies in
`src/` and `data/` as siblings):

```bash
cd ..
docker build -f service-pricing-optimization/Dockerfile -t elastiq-pricing-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data/processed:/srv/data/processed:ro" \
  elastiq-pricing-api
```

The volume mount keeps served recommendations in sync with whatever the
batch pipeline most recently produced, without rebuilding the image. Omit
it to use the `data/` snapshot baked into the image at build time.

## Frontend integration

The ELASTIQ frontend (`app/frontend`) calls this service directly over
HTTP for its live pricing path -- see `app/frontend/src/lib/api.js`. Point
it at a non-default URL with `VITE_PRICING_API_BASE_URL` (see the
frontend's own README/`.env.example`). CORS must allow the frontend's
origin (see Configuration above); the defaults already cover the Vite dev
server.
