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
| GET | `/health` | Liveness check. Deliberately unversioned -- this is the path a hosting platform's health check/orchestrator polls, and shouldn't move if the resource API below ever needs a `/api/v2`. |
| GET | `/api/v1/items` | Every priceable `(product, store)` decision unit. |
| POST | `/api/v1/recommendations` | Score requested `item_id`s (e.g. `P001_S001`); body `{"item_ids": [...]}`. Unknown ids come back under `not_found` instead of failing the request. |
| GET | `/api/v1/products` | Live version of the frontend's bundled demo catalog -- one row per product (input data: price, cost, elasticity...). |
| GET | `/api/v1/products/recommendations` | One pricing decision per product, aggregated across its stores. |
| POST | `/api/v1/refresh` | Reload artifacts from disk, e.g. after a new pipeline run. |

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
| `PRICING_API_CORS_ALLOW_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173,https://bellarose77.github.io` | Comma-separated allowlist of frontend origins. Covers local dev and the deployed GitHub Pages frontend out of the box; override to add another origin (never set to `*` in production -- this API serves a live data volume, not static assets). |
| `PRICING_API_PROCESSED_DIR` | `<repo>/data/processed` | Where to read pipeline artifacts from. |

See `.env.example` for a copy-pasteable local override file.

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
frontend's own `.env.example`). CORS must allow the frontend's origin
(see Configuration above); the defaults already cover both the Vite dev
server and the deployed GitHub Pages frontend.

## Deployment

GitHub Pages only serves static files, so it cannot run this service --
it needs a separate host that can run a container and keep it listening.
The image already builds and runs with no external database or required
volume (see Docker above: the `data/` snapshot baked in at build time is
enough to serve demo recommendations), so any container host works;
recommended, roughly in order:

- **Render** (Web Service, "Docker" runtime) -- simplest option: point it
  at this repo, set the Dockerfile path to
  `service-pricing-optimization/Dockerfile` and the build context to the
  repo root, and it builds, assigns `$PORT`, and terminates TLS for you.
  Free tier is enough for a demo (cold starts after idling).
- **Fly.io** -- similar simplicity via `fly launch` against the same
  Dockerfile, if you'd rather not have cold starts on a free tier.
- **Google Cloud Run** -- best fit if you're already on GCP; also
  builds straight from the Dockerfile and scales to zero.

All three: build from `service-pricing-optimization/Dockerfile` with the
**repository root** as build context (same as the local `docker build`
command above), inject `$PORT` (the image already respects it, see
Dockerfile), and terminate HTTPS at the platform's edge -- none of them
require you to configure TLS in the app itself.

### Required configuration on the chosen host

| Setting | Value |
|---|---|
| Dockerfile path | `service-pricing-optimization/Dockerfile` |
| Build context | repository root (`.`) |
| Container port | `$PORT` (platform-injected; defaults to `8000` if unset) |
| Startup command | already set by the Dockerfile's `CMD`; leave blank unless the platform requires one explicitly (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`) |
| Health check path | `/health` |
| `PRICING_API_CORS_ALLOW_ORIGINS` | defaults already include `https://bellarose77.github.io`; only set this if deploying under a different frontend origin |
| Python version | pinned by the Dockerfile's `python:3.11-slim` base image -- no separate host setting needed |
| Dependency installation | handled by the Dockerfile (`pip install -r requirements.txt`) -- no separate host setting needed |

### Once it's deployed

1. Note the HTTPS URL the host assigns (e.g. `https://elastiq-pricing-api.onrender.com`).
2. Set the GitHub repository variable `PRICING_API_BASE_URL` to that URL
   (Settings -> Secrets and variables -> Actions -> Variables) -- the
   `deploy-pages.yml` workflow already reads it into
   `VITE_PRICING_API_BASE_URL` at build time, no workflow edit needed.
3. Re-run the Pages workflow (or push to `main`) so the frontend rebuilds
   against the live API URL.
4. If the API's own host origin differs from `https://bellarose77.github.io`
   (a custom domain, a fork), also set `PRICING_API_CORS_ALLOW_ORIGINS` on
   the API host to match.
