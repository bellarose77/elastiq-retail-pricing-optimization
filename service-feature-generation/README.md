# ELASTIQ Feature Generation API

A standalone FastAPI microservice that serves structured
market-intelligence features -- retrieved evidence, signal counts, and a
weighted demand-impact score -- computed by the ELASTIQ pipeline's
feature-generation step. It does **not** run retrieval or feature
extraction itself -- see [Architecture](#architecture).

## Quick start

From this directory:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

Open `http://127.0.0.1:8002/docs` for interactive API docs.

The service reads `../data/processed/retail_with_rag_features.csv` by
default. That file is produced by:

```bash
cd ..
python -m pip install -r requirements.txt
python -m src.pipelines.step_07_rag_features
```

The repository ships with pre-computed demo features, so this is only
needed for fresh numbers.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check. |
| GET | `/items` | Every `(product, store)` item's generated features. |
| POST | `/features` | Look up requested `item_id`s (e.g. `P001_S001`); body `{"item_ids": [...]}`. Unknown ids come back under `not_found`. |
| GET | `/products/features` | One feature summary per product, aggregated across its stores. |
| POST | `/refresh` | Reload the artifact from disk, e.g. after a new pipeline run. |

Each item's features include retrieval diagnostics
(`rag_evidence_count`, `rag_max/mean/min_similarity`), the retrieved
evidence text itself (`rag_combined_evidence`), a bounded
`rag_weighted_impact_score` (the signal `service-pricing-optimization`
uses as its market-demand adjustment), and per-theme signal counts
(demand growth/decline, inflation, supply risk, promotion, competition,
seasonality, premium, value) rolling up into `net_demand_signal`.

## Architecture

```
app/
  main.py              FastAPI app: creates the app, configures CORS, mounts routes.
  config.py            Environment-driven settings.
  schemas.py           Pydantic request/response models.
  api/
    routes.py          HTTP handlers. Thin: validate via schemas.py, delegate to
                        services/feature_service.py.
  services/
    artifact_store.py  Loads and caches the feature columns of
                        retail_with_rag_features.csv.
    feature_service.py Business layer: slices/aggregates the cached feature
                        table. No retrieval/extraction logic lives here.
```

This service is intentionally simpler than `service-pricing-optimization`:
the actual retrieval-augmented feature extraction
(`src/features/text.py`, chunking and embedding market notes, retrieving
relevant evidence per item, and scoring it into structured features)
only runs inside the offline batch pipeline
(`src/pipelines/step_07_rag_features.py`). This service reuses none of
that code -- it loads the batch step's already-computed *output* (the
feature-generation columns of `retail_with_rag_features.csv`; the
pricing/forecast columns in that same file belong to other pipeline
steps and aren't served here) and serves it, so it has no retrieval/NLP
dependency and no algorithm to keep in sync. If a future need arises for
on-demand feature extraction from new market notes rather than serving a
batch artifact, that would call for reusing `src/features/text.py` from
this service's business layer the same way `service-pricing-optimization`
reuses `src/optimization` -- a larger change than what's implemented
today.

## Configuration

Environment variables, prefixed `FEATURE_API_`:

| Variable | Default | Purpose |
|---|---|---|
| `FEATURE_API_CORS_ALLOW_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Comma-separated list of allowed frontend origins. |
| `FEATURE_API_PROCESSED_DIR` | `<repo>/data/processed` | Where to read the feature artifact from. |

## Tests

```bash
pip install -r requirements.txt
pytest
```

Tests run against the real demo artifact checked into the repository
rather than mocks.

## Docker

Build context must be the **repository root**:

```bash
cd ..
docker build -f service-feature-generation/Dockerfile -t elastiq-feature-api .
docker run --rm -p 8002:8002 \
  -v "$(pwd)/data/processed:/srv/data/processed:ro" \
  elastiq-feature-api
```

## Relationship to the other services

`service-pricing-optimization` currently reads
`data/processed/*.csv` directly (including the same feature columns this
service serves) rather than calling this service over HTTP -- see its
README. Wiring pricing to call this service instead of reading the
shared filesystem artifact is a deliberate follow-up, not done here.
