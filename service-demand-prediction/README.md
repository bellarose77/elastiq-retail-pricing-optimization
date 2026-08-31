# ELASTIQ Demand Prediction API

A standalone FastAPI microservice that serves next-period demand
forecasts computed by the ELASTIQ pipeline's demand-forecasting step. It
does **not** train or run the forecasting model itself -- see
[Architecture](#architecture).

## Quick start

From this directory:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Open `http://127.0.0.1:8001/docs` for interactive API docs.

The service reads `../data/processed/xgboost_next_period_forecast.csv`
by default. That file is produced by:

```bash
cd ..
python -m pip install -r requirements.txt
python -m src.pipelines.step_06_demand_forecasting
```

The repository ships with a pre-computed demo forecast, so this is only
needed for fresh numbers.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check. |
| GET | `/items` | Every `(product, store)` item's next-period forecast. |
| POST | `/forecasts` | Look up requested `item_id`s (e.g. `P001_S001`); body `{"item_ids": [...]}`. Unknown ids come back under `not_found`. |
| GET | `/products/forecasts` | One forecast per product, aggregated across its stores. |
| POST | `/refresh` | Reload the artifact from disk, e.g. after a new pipeline run. |

## Architecture

```
app/
  main.py              FastAPI app: creates the app, configures CORS, mounts routes.
  config.py            Environment-driven settings.
  schemas.py           Pydantic request/response models.
  api/
    routes.py          HTTP handlers. Thin: validate via schemas.py, delegate to
                        services/forecast_service.py.
  services/
    artifact_store.py  Loads and caches xgboost_next_period_forecast.csv.
    forecast_service.py Business layer: slices/aggregates the cached forecast
                        table. No model logic lives here.
```

This service is intentionally simpler than `service-pricing-optimization`:
the actual forecasting model (`src/models/forecasting.py`, an XGBoost
regressor trained on lag/rolling/calendar features) only runs inside the
offline batch pipeline
(`src/pipelines/step_06_demand_forecasting.py`). This service reuses
none of that code -- it loads the batch step's already-computed *output*
and serves it, so it has no ML dependency (no xgboost, scikit-learn, or
statsmodels) and no algorithm to keep in sync. If a future need arises
for on-demand (re-)forecasting rather than serving a batch artifact, that
would call for reusing `src/models/forecasting.py` from this service's
business layer the same way `service-pricing-optimization` reuses
`src/optimization` -- a larger change than what's implemented today.

## Configuration

Environment variables, prefixed `DEMAND_API_`:

| Variable | Default | Purpose |
|---|---|---|
| `DEMAND_API_CORS_ALLOW_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Comma-separated list of allowed frontend origins. |
| `DEMAND_API_PROCESSED_DIR` | `<repo>/data/processed` | Where to read the forecast artifact from. |

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
docker build -f service-demand-prediction/Dockerfile -t elastiq-demand-api .
docker run --rm -p 8001:8001 \
  -v "$(pwd)/data/processed:/srv/data/processed:ro" \
  elastiq-demand-api
```

## Relationship to the other services

`service-pricing-optimization` currently reads
`data/processed/*.csv` directly rather than calling this service over
HTTP -- see its README. Wiring pricing to call this service instead of
reading the shared filesystem artifact is a deliberate follow-up, not
done here.
