"""Load and cache the demand forecast artifact this service serves.

Step 06 of the batch pipeline (src/pipelines/step_06_demand_forecasting.py)
trains an XGBoost model and writes next-period forecasts to
xgboost_next_period_forecast.csv -- that training/inference step stays an
offline batch job. This module loads its *output* once and caches it;
call refresh() after a new pipeline run lands to pick up fresh numbers
without restarting the process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import settings

FORECAST_FILE = "xgboost_next_period_forecast.csv"

REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "store_id",
    "category",
    "forecast_quantity",
    "predicted_value",
    "prediction_interval_lower",
    "prediction_interval_upper",
]


@dataclass
class ArtifactStore:
    """The in-memory forecast table plus when it was built."""

    forecasts: pd.DataFrame
    generated_from: str


class MissingArtifactsError(RuntimeError):
    """Raised when the forecast artifact is not on disk."""


_lock = threading.Lock()
_store: ArtifactStore | None = None


def _forecast_path() -> Path:
    return settings.processed_dir / FORECAST_FILE


def _load() -> ArtifactStore:
    path = _forecast_path()

    if not path.exists():
        raise MissingArtifactsError(
            "Missing forecast artifact, run the batch pipeline first "
            f"(python scripts/run_pipeline.py): {path}"
        )

    forecasts = pd.read_csv(path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in forecasts.columns]

    if missing_columns:
        raise MissingArtifactsError(
            f"{path} is missing expected column(s): {missing_columns}. "
            "Re-run step 06 (python -m src.pipelines.step_06_demand_forecasting)."
        )

    forecasts = forecasts[REQUIRED_COLUMNS].copy()
    forecasts["item_id"] = (
        forecasts["product_id"].astype(str)
        + "_"
        + forecasts["store_id"].astype(str)
    )

    return ArtifactStore(
        forecasts=forecasts,
        generated_from=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    )


def get_store() -> ArtifactStore:
    """Return the cached artifact store, loading it on first use."""

    global _store

    with _lock:
        if _store is None:
            _store = _load()

        return _store


def refresh() -> ArtifactStore:
    """Reload the forecast artifact from disk."""

    global _store

    with _lock:
        _store = _load()

        return _store
