"""Load and cache the upstream pipeline artifacts the pricing service scores against.

Steps 01-07 of the batch pipeline (see ``src/pipelines``) are expensive,
data-hungry, and produce artifacts, not request/response answers -- they
stay an offline job. This module loads their *outputs* once and rebuilds
the same reconciled per-item table step 08 uses
(``src.optimization.build_optimization_dataset``), so this service prices
on identical evidence to the batch pipeline. Call ``refresh()`` after a
new pipeline run lands to pick up fresh artifacts without restarting the
process.

This is the only module below the API layer that touches the filesystem;
``pricing_service.py`` (the business layer) depends on it for data, and
``api/routes.py`` never imports it directly.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import settings
from src.data import load_csv
from src.optimization import build_optimization_dataset


def _required_files() -> dict[str, Path]:
    processed_dir = settings.processed_dir

    return {
        "retail_rag": processed_dir / "retail_with_rag_features.csv",
        "elasticity": processed_dir / "product_elasticity_estimates.csv",
        "promotion_uplift": processed_dir / "promotion_uplift_by_product.csv",
        "iv_product_elasticity": (
            processed_dir / "iv_product_elasticity_estimates.csv"
        ),
        "iv_pooled_elasticity": processed_dir / "iv_pooled_elasticity.json",
    }


@dataclass
class ArtifactStore:
    """The in-memory optimization dataset plus when it was built."""

    optimization_data: pd.DataFrame
    elasticity_estimates: pd.DataFrame
    retail_rag: pd.DataFrame
    generated_from: str


class MissingArtifactsError(RuntimeError):
    """Raised when a required upstream pipeline output is not on disk."""


_lock = threading.Lock()
_store: ArtifactStore | None = None


def _load() -> ArtifactStore:
    required_files = _required_files()
    missing = [
        str(path) for path in required_files.values() if not path.exists()
    ]

    if missing:
        raise MissingArtifactsError(
            "Missing upstream pipeline outputs, run the batch pipeline "
            f"first (python scripts/run_pipeline.py): {missing}"
        )

    retail_rag = load_csv(
        required_files["retail_rag"],
        parse_dates=["date"],
        low_memory=False,
    )
    elasticity_estimates = load_csv(required_files["elasticity"])
    promotion_uplift = load_csv(required_files["promotion_uplift"])
    iv_product_elasticity = load_csv(required_files["iv_product_elasticity"])

    with open(
        required_files["iv_pooled_elasticity"],
        encoding="utf-8",
    ) as handle:
        iv_pooled = json.load(handle)

    optimization_data = build_optimization_dataset(
        retail_rag,
        elasticity_estimates,
        promotion_uplift,
        iv_product_elasticity,
        iv_pooled,
        verbose=False,
    )

    newest_mtime = max(
        path.stat().st_mtime for path in required_files.values()
    )

    return ArtifactStore(
        optimization_data=optimization_data,
        elasticity_estimates=elasticity_estimates,
        retail_rag=retail_rag,
        generated_from=datetime.fromtimestamp(
            newest_mtime, tz=timezone.utc
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
    """Reload artifacts from disk, e.g. after a new pipeline run lands."""

    global _store

    with _lock:
        _store = _load()

        return _store
