"""Load and cache the market-intelligence feature artifact this service serves.

Step 07 of the batch pipeline (src/pipelines/step_07_rag_features.py)
retrieves relevant market notes per item and converts them into
structured features -- see src/features/text.py -- merging the result
into retail_with_rag_features.csv. That retrieval/extraction step stays
an offline batch job. This module loads that file's *output* once,
caches just the feature-generation columns (not the full 78-column
merged dataset, which also carries pricing/forecast columns owned by
other steps), and refreshes on demand.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import settings

FEATURES_FILE = "retail_with_rag_features.csv"

IDENTITY_COLUMNS = ["product_id", "store_id", "category"]

FEATURE_COLUMNS = [
    "rag_evidence_count",
    "rag_max_similarity",
    "rag_mean_similarity",
    "rag_min_similarity",
    "rag_weighted_impact_score",
    "rag_combined_evidence",
    "demand_growth_signal_count",
    "demand_decline_signal_count",
    "inflation_signal_count",
    "supply_risk_signal_count",
    "promotion_signal_count",
    "competition_signal_count",
    "seasonality_signal_count",
    "premium_signal_count",
    "value_signal_count",
    "net_demand_signal",
    "evidence_character_count",
    "evidence_word_count",
]


@dataclass
class ArtifactStore:
    """The in-memory per-item feature table plus when it was built."""

    features: pd.DataFrame
    generated_from: str


class MissingArtifactsError(RuntimeError):
    """Raised when the feature artifact is not on disk."""


_lock = threading.Lock()
_store: ArtifactStore | None = None


def _features_path() -> Path:
    return settings.processed_dir / FEATURES_FILE


def _load() -> ArtifactStore:
    path = _features_path()

    if not path.exists():
        raise MissingArtifactsError(
            "Missing feature artifact, run the batch pipeline first "
            f"(python scripts/run_pipeline.py): {path}"
        )

    required_columns = IDENTITY_COLUMNS + FEATURE_COLUMNS
    raw = pd.read_csv(path)
    missing_columns = [c for c in required_columns if c not in raw.columns]

    if missing_columns:
        raise MissingArtifactsError(
            f"{path} is missing expected column(s): {missing_columns}. "
            "Re-run step 07 (python -m src.pipelines.step_07_rag_features)."
        )

    features = raw[required_columns].copy()
    features["item_id"] = (
        features["product_id"].astype(str)
        + "_"
        + features["store_id"].astype(str)
    )

    return ArtifactStore(
        features=features,
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
    """Reload the feature artifact from disk."""

    global _store

    with _lock:
        _store = _load()

        return _store
