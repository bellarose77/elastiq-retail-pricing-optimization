"""Service configuration.

Like service-demand-prediction, this service serves an already-computed
pipeline artifact rather than reimplementing any algorithm -- see
app/services/artifact_store.py's docstring. The sys.path bootstrap and
src.config.PROCESSED_DIR reuse below exist only so the default data path
stays a single source of truth with the rest of the monorepo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# service-feature-generation/app/config.py -> parents[0]=app,
# [1]=service-feature-generation, [2]=repository root (contains src/).
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import PROCESSED_DIR as _DEFAULT_PROCESSED_DIR


class Settings(BaseSettings):
    """Environment-driven configuration, prefixed ``FEATURE_API_``."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_API_",
        extra="ignore",
    )

    title: str = "ELASTIQ Feature Generation API"
    version: str = "1.0.0"
    description: str = (
        "Structured market-intelligence features (retrieved evidence, "
        "signal counts, weighted demand impact), served from the batch "
        "pipeline's most recent RAG feature-extraction artifact."
    )

    # See service-pricing-optimization/app/config.py's identically-named
    # field for why this defaults to a fixed allowlist (never "*") and
    # already covers both the Vite dev server and the deployed GitHub
    # Pages frontend.
    cors_allow_origins: str = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "https://bellarose77.github.io"
    )

    processed_dir: Path = _DEFAULT_PROCESSED_DIR

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
