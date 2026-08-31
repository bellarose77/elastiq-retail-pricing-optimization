"""Service configuration.

This service lives in a monorepo alongside the ``src/`` package that owns
the actual pricing algorithm (see ``src/optimization``). It reuses that
package by import rather than duplicating it (see ``app/services/`` for
the boundary), which means the repository root -- the directory
containing both ``src/`` and this service -- must be importable. That is
true when this service is run from within its own directory (as the
README instructs: ``cd service-pricing-optimization && uvicorn app.main:app``)
and true inside the Docker image (see ``Dockerfile``), but not guaranteed
by default, so it's made true here explicitly rather than relying on the
caller to set PYTHONPATH correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# service-pricing-optimization/app/config.py -> parents[0]=app,
# [1]=service-pricing-optimization, [2]=repository root (contains src/).
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import PROCESSED_DIR as _DEFAULT_PROCESSED_DIR


class Settings(BaseSettings):
    """Environment-driven configuration, prefixed ``PRICING_API_``.

    Example: ``PRICING_API_CORS_ALLOW_ORIGINS=https://app.example.com``.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRICING_API_",
        extra="ignore",
    )

    title: str = "ELASTIQ Pricing API"
    version: str = "1.0.0"
    description: str = (
        "Online price recommendations, scored from the batch pipeline's "
        "latest elasticity, causal, uplift, and forecast artifacts."
    )

    # Comma-separated. The frontend is a different origin (the Vite dev
    # server on :5173, or the deployed GitHub Pages site), so it needs
    # CORS to call this service directly. Defaults cover both out of the
    # box; override for a different dev port or a different deployed
    # frontend origin. Deliberately a fixed allowlist, never "*" -- this
    # API is a real backend with a live processed-data volume, not a
    # static site with nothing to protect.
    cors_allow_origins: str = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "https://bellarose77.github.io"
    )

    # Overridable so the service can point at a data/processed directory
    # that isn't a sibling of this checkout (e.g. a mounted volume in a
    # container). Defaults to the monorepo's own data/processed/.
    processed_dir: Path = _DEFAULT_PROCESSED_DIR

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
