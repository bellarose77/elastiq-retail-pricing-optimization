"""Service configuration.

Unlike service-pricing-optimization, this service doesn't import any
algorithm from src/ -- it only serves a pipeline artifact that's already
fully computed (see app/services/artifact_store.py), so there's no
algorithm-reuse boundary to maintain here. The sys.path bootstrap and
src.config.PROCESSED_DIR reuse below exist only so the default data path
stays a single source of truth with the rest of the monorepo, not because
this service needs anything else from src/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# service-demand-prediction/app/config.py -> parents[0]=app,
# [1]=service-demand-prediction, [2]=repository root (contains src/).
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import PROCESSED_DIR as _DEFAULT_PROCESSED_DIR


class Settings(BaseSettings):
    """Environment-driven configuration, prefixed ``DEMAND_API_``."""

    model_config = SettingsConfigDict(
        env_prefix="DEMAND_API_",
        extra="ignore",
    )

    title: str = "ELASTIQ Demand Prediction API"
    version: str = "1.0.0"
    description: str = (
        "Next-period demand forecasts, served from the batch pipeline's "
        "most recent XGBoost forecast artifact."
    )

    cors_allow_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    processed_dir: Path = _DEFAULT_PROCESSED_DIR

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
