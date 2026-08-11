"""
Reusable retail pricing and promotion optimization utilities.

This package provides modular tools for:
- Data ingestion, validation, and quality assessment
- Feature engineering and text processing
- Statistical modeling, forecasting, and causal inference
- Promotion uplift and RAG-based market features
- Constrained price optimization
- End-to-end pipeline orchestration

Directory structure:
- src/data/         - Data I/O, validation, quality, splitting, merging
- src/features/     - Feature engineering, text normalization
- src/models/       - Elasticity, causal, promotion, forecasting, RAG, evaluation
- src/analysis/     - EDA and reporting functions
- src/optimization/ - Pricing optimization
- src/visualization/ - Plotting utilities
- src/pipelines/    - Executable workflow orchestration scripts
"""

__all__ = [
    "analysis",
    "data",
    "features",
    "models",
    "optimization",
    "pipelines",
    "visualization",
]
