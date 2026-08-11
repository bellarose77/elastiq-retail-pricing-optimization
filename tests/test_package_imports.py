"""
Smoke tests guarding against package-import drift.

These exist because a prior cleanup commit deleted every __init__.py in
src/, which silently broke every pipeline step's imports (package-level
re-exports like `from src.data import load_csv` stopped resolving) without
any test catching it. These tests fail loudly if that happens again, or if
a package's __all__ lists a name that no longer exists in the underlying
module.
"""

import importlib

import pytest


PACKAGES_WITH_ALL = [
    "src.data",
    "src.analysis",
    "src.features",
    "src.models",
    "src.optimization",
    "src.visualization",
    "src.pipelines",
]

PIPELINE_STEP_MODULES = [
    "src.pipelines.step_01_data_validation",
    "src.pipelines.step_02_exploratory_analysis",
    "src.pipelines.step_03_price_elasticity",
    "src.pipelines.step_04_iv_causal_model",
    "src.pipelines.step_05_promotion_uplift",
    "src.pipelines.step_06_demand_forecasting",
    "src.pipelines.step_07_rag_features",
    "src.pipelines.step_08_price_optimization",
]


@pytest.mark.parametrize("package_name", PACKAGES_WITH_ALL)
def test_package_all_exports_are_importable(package_name):
    """Test that every name in a package's __all__ actually resolves."""
    module = importlib.import_module(package_name)
    exported_names = getattr(module, "__all__", None)

    assert exported_names, f"{package_name} has no __all__ to verify"

    for name in exported_names:
        assert hasattr(module, name), (
            f"{package_name}.__all__ lists '{name}' but it is not "
            "actually importable from the package."
        )


@pytest.mark.parametrize("module_name", PIPELINE_STEP_MODULES)
def test_pipeline_step_module_imports_and_has_main(module_name):
    """Test that every pipeline step module imports cleanly and has main()."""
    module = importlib.import_module(module_name)

    assert callable(module.main)
