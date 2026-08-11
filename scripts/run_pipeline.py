"""Run the complete analytical pipeline in its required order."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "step_01_data_validation",
    "step_02_exploratory_analysis",
    "step_03_price_elasticity",
    "step_04_iv_causal_model",
    "step_05_promotion_uplift",
    "step_06_demand_forecasting",
    "step_07_rag_features",
    "step_08_price_optimization",
]


def run(*args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    print("\nRefreshing ELASTIQ analytical outputs...")
    for index, step in enumerate(STEPS, start=1):
        print(f"\n[{index}/8] {step}")
        run("-m", f"src.pipelines.{step}")
    print("\nUpdating browser data and parity fixtures...")
    run("scripts/generate_parity_fixture.py")
    run("scripts/export_demo_data.py")
    print("\nELASTIQ data refresh completed successfully.")


if __name__ == "__main__":
    main()

