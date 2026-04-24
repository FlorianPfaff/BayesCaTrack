"""Shared helpers for test bootstrap and subprocess environments."""

import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

MODULE_RUN_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC_PATH)
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
}


def run_module(*args: str):
    """Run a module with the test source tree exposed on ``PYTHONPATH``."""

    return subprocess.run(  # nosec B603
        [sys.executable, *args],
        check=True,
        env=MODULE_RUN_ENV,
        capture_output=True,
        text=True,
    )


def assert_module_reexports(module: ModuleType, source_module: ModuleType) -> None:
    """Assert that every public name of ``module`` comes from ``source_module``."""

    for name in module.__all__:
        assert getattr(module, name) is getattr(source_module, name)
