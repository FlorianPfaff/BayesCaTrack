from __future__ import annotations

from pathlib import Path

from bayescatrack.dependency_pins import PYRECEST_COMMIT, PYRECEST_DIRECT_URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_pins_pyrecest_to_exact_commit() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert PYRECEST_DIRECT_URL in pyproject
    assert "PyRecEst.git@main" not in pyproject


def test_ci_does_not_override_pyrecest_pin() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "bayescatrack-ci.yml").read_text(
        encoding="utf-8"
    )

    assert "--force-reinstall --no-deps" not in workflow
    assert "PyRecEst.git@main" not in workflow
    assert "2154731b9954ba004cac0b48be4539d5bcdcb468" not in workflow


def test_pyrecest_pin_is_documented_for_benchmarks() -> None:
    benchmark = (
        PROJECT_ROOT / ".github" / "scripts" / "run_track2p_benchmark.py"
    ).read_text(encoding="utf-8")

    assert "PYRECEST_COMMIT" in benchmark
    assert "pyrecest_commit" in benchmark
    assert PYRECEST_COMMIT in PYRECEST_DIRECT_URL
