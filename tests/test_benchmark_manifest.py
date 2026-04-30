from __future__ import annotations

import csv
import json

import pytest

from bayescatrack.datasets.track2p import (
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from tests._support import run_module


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_benchmark_manifest_runs_suite_and_comparison(tmp_path):
    write_synthetic_track2p_subject(
        tmp_path / "data",
        SyntheticTrack2pSubjectConfig(subject_name="jm_manifest"),
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data/jm_manifest",
                "method": "track2p-baseline",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "track2p-default",
                    "output": "results/track2p.csv",
                },
                {
                    "name": "repeat-default",
                },
            ],
            "comparisons": [
                {
                    "name": "summary",
                    "inputs": {
                        "Track2p": "track2p-default",
                        "Repeat": "repeat-default",
                    },
                    "output": "results/comparison.md",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.name for run in result.runs] == ["track2p-default", "repeat-default"]
    assert (tmp_path / "results" / "track2p.csv").exists()
    assert (tmp_path / "benchmark-results" / "repeat-default.csv").exists()
    assert (tmp_path / "results" / "comparison.md").exists()
    assert (
        _read_csv_rows(tmp_path / "results" / "track2p.csv")[0]["reference_source"]
        == "ground_truth_csv"
    )
    assert "Track2p" in (tmp_path / "results" / "comparison.md").read_text(
        encoding="utf-8"
    )


def test_benchmark_manifest_rejects_unknown_run_keys(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "track2p-baseline",
            },
            "runs": [
                {
                    "name": "bad",
                    "unexpected": True,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_suite_cli_runs_manifest(tmp_path):
    write_synthetic_track2p_subject(
        tmp_path / "data",
        SyntheticTrack2pSubjectConfig(subject_name="jm_cli_manifest"),
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data/jm_cli_manifest",
                "method": "track2p-baseline",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "track2p-default",
                    "output": "results/track2p.csv",
                }
            ],
        },
    )

    proc = run_module(
        "-m",
        "bayescatrack",
        "benchmark",
        "suite",
        str(manifest_path),
        "--summary-format",
        "table",
        "--no-progress",
    )

    assert "track2p-default" in proc.stdout
    assert (tmp_path / "results" / "track2p.csv").exists()
