from __future__ import annotations

import csv

import numpy as np
import pytest
from bayescatrack.experiments import track2p_cost_sweep as sweep_module
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
)
from bayescatrack.experiments.track2p_cost_sweep import (
    CostSweepConfig,
    _parse_cost_scales,
    _parse_nonnegative_values,
    _parse_positive_values,
    _parse_thresholds,
    format_sweep_table,
    run_track2p_cost_sweep,
    write_sweep_results_incrementally,
)


def _write_subject(subject_dir, write_raw_npy_session, *, write_reference=True):
    masks_a = np.zeros((2, 4, 4), dtype=bool)
    masks_a[0, 0:2, 0:2] = True
    masks_a[1, 2:4, 2:4] = True
    masks_b = masks_a.copy()
    masks_c = masks_a.copy()

    write_raw_npy_session(subject_dir, "2024-05-01_a", masks_a, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks_b, offset=10.0)
    write_raw_npy_session(subject_dir, "2024-05-03_a", masks_c, offset=20.0)

    if not write_reference:
        return

    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir()
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.array(
                [
                    str(subject_dir / "2024-05-01_a"),
                    str(subject_dir / "2024-05-02_a"),
                    str(subject_dir / "2024-05-03_a"),
                ],
                dtype=object,
            ),
            "vector_curation_plane_0": np.array([1.0, 1.0]),
        },
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        np.array([[0, 0, 0], [1, 1, 1]], dtype=object),
        allow_pickle=True,
    )


def test_track2p_cost_sweep_reuses_costs_and_varies_solver_knobs(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session)

    build_calls = []
    solver_calls = []

    def fake_build_registered_pairwise_costs(sessions, **kwargs):
        build_calls.append((len(sessions), kwargs))
        return {(0, 1): np.asarray([[1.0, 2.0], [10.0, np.inf]], dtype=float)}

    class SolverResult:
        tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 1: 1, 2: 1}]
        matched_edges = []
        total_cost = 0.0

    def fake_solver(pairwise_costs, **kwargs):
        solver_calls.append((pairwise_costs, kwargs))
        return SolverResult()

    monkeypatch.setattr(
        sweep_module,
        "build_registered_pairwise_costs",
        fake_build_registered_pairwise_costs,
    )
    monkeypatch.setattr(
        sweep_module,
        "_load_pyrecest_multisession_solver",
        lambda: fake_solver,
    )

    rows = run_track2p_cost_sweep(
        CostSweepConfig(
            benchmark=Track2pBenchmarkConfig(
                data=subject_dir,
                method="global-assignment",
                cost="registered-iou",
                allow_track2p_as_reference_for_smoke_test=True,
                progress=False,
            ),
            cost_scales=(0.5, 2.0),
            cost_thresholds=(None, 6.0),
        )
    )

    assert len(build_calls) == 1
    assert len(solver_calls) == 4
    assert [call[1]["cost_threshold"] for call in solver_calls] == [
        None,
        6.0,
        None,
        6.0,
    ]
    assert [call[1]["start_cost"] for call in solver_calls] == [5.0] * 4
    assert [call[1]["end_cost"] for call in solver_calls] == [5.0] * 4
    assert [call[1]["gap_penalty"] for call in solver_calls] == [1.0] * 4
    np.testing.assert_allclose(solver_calls[0][0][(0, 1)], [[0.5, 1.0], [5.0, np.inf]])
    np.testing.assert_allclose(solver_calls[3][0][(0, 1)], [[2.0, 4.0], [20.0, np.inf]])

    result_dicts = [row.to_dict() for row in rows]
    assert [row["sweep_index"] for row in result_dicts] == [1, 2, 3, 4]
    assert result_dicts[0]["cost_threshold"] == "none"
    assert result_dicts[0]["start_cost"] == pytest.approx(5.0)
    assert result_dicts[0]["end_cost"] == pytest.approx(5.0)
    assert result_dicts[0]["gap_penalty"] == pytest.approx(1.0)
    assert result_dicts[0]["cost_median"] == pytest.approx(1.0)
    assert result_dicts[1]["cost_threshold_admitted_fraction"] == pytest.approx(1.0)
    assert result_dicts[3]["cost_threshold_admitted_fraction"] == pytest.approx(2 / 3)

    table = format_sweep_table(result_dicts)
    assert "cost_scale" in table
    assert "start_cost" in table
    assert "cost_threshold_admitted_fraction" in table


def test_track2p_cost_sweep_varies_start_end_and_gap(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session)

    solver_calls = []

    class SolverResult:
        tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 1: 1, 2: 1}]
        matched_edges = []
        total_cost = 0.0

    def fake_solver(_pairwise_costs, **kwargs):
        solver_calls.append(kwargs)
        return SolverResult()

    monkeypatch.setattr(
        sweep_module,
        "build_registered_pairwise_costs",
        lambda sessions, **kwargs: {(0, 1): np.asarray([[1.0]], dtype=float)},
    )
    monkeypatch.setattr(
        sweep_module,
        "_load_pyrecest_multisession_solver",
        lambda: fake_solver,
    )

    rows = run_track2p_cost_sweep(
        CostSweepConfig(
            benchmark=Track2pBenchmarkConfig(
                data=subject_dir,
                method="global-assignment",
                cost="registered-iou",
                allow_track2p_as_reference_for_smoke_test=True,
                progress=False,
            ),
            cost_scales=(1.0,),
            cost_thresholds=(6.0,),
            start_costs=(0.5, 2.0),
            end_costs=(0.5,),
            gap_penalties=(0.0, 1.0),
        )
    )

    assert len(solver_calls) == 4
    assert [
        (call["start_cost"], call["end_cost"], call["gap_penalty"])
        for call in solver_calls
    ] == [(0.5, 0.5, 0.0), (0.5, 0.5, 1.0), (2.0, 0.5, 0.0), (2.0, 0.5, 1.0)]
    result_dicts = [row.to_dict() for row in rows]
    assert [row["sweep_count"] for row in result_dicts] == [4, 4, 4, 4]
    assert [row["start_cost"] for row in result_dicts] == [0.5, 0.5, 2.0, 2.0]


def test_cost_sweep_parser_accepts_none_thresholds():
    assert _parse_cost_scales("0.25,1,4") == (0.25, 1.0, 4.0)
    assert _parse_thresholds("none,2,6") == (None, 2.0, 6.0)
    assert _parse_positive_values("0.5,2", name="--start-costs") == (0.5, 2.0)
    assert _parse_nonnegative_values("0,1", name="--gap-penalties") == (0.0, 1.0)


def test_write_sweep_results_incrementally_writes_csv_rows(tmp_path):
    rows = [
        SubjectBenchmarkResult(
            subject="jm001",
            variant="registered_iou",
            method="global-assignment",
            scores={
                "sweep_index": index,
                "sweep_count": 2,
                "cost_scale": 1.0,
                "cost_threshold": 4.0,
                "start_cost": 0.25,
                "end_cost": 5.0,
                "gap_penalty": 1.0,
                "pairwise_f1": pairwise_f1,
                "complete_track_f1": 0.0,
            },
            n_sessions=3,
            reference_source="manual_gt",
        )
        for index, pairwise_f1 in [(1, 0.25), (2, 0.5)]
    ]
    output = tmp_path / "sweep.csv"

    rows_written = write_sweep_results_incrementally(rows, output)

    assert rows_written == 2
    parsed_rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert [row["sweep_index"] for row in parsed_rows] == ["1", "2"]
    assert [row["pairwise_f1"] for row in parsed_rows] == ["0.25", "0.5"]


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "1,"])
def test_cost_scale_parser_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        _parse_cost_scales(raw)


def test_cost_sweep_rejects_non_global_assignment():
    with pytest.raises(ValueError, match="global-assignment"):
        run_track2p_cost_sweep(
            CostSweepConfig(
                benchmark=Track2pBenchmarkConfig(
                    data="missing", method="track2p-baseline", progress=False
                ),
                cost_scales=(1.0,),
                cost_thresholds=(None,),
            )
        )
