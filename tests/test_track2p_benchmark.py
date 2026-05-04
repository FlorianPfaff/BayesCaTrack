from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    format_benchmark_table,
    run_track2p_benchmark,
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


def _write_ground_truth_csv(
    subject_dir: Path, session_names: tuple[str, ...], rows: tuple[tuple[int, ...], ...]
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def _write_suite2p_session(
    subject_dir: Path, session_name: str, *, iscell: np.ndarray
) -> Path:
    plane_dir = subject_dir / session_name / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = np.array(
        [
            {
                "ypix": np.array([0, 0]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
            {
                "ypix": np.array([1, 1]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
            {
                "ypix": np.array([2, 2]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 4, "Lx": 4, "meanImg": np.zeros((4, 4), dtype=float)},
        allow_pickle=True,
    )
    np.save(plane_dir / "F.npy", np.arange(6, dtype=float).reshape(3, 2))
    return plane_dir


def _install_fake_multisession_assignment(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_assignment = types.ModuleType("pyrecest.utils.multisession_assignment")

    class Result:
        def __init__(self):
            self.tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 2: 1}]
            self.matched_edges = []
            self.total_cost = 0.0

    def solve_multisession_assignment(pairwise_costs, **kwargs):
        assert (0, 1) in pairwise_costs
        assert (0, 2) in pairwise_costs
        assert kwargs["session_sizes"] == (2, 2, 2)
        assert kwargs["gap_penalty"] == pytest.approx(1.0)
        return Result()

    fake_assignment.solve_multisession_assignment = solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "pyrecest.utils.multisession_assignment", fake_assignment
    )


def test_track2p_baseline_benchmark_scores_track2p_output_only_as_smoke_test(
    tmp_path, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session)

    with pytest.raises(ValueError, match="not independent manual ground truth"):
        run_track2p_benchmark(
            Track2pBenchmarkConfig(data=tmp_path, method="track2p-baseline")
        )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="track2p-baseline",
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Track2p default"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
    assert result["complete_tracks"] == 2
    assert "Track2p default" in format_benchmark_table([result])


def test_track2p_baseline_benchmark_scores_aligned_rows_without_track2p_output(
    tmp_path, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="track2p-baseline",
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    result = rows[0].to_dict()
    assert result["variant"] == "Track2p default"
    assert result["reference_source"] == "aligned_subject_rows"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)


def test_benchmark_uses_ground_truth_csv_reference(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        ((0, 0, 0), (1, 1, 1)),
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(data=tmp_path, method="track2p-baseline")
    )

    result = rows[0].to_dict()
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)


def test_ground_truth_csv_validation_catches_filtered_stat_rows(tmp_path):
    subject_dir = tmp_path / "jm003"
    iscell = np.array([[1.0, 0.95], [0.0, 0.1], [1.0, 0.9]], dtype=float)
    _write_suite2p_session(subject_dir, "2024-05-01_a", iscell=iscell)
    _write_suite2p_session(subject_dir, "2024-05-02_a", iscell=iscell)
    _write_ground_truth_csv(
        subject_dir, ("2024-05-01_a", "2024-05-02_a"), ((0, 0), (1, 1))
    )

    config = Track2pBenchmarkConfig(
        data=subject_dir, method="track2p-baseline", input_format="suite2p"
    )
    with pytest.raises(ValueError, match="--include-non-cells"):
        run_track2p_benchmark(config)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="track2p-baseline",
            input_format="suite2p",
            include_non_cells=True,
        )
    )

    result = rows[0].to_dict()
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_recall"] == pytest.approx(1.0)
    assert result["pairwise_precision"] == pytest.approx(1.0)
    assert result["dropped_prediction_tracks"] == 1


def test_ground_truth_scoring_filters_predictions_to_reference_seed_rois(tmp_path):
    subject_dir = tmp_path / "jm004"
    iscell = np.ones((3, 2), dtype=float)
    _write_suite2p_session(subject_dir, "2024-05-01_a", iscell=iscell)
    _write_suite2p_session(subject_dir, "2024-05-02_a", iscell=iscell)
    _write_ground_truth_csv(
        subject_dir, ("2024-05-01_a", "2024-05-02_a"), ((0, 0), (1, 1))
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="track2p-baseline",
            input_format="suite2p",
            include_non_cells=True,
        )
    )

    result = rows[0].to_dict()
    assert result["reference_seed_rois"] == 2
    assert result["evaluated_prediction_tracks"] == 2
    assert result["dropped_prediction_tracks"] == 1
    assert result["pairwise_precision"] == pytest.approx(1.0)


def test_global_assignment_benchmark_uses_skip_edges(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm002"
    _write_subject(subject_dir, write_raw_npy_session)
    _install_fake_multisession_assignment(monkeypatch)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(
        global_assignment,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="global-assignment",
            cost="registered-iou",
            max_gap=2,
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Same costs + global assignment"
    assert result["pairwise_f1"] == pytest.approx(2 / 3)
    assert result["complete_track_f1"] == pytest.approx(2 / 3)
    assert result["complete_tracks"] == 1
