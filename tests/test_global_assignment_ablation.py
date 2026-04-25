from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bayescatrack.association.pyrecest_global_assignment import session_edge_pairs, tracks_to_suite2p_index_matrix
from bayescatrack.experiments.track2p_global_assignment_ablation import run_track2p_global_assignment_ablation


def _write_subject(subject_dir, write_raw_npy_session):
    masks_a = np.zeros((2, 4, 4), dtype=bool)
    masks_a[0, 0:2, 0:2] = True
    masks_a[1, 2:4, 2:4] = True
    masks_b = masks_a.copy()
    masks_c = masks_a.copy()

    write_raw_npy_session(subject_dir, "2024-05-01_a", masks_a, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks_b, offset=10.0)
    write_raw_npy_session(subject_dir, "2024-05-03_a", masks_c, offset=20.0)

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
    np.save(track2p_dir / "plane0_suite2p_indices.npy", np.array([[0, 0, 0], [1, 1, 1]], dtype=object), allow_pickle=True)


def _install_fake_multisession_assignment(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_assignment = types.ModuleType("pyrecest.utils.multisession_assignment")

    class Result:
        def __init__(self):
            self.tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 2: 1}]
            self.matched_edges = [((0, 1), (2, 1), 1.0)]
            self.total_cost = 0.0

    def solve_multisession_assignment(pairwise_costs, **kwargs):
        assert set(pairwise_costs) == {(0, 1), (0, 2), (1, 2)}
        assert pairwise_costs[(0, 2)].shape == (2, 2)
        assert kwargs["session_sizes"] == (2, 2, 2)
        assert kwargs["gap_penalty"] == pytest.approx(1.0)
        return Result()

    fake_assignment.solve_multisession_assignment = solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.multisession_assignment", fake_assignment)


def test_session_edge_pairs_controls_skip_edges():
    assert session_edge_pairs(4, max_gap=1) == ((0, 1), (1, 2), (2, 3))
    assert session_edge_pairs(4, max_gap=2) == ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
    assert session_edge_pairs(1, max_gap=2) == ()

    with pytest.raises(ValueError, match="max_gap"):
        session_edge_pairs(3, max_gap=0)
    with pytest.raises(ValueError, match="num_sessions"):
        session_edge_pairs(-1, max_gap=1)


def test_global_assignment_ablation_builds_skip_edges_and_scores(tmp_path, monkeypatch, write_raw_npy_session):
    subject_dir = tmp_path / "jm002"
    _write_subject(subject_dir, write_raw_npy_session)
    _install_fake_multisession_assignment(monkeypatch)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(global_assignment, "register_plane_pair", lambda _reference, moving, **_kwargs: moving)

    result = run_track2p_global_assignment_ablation(
        subject_dir,
        cost="registered-iou",
        max_gap=2,
    )

    assert result.variant == "Same costs + global assignment"
    assert result.assignment.session_edges == ((0, 1), (0, 2), (1, 2))
    assert result.scores["pairwise_f1"] == pytest.approx(2 / 3)
    assert result.scores["complete_track_f1"] == pytest.approx(2 / 3)
    assert result.scores["complete_tracks"] == 1
    np.testing.assert_array_equal(
        result.predicted_track_matrix,
        np.array([[0, 0, 0], [1, None, 1]], dtype=object),
    )


def test_tracks_to_suite2p_index_matrix_uses_original_roi_indices():
    class Plane:
        n_rois = 2
        roi_indices = np.array([5, 7])

    class Session:
        plane_data = Plane()

    tracks = [{0: 0, 1: 1}]
    matrix = tracks_to_suite2p_index_matrix(tracks, [Session(), Session()])

    np.testing.assert_array_equal(matrix, np.array([[5, 7]], dtype=object))
