from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)


def _write_subject(subject_dir, write_raw_npy_session):
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    session_names = ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")

    for session_index, session_name in enumerate(session_names):
        write_raw_npy_session(
            subject_dir,
            session_name,
            masks.copy(),
            offset=float(10 * session_index),
        )

    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir()
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.array(
                [str(subject_dir / session_name) for session_name in session_names],
                dtype=object,
            ),
            "vector_curation_plane_0": np.ones((2,), dtype=float),
        },
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        np.array([[0, 0, 0], [1, 1, 1]], dtype=object),
        allow_pickle=True,
    )


def _write_aligned_subject(subject_dir, write_raw_npy_session):
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True

    for session_index, session_name in enumerate(("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")):
        write_raw_npy_session(
            subject_dir,
            session_name,
            masks.copy(),
            offset=float(10 * session_index),
        )


def _install_fake_pyrecest(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_models = types.ModuleType("pyrecest.utils.association_models")
    fake_assignment = types.ModuleType("pyrecest.utils.multisession_assignment")

    class LogisticPairwiseAssociationModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_args = None

        def fit(self, features, labels, sample_weight=None):
            self.fit_args = (np.asarray(features), np.asarray(labels), sample_weight)
            return self

        def pairwise_cost_matrix(self, features):
            return np.sum(np.asarray(features, dtype=float), axis=-1)

    class Result:
        def __init__(self):
            self.tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 1: 1, 2: 1}]
            self.matched_edges = []
            self.total_cost = 0.0

    def solve_multisession_assignment(pairwise_costs, **kwargs):
        assert set(pairwise_costs) == {(0, 1), (0, 2), (1, 2)}
        assert kwargs["session_sizes"] == (2, 2, 2)
        return Result()

    fake_models.LogisticPairwiseAssociationModel = LogisticPairwiseAssociationModel
    fake_assignment.solve_multisession_assignment = solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.association_models", fake_models)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.multisession_assignment", fake_assignment)


def test_loso_calibration_trains_on_other_subjects(tmp_path, monkeypatch, write_raw_npy_session):
    subject_a = tmp_path / "jm001"
    subject_b = tmp_path / "jm002"
    _write_subject(subject_a, write_raw_npy_session)
    _write_subject(subject_b, write_raw_npy_session)
    _install_fake_pyrecest(monkeypatch)

    from bayescatrack.association import calibrated_costs
    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(
        calibrated_costs,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )
    monkeypatch.setattr(
        global_assignment,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    results = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="global-assignment",
            split="leave-one-subject-out",
            cost="calibrated",
            max_gap=2,
            include_behavior=False,
        )
    )

    assert [result.subject for result in results] == ["jm001", "jm002"]
    for result in results:
        row = result.to_dict()
        assert row["variant"] == "Calibrated costs + LOSO global assignment"
        assert row["pairwise_f1"] == pytest.approx(1.0)
        assert row["complete_track_f1"] == pytest.approx(1.0)
        assert row["training_examples"] == 12
        assert row["positive_examples"] == 6
        assert row["negative_examples"] == 6


def test_loso_calibration_uses_aligned_rows_when_track2p_reference_is_absent(tmp_path, monkeypatch, write_raw_npy_session):
    subject_a = tmp_path / "jm001"
    subject_b = tmp_path / "jm002"
    _write_aligned_subject(subject_a, write_raw_npy_session)
    _write_aligned_subject(subject_b, write_raw_npy_session)
    _install_fake_pyrecest(monkeypatch)

    from bayescatrack.association import calibrated_costs
    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(
        calibrated_costs,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )
    monkeypatch.setattr(
        global_assignment,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    results = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="global-assignment",
            split="leave-one-subject-out",
            cost="calibrated",
            max_gap=2,
            include_behavior=False,
        )
    )

    assert [result.subject for result in results] == ["jm001", "jm002"]
    assert {result.reference_source for result in results} == {"aligned_subject_rows"}
    assert [result.to_dict()["pairwise_f1"] for result in results] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_loso_calibration_requires_calibrated_cost(tmp_path, write_raw_npy_session):
    _write_subject(tmp_path / "jm001", write_raw_npy_session)
    _write_subject(tmp_path / "jm002", write_raw_npy_session)

    with pytest.raises(ValueError, match="cost='calibrated'"):
        run_track2p_benchmark(
            Track2pBenchmarkConfig(
                data=tmp_path,
                method="global-assignment",
                split="leave-one-subject-out",
                cost="registered-iou",
            )
        )
