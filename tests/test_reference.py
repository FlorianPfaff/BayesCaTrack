from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import numpy.testing as npt
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from track2p_pyrecest_bridge.reference import (  # noqa: E402
    Track2pReference,
    load_aligned_subject_reference,
    load_track2p_reference,
    pairs_from_label_vectors,
    score_label_vectors_against_reference,
    score_pairwise_matches,
)


class _DummyPlaneData:
    def __init__(self, n_rois: int):
        self.n_rois = int(n_rois)


class _DummySession:
    def __init__(self, name: str, session_date: date | None, n_rois: int):
        self.session_name = name
        self.session_date = session_date
        self.plane_data = _DummyPlaneData(n_rois)


def test_load_track2p_reference_prefers_suite2p_indices_and_extracts_curation(tmp_path: Path):
    track2p_dir = tmp_path / "track2p"
    track2p_dir.mkdir()

    track_ops = {
        "all_ds_path": np.array(
            [
                str(tmp_path / "2024-05-01_a"),
                str(tmp_path / "2024-05-02_a"),
                str(tmp_path / "2024-05-03_a"),
            ],
            dtype=object,
        ),
        "vector_curation_plane_0": np.array([1.0, 0.0, 1.0]),
    }
    np.save(track2p_dir / "track_ops.npy", track_ops, allow_pickle=True)

    reference_matrix = np.array(
        [
            [0, 1, 2],
            [5, None, 7],
            [9, 10, 11],
        ],
        dtype=object,
    )
    np.save(track2p_dir / "plane0_suite2p_indices.npy", reference_matrix, allow_pickle=True)

    reference = load_track2p_reference(track2p_dir, plane_name="plane0")

    assert reference.source == "track2p_output_suite2p_indices"
    assert reference.session_names == ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")
    assert reference.session_dates == (
        date(2024, 5, 1),
        date(2024, 5, 2),
        date(2024, 5, 3),
    )
    npt.assert_array_equal(reference.curated_mask, np.array([True, False, True]))
    npt.assert_array_equal(reference.all_day_mask(), np.array([True, False, True]))
    npt.assert_array_equal(reference.pairwise_matches(0, 2), np.array([[0, 2], [5, 7], [9, 11]]))
    npt.assert_array_equal(
        reference.pairwise_matches(0, 2, curated_only=True),
        np.array([[0, 2], [9, 11]]),
    )


def test_load_track2p_reference_falls_back_to_match_mat(tmp_path: Path):
    track2p_dir = tmp_path / "track2p"
    track2p_dir.mkdir()

    track_ops = {
        "all_ds_path": np.array(
            [
                str(tmp_path / "2024-04-30_a"),
                str(tmp_path / "2024-05-01_a"),
            ],
            dtype=object,
        ),
    }
    np.save(track2p_dir / "track_ops.npy", track_ops, allow_pickle=True)
    np.save(
        track2p_dir / "plane0_match_mat.npy",
        np.array([[0, 4], [1, None]], dtype=object),
        allow_pickle=True,
    )

    reference = load_track2p_reference(track2p_dir, plane_name="plane0")
    assert reference.source == "track2p_output_match_mat"
    npt.assert_array_equal(reference.all_day_mask(), np.array([True, False]))
    npt.assert_array_equal(reference.pairwise_matches(0, 1), np.array([[0, 4]]))


def test_load_aligned_subject_reference_builds_identity_matrix(monkeypatch):
    fake_sessions = [
        _DummySession("2024-05-01_a", date(2024, 5, 1), 3),
        _DummySession("2024-05-02_a", date(2024, 5, 2), 3),
        _DummySession("2024-05-03_a", date(2024, 5, 3), 3),
    ]

    monkeypatch.setattr(
        "track2p_pyrecest_bridge.reference.load_track2p_subject",
        lambda *args, **kwargs: fake_sessions,
    )

    reference = load_aligned_subject_reference("subject_dir", plane_name="plane0")

    npt.assert_array_equal(
        reference.suite2p_indices,
        np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=object,
        ),
    )
    npt.assert_array_equal(reference.curated_mask, np.array([True, True, True]))
    for labels in reference.to_session_track_labels():
        npt.assert_array_equal(labels, np.array([0, 1, 2]))


def test_pairs_from_label_vectors_and_scoring():
    labels_a = np.array([10, -1, 11, 12, None], dtype=object)
    labels_b = np.array([99, 11, 12, 10, -1], dtype=object)

    predicted_pairs = pairs_from_label_vectors(labels_a, labels_b)
    npt.assert_array_equal(predicted_pairs, np.array([[0, 3], [2, 1], [3, 2]]))

    scores = score_pairwise_matches(predicted_pairs, np.array([[0, 3], [2, 1], [4, 0]]))
    assert scores["true_positives"] == 2
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 1
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)


def test_score_label_vectors_against_reference_and_duplicate_detection():
    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[0, 5], [1, 6]], dtype=object),
        curated_mask=np.array([True, True]),
        source="unit_test",
    )

    scores = score_label_vectors_against_reference(
        labels_a=np.array([0, 1]),
        labels_b=np.array([-1, -1, -1, -1, -1, 0, 1]),
        reference=reference,
        session_a=0,
        session_b=1,
    )
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(1.0)

    broken_reference = Track2pReference(
        session_names=("day0",),
        suite2p_indices=np.array([[0], [0]], dtype=object),
    )
    with pytest.raises(ValueError, match="Multiple tracks map to the same ROI"):
        broken_reference.to_session_track_labels(n_rois_per_session=[1])

    with pytest.raises(ValueError, match="same track id"):
        pairs_from_label_vectors(np.array([7, 7]), np.array([7, -1]))
