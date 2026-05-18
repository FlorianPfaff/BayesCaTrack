from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack.reference import (  # noqa: E402
    Track2pReference,
    load_aligned_subject_reference,
    load_track2p_reference,
    pairs_from_label_vectors,
    score_complete_tracks,
    score_complete_tracks_against_reference,
    score_label_vectors_against_reference,
    score_pairwise_matches,
)


class _DummyPlaneData:
    def __init__(self, n_rois: int, roi_indices: list[int] | None = None):
        self.n_rois = int(n_rois)
        if roi_indices is not None:
            self.roi_indices = np.asarray(roi_indices, dtype=int)


class _DummySession:
    def __init__(
        self,
        name: str,
        session_date: date | None,
        n_rois: int,
        roi_indices: list[int] | None = None,
    ):
        self.session_name = name
        self.session_date = session_date
        self.plane_data = _DummyPlaneData(n_rois, roi_indices=roi_indices)


def test_load_track2p_reference_prefers_suite2p_indices_and_extracts_curation(
    tmp_path: Path,
):
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
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy", reference_matrix, allow_pickle=True
    )

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
    npt.assert_array_equal(
        reference.pairwise_matches(0, 2), np.array([[0, 2], [5, 7], [9, 11]])
    )
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
        "bayescatrack.reference.load_track2p_subject",
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


def test_load_aligned_subject_reference_preserves_original_suite2p_indices(
    monkeypatch,
):
    fake_sessions = [
        _DummySession(
            "2024-05-01_a",
            date(2024, 5, 1),
            3,
            roi_indices=[0, 2, 5],
        ),
        _DummySession(
            "2024-05-02_a",
            date(2024, 5, 2),
            2,
            roi_indices=[4, 6],
        ),
    ]

    monkeypatch.setattr(
        "bayescatrack.reference.load_track2p_subject",
        lambda *args, **kwargs: fake_sessions,
    )

    reference = load_aligned_subject_reference("subject_dir", plane_name="plane0")

    npt.assert_array_equal(
        reference.suite2p_indices,
        np.array(
            [
                [0, 4],
                [2, 6],
                [5, None],
            ],
            dtype=object,
        ),
    )
    npt.assert_array_equal(
        reference.pairwise_matches(0, 1),
        np.array([[0, 4], [2, 6]]),
    )

    labels_a, labels_b = reference.to_session_track_labels()
    npt.assert_array_equal(labels_a, np.array([0, -1, 1, -1, -1, 2]))
    npt.assert_array_equal(labels_b, np.array([-1, -1, -1, -1, 0, -1, 1]))


@pytest.mark.parametrize("roi_indices", ([7], [-1, 3], [2, 2]))
def test_load_aligned_subject_reference_validates_roi_indices(
    monkeypatch,
    roi_indices,
):
    fake_sessions = [
        _DummySession(
            "2024-05-01_a",
            date(2024, 5, 1),
            2,
            roi_indices=roi_indices,
        )
    ]
    monkeypatch.setattr(
        "bayescatrack.reference.load_track2p_subject",
        lambda *args, **kwargs: fake_sessions,
    )

    with pytest.raises(ValueError, match="roi_indices"):
        load_aligned_subject_reference("subject_dir", plane_name="plane0")


def test_complete_tracks_and_complete_tracks_score_against_reference():
    reference = Track2pReference(
        session_names=("day0", "day1", "day2"),
        suite2p_indices=np.array(
            [
                [0, 10, 20],
                [1, 11, 21],
                [2, 12, None],
                [3, 13, 23],
            ],
            dtype=object,
        ),
        curated_mask=np.array([True, True, True, False]),
        source="unit_test",
    )

    npt.assert_array_equal(
        reference.complete_tracks(curated_only=True),
        np.array([[0, 10, 20], [1, 11, 21]]),
    )
    npt.assert_array_equal(
        reference.complete_tracks(session_indices=(0, 1), curated_only=True),
        np.array([[0, 10], [1, 11], [2, 12]]),
    )

    predicted_tracks = np.array(
        [
            [0, 10, 20],
            [1, 11, 99],
            [2, 12, 22],
            [99, 98, 97],
        ],
        dtype=object,
    )
    scores = score_complete_tracks_against_reference(
        predicted_tracks,
        reference,
        curated_only=True,
    )
    assert scores["perfectly_reconstructed_tracks"] == 1
    assert scores["reconstructed_complete_tracks"] == 3
    assert scores["ground_truth_complete_tracks"] == 2
    assert scores["T_rc"] == 1
    assert scores["T_c"] == 3
    assert scores["T_gt"] == 2
    assert scores["complete_tracks_score"] == pytest.approx(2 / 5)
    assert scores["ct"] == pytest.approx(2 / 5)

    prefix_scores = score_complete_tracks_against_reference(
        predicted_tracks,
        reference,
        session_indices=(0, 1),
        curated_only=True,
    )
    assert prefix_scores["T_rc"] == 3
    assert prefix_scores["T_c"] == 3
    assert prefix_scores["T_gt"] == 3
    assert prefix_scores["ct"] == pytest.approx(1.0)


def test_score_complete_tracks_counts_duplicate_predictions_as_reconstructed_tracks():
    scores = score_complete_tracks(
        np.array([[0, 1], [0, 1], [2, None]], dtype=object),
        np.array([[0, 1], [2, 3]], dtype=object),
    )

    assert scores["T_rc"] == 1
    assert scores["T_c"] == 2
    assert scores["T_gt"] == 2
    assert scores["ct"] == pytest.approx(0.5)


def test_score_complete_tracks_validates_shapes():
    with pytest.raises(ValueError, match="same number of sessions"):
        score_complete_tracks(
            np.array([[0, 1]], dtype=object),
            np.array([[0, 1, 2]], dtype=object),
        )

    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[0, 1]], dtype=object),
    )
    with pytest.raises(ValueError, match="reference.n_sessions"):
        score_complete_tracks_against_reference(
            np.array([[0, 1, 2]], dtype=object),
            reference,
        )


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
