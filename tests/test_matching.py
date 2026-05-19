from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.matching import SessionMatchResult, build_track_rows_from_matches


@pytest.mark.parametrize(
    ("match", "message"),
    [
        (
            np.array([[1, 10], [1, 11]], dtype=int),
            "duplicate reference ROI index 1",
        ),
        (([1, 2], [10, 10]), "duplicate measurement ROI index 10"),
        ({1: 10, 2: 10}, "duplicate measurement ROI index 10"),
    ],
)
def test_build_track_rows_rejects_non_one_to_one_match_inputs(match, message):
    with pytest.raises(ValueError, match=message):
        build_track_rows_from_matches(
            ["s0", "s1"],
            [match],
            start_roi_indices=[1, 2],
        )


def test_session_match_result_rejects_duplicate_roi_indices_when_stitching():
    match = SessionMatchResult(
        reference_session_name="s0",
        measurement_session_name="s1",
        reference_positions=np.array([0, 1], dtype=int),
        measurement_positions=np.array([0, 1], dtype=int),
        reference_roi_indices=np.array([1, 1], dtype=int),
        measurement_roi_indices=np.array([10, 11], dtype=int),
        costs=np.array([0.1, 0.2], dtype=float),
    )

    with pytest.raises(ValueError, match="duplicate reference ROI index 1"):
        build_track_rows_from_matches(
            ["s0", "s1"],
            [match],
            start_roi_indices=[1, 2],
        )


def test_build_track_rows_accepts_one_to_one_array_match_input():
    rows = build_track_rows_from_matches(
        ["s0", "s1"],
        [np.array([[1, 10], [2, 11]], dtype=int)],
        start_roi_indices=[1, 2],
    )

    assert rows.tolist() == [[1, 10], [2, 11]]
