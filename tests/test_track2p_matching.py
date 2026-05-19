import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.matching import (
    SessionMatchResult,
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self, costs):
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.array([10, 20], dtype=int)
        self.measurement_roi_indices = np.array([100, 200], dtype=int)


def test_build_track_rows_from_consecutive_matches():
    rows = build_track_rows_from_matches(
        ("s1", "s2", "s3"),
        [np.array([[0, 1], [2, 3]]), np.array([[1, 5], [3, 6]])],
        start_roi_indices=np.array([0, 2]),
    )

    npt.assert_array_equal(rows, np.array([[0, 1, 5], [2, 3, 6]]))


def test_build_track_rows_from_later_seed_session_stitches_both_directions():
    rows = build_track_rows_from_matches(
        ("s1", "s2", "s3"),
        [np.array([[0, 10], [2, 20]]), np.array([[10, 5], [20, 6]])],
        start_roi_indices=np.array([10, 20, 30]),
        start_session_index=1,
    )

    npt.assert_array_equal(
        rows,
        np.array([[0, 10, 5], [2, 20, 6], [-1, 30, -1]]),
    )


def test_build_track_rows_from_tuple_matches_rejects_duplicate_reference_indices():
    with pytest.raises(ValueError, match="duplicate reference ROI index 1"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [([1, 1], [10, 11])],
            start_roi_indices=[1],
        )


def test_build_track_rows_from_array_matches_rejects_duplicate_reference_indices():
    with pytest.raises(ValueError, match="duplicate reference ROI index 1"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[1, 10], [1, 11]], dtype=int)],
            start_roi_indices=[1],
        )


def test_build_track_rows_from_mapping_matches_rejects_duplicate_measurement_indices():
    with pytest.raises(ValueError, match="duplicate measurement ROI index 10"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [{1: 10, 2: 10}],
            start_roi_indices=[1, 2],
        )


def test_build_track_rows_from_session_match_result_rejects_duplicate_reference_indices():
    match_result = SessionMatchResult(
        reference_session_name="s1",
        measurement_session_name="s2",
        reference_positions=np.array([0, 1], dtype=int),
        measurement_positions=np.array([0, 1], dtype=int),
        reference_roi_indices=np.array([1, 1], dtype=int),
        measurement_roi_indices=np.array([10, 11], dtype=int),
        costs=np.array([0.0, 0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="duplicate reference ROI index 1"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [match_result],
            start_roi_indices=[1],
        )


def test_solve_bundle_linear_assignment_uses_default_cost_gate():
    result = solve_bundle_linear_assignment(_Bundle([[0.0, 100.0], [100.0, 100.0]]))

    npt.assert_array_equal(result.reference_roi_indices, np.array([10]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100]))
    npt.assert_array_equal(result.costs, np.array([0.0]))


def test_solve_bundle_linear_assignment_can_disable_cost_gate():
    result = solve_bundle_linear_assignment(
        _Bundle([[0.0, 100.0], [100.0, 100.0]]),
        max_cost=None,
    )

    npt.assert_array_equal(result.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 200]))
    npt.assert_array_equal(result.costs, np.array([0.0, 100.0]))


def test_solve_bundle_linear_assignment_gates_before_hungarian():
    result = solve_bundle_linear_assignment(_Bundle([[6.0, 7.0], [0.0, 6.0]]))

    npt.assert_array_equal(result.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 200]))
    npt.assert_array_equal(result.costs, np.array([6.0, 6.0]))


def test_build_track_rows_from_bundles_uses_default_cost_gate():
    session_names, rows, match_results = build_track_rows_from_bundles(
        [_Bundle([[0.0, 100.0], [100.0, 100.0]])]
    )

    assert session_names == ("s1", "s2")
    npt.assert_array_equal(rows, np.array([[10, 100], [20, -1]]))
    assert match_results[0].n_matches == 1
    npt.assert_array_equal(
        match_results[0].reference_roi_indices,
        np.array([10]),
    )
    npt.assert_array_equal(
        match_results[0].measurement_roi_indices,
        np.array([100]),
    )
    npt.assert_array_equal(match_results[0].costs, np.array([0.0]))


def test_build_track_rows_from_bundles_can_disable_cost_gate():
    session_names, rows, match_results = build_track_rows_from_bundles(
        [_Bundle([[0.0, 100.0], [100.0, 100.0]])],
        max_cost=None,
    )

    assert session_names == ("s1", "s2")
    npt.assert_array_equal(rows, np.array([[10, 100], [20, 200]]))
    assert match_results[0].n_matches == 2
    npt.assert_array_equal(match_results[0].costs, np.array([0.0, 100.0]))
