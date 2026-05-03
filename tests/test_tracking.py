from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.tracking import run_registered_subject_tracking


# jscpd:ignore-start
def _install_fake_point_set_registration(monkeypatch) -> None:
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_registration = types.ModuleType("pyrecest.utils.point_set_registration")

    class AffineTransform:
        def __init__(self, matrix, offset):
            self.matrix = np.asarray(matrix, dtype=float)
            self.offset = np.asarray(offset, dtype=float)

        def apply(self, points):
            points = np.asarray(points, dtype=float)
            return (self.matrix @ points.T).T + self.offset

    # pylint: disable=too-many-instance-attributes
    class RegistrationResult:
        def __init__(self, transform, assignment, transformed_reference_points):
            self.transform = transform
            self.assignment = assignment
            self.matched_reference_indices = np.where(assignment >= 0)[0]
            self.matched_moving_indices = assignment[self.matched_reference_indices]
            self.transformed_reference_points = transformed_reference_points
            self.matched_costs = np.zeros(
                self.matched_reference_indices.shape[0], dtype=float
            )
            self.rmse = 0.0
            self.n_iterations = 1
            self.converged = True

    def joint_registration_assignment(
        reference_points,
        moving_points,
        *,
        model="affine",
        max_cost=float("inf"),
        **_,
    ):
        del model, max_cost
        reference_points = np.asarray(reference_points, dtype=float)
        moving_points = np.asarray(moving_points, dtype=float)
        offset = np.mean(moving_points, axis=0) - np.mean(reference_points, axis=0)
        transform = AffineTransform(
            np.eye(reference_points.shape[1], dtype=float), offset
        )
        assignment = np.arange(reference_points.shape[0], dtype=int)
        return RegistrationResult(
            transform=transform,
            assignment=assignment,
            transformed_reference_points=transform.apply(reference_points),
        )

    setattr(fake_registration, "AffineTransform", AffineTransform)
    setattr(fake_registration, "RegistrationResult", RegistrationResult)
    setattr(
        fake_registration,
        "joint_registration_assignment",
        joint_registration_assignment,
    )

    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "pyrecest.utils.point_set_registration", fake_registration
    )


# jscpd:ignore-end


def _write_raw_npy_session(
    subject_dir: Path,
    session_name: str,
    roi_masks: np.ndarray,
    *,
    offset: float,
) -> None:
    plane_dir = subject_dir / session_name / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(
        plane_dir / "F.npy",
        np.asarray(
            [
                [offset, offset + 1.0],
                [offset + 2.0, offset + 3.0],
                [offset + 4.0, offset + 5.0],
            ],
            dtype=float,
        ),
    )
    np.save(plane_dir / "fov.npy", np.full(roi_masks.shape[1:], offset, dtype=float))


def _make_three_roi_masks(shift_x: int = 0) -> np.ndarray:
    masks = np.zeros((3, 8, 10), dtype=bool)
    roi_columns = (
        slice(shift_x, 2 + shift_x),
        slice(4 + shift_x, 6 + shift_x),
        slice(1 + shift_x, 3 + shift_x),
    )
    masks[0, 0:2, roi_columns[0]] = True
    masks[1, 2:4, roi_columns[1]] = True
    masks[2, 5:7, roi_columns[2]] = True
    return masks


def test_run_registered_subject_tracking_builds_full_track_rows(
    tmp_path: Path, monkeypatch
):
    _install_fake_point_set_registration(monkeypatch)
    subject_dir = tmp_path / "jm271"
    for session_name, shift_x, offset in (
        ("2024-05-01_a", 0, 0.0),
        ("2024-05-02_a", 1, 10.0),
        ("2024-05-03_a", 2, 20.0),
    ):
        _write_raw_npy_session(
            subject_dir,
            session_name,
            _make_three_roi_masks(shift_x),
            offset=offset,
        )

    result = run_registered_subject_tracking(
        subject_dir,
        plane_name="plane0",
        input_format="auto",
        include_behavior=False,
        registration_model="affine",
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )

    assert result.session_names == ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")
    npt.assert_array_equal(
        result.track_rows,
        np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=int,
        ),
    )
    assert result.link_costs.shape == (3, 2)
    assert np.all(np.isfinite(result.link_costs))
    assert len(result.registered_bundles.bundles) == 2
    assert [match.n_matches for match in result.match_results] == [3, 3]

    scores = result.score_summary()
    assert scores["n_tracks_started"] == 3
    assert scores["n_complete_tracks"] == 3
    assert scores["complete_track_fraction"] == pytest.approx(1.0)
    assert scores["n_pairwise_matches"] == 6
    assert scores["mean_track_length"] == pytest.approx(3.0)
    assert scores["pairs"][0]["n_reference_rois"] == 3
    assert scores["pairs"][0]["n_measurement_rois"] == 3
    assert scores["pairs"][0]["reference_match_fraction"] == pytest.approx(1.0)
    assert scores["pairs"][0]["measurement_match_fraction"] == pytest.approx(1.0)

    export = result.to_export_dict()
    npt.assert_array_equal(export["track_rows"], result.track_rows)
    npt.assert_array_equal(export["track_lengths"], np.array([3, 3, 3], dtype=int))
    npt.assert_array_equal(export["complete_track_mask"], np.array([True, True, True]))


def test_run_registered_subject_tracking_handles_single_session(tmp_path: Path):
    subject_dir = tmp_path / "jm271"
    _write_raw_npy_session(
        subject_dir,
        "2024-05-01_a",
        _make_three_roi_masks(0),
        offset=0.0,
    )

    result = run_registered_subject_tracking(
        subject_dir,
        plane_name="plane0",
        input_format="auto",
        include_behavior=False,
    )

    npt.assert_array_equal(result.track_rows, np.array([[0], [1], [2]], dtype=int))
    assert result.link_costs.shape == (3, 0)
    assert result.match_results == ()
    scores = result.score_summary()
    assert scores["n_tracks_started"] == 3
    assert scores["n_complete_tracks"] == 3
    assert scores["n_pairwise_matches"] == 0
