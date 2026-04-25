from __future__ import annotations

import sys
import types

import numpy as np
import numpy.testing as npt

from bayescatrack.registration import (
    build_registered_consecutive_session_association_bundles,
    build_registered_session_pair_association_bundle,
)


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
            self.matched_costs = np.zeros(self.matched_reference_indices.shape[0], dtype=float)
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
        transform = AffineTransform(np.eye(reference_points.shape[1], dtype=float), offset)
        assignment = np.arange(reference_points.shape[0], dtype=int)
        return RegistrationResult(
            transform=transform,
            assignment=assignment,
            transformed_reference_points=transform.apply(reference_points),
        )

    setattr(fake_registration, "AffineTransform", AffineTransform)
    setattr(fake_registration, "RegistrationResult", RegistrationResult)
    setattr(fake_registration, "joint_registration_assignment", joint_registration_assignment)

    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.point_set_registration", fake_registration)


def test_build_registered_session_pair_association_bundle_recovers_translation(
    monkeypatch,
    make_track2p_session,
    assert_diagonal_association,
):
    _install_fake_point_set_registration(monkeypatch)

    image_shape = (4, 6)
    reference_masks = np.zeros((2, *image_shape), dtype=bool)
    reference_masks[0, 0:2, 0:2] = True
    reference_masks[1, 1:3, 3:5] = True

    shifted_masks = np.zeros_like(reference_masks)
    shifted_masks[0, 0:2, 1:3] = True
    shifted_masks[1, 1:3, 4:6] = True

    reference_session = make_track2p_session("2024-05-01_a", reference_masks)
    measurement_session = make_track2p_session("2024-05-02_a", shifted_masks)

    registered_bundle = build_registered_session_pair_association_bundle(
        reference_session,
        measurement_session,
        registration_model="translation",
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )

    npt.assert_allclose(
        registered_bundle.plane_registration.reference_to_measurement_matrix,
        np.eye(2),
    )
    npt.assert_allclose(
        registered_bundle.plane_registration.reference_to_measurement_offset,
        np.array([1.0, 0.0]),
    )
    npt.assert_allclose(
        registered_bundle.plane_registration.measurement_to_reference_offset,
        np.array([-1.0, 0.0]),
    )
    npt.assert_allclose(
        registered_bundle.plane_registration.registered_measurement_plane.centroids(order="xy"),
        reference_session.plane_data.centroids(order="xy"),
    )
    assert_diagonal_association(registered_bundle.association_bundle, iou_atol=1e-8)


def test_build_registered_consecutive_session_association_bundles(
    monkeypatch,
    make_track2p_session,
):
    _install_fake_point_set_registration(monkeypatch)

    image_shape = (4, 6)
    masks_a = np.zeros((1, *image_shape), dtype=bool)
    masks_a[0, 1:3, 1:3] = True

    masks_b = np.zeros((1, *image_shape), dtype=bool)
    masks_b[0, 1:3, 2:4] = True

    masks_c = np.zeros((1, *image_shape), dtype=bool)
    masks_c[0, 1:3, 3:5] = True

    sessions = [
        make_track2p_session("2024-05-01_a", masks_a),
        make_track2p_session("2024-05-02_a", masks_b),
        make_track2p_session("2024-05-03_a", masks_c),
    ]
    registered = build_registered_consecutive_session_association_bundles(
        sessions,
        registration_model="translation",
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )
    assert len(registered.bundles) == 2
    for bundle in registered.bundles:
        assert bundle.association_bundle.pairwise_cost_matrix.shape == (1, 1)
        npt.assert_allclose(
            bundle.association_bundle.pairwise_components["iou"],
            np.ones((1, 1)),
            atol=1e-8,
        )
