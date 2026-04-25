from __future__ import annotations

import numpy as np
import numpy.testing as npt

from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    build_fov_registered_consecutive_session_association_bundles,
    build_fov_registered_session_pair_association_bundle,
    estimate_integer_fov_shift,
    register_measurement_plane_by_fov_translation,
)


def test_estimate_integer_fov_shift_recovers_known_translation():
    reference_fov = np.zeros((8, 9), dtype=float)
    reference_fov[2:5, 3:7] = 1.0
    measurement_fov = apply_integer_image_translation(reference_fov, np.array([1, 2]))

    shift_yx, peak_correlation = estimate_integer_fov_shift(reference_fov, measurement_fov)

    npt.assert_array_equal(shift_yx, np.array([-1, -2]))
    assert peak_correlation > 0.9


def test_register_measurement_plane_by_fov_translation_aligns_masks_and_fov(
    make_track2p_session,
):
    reference_masks = np.zeros((1, 8, 9), dtype=bool)
    reference_masks[0, 2:5, 3:6] = True
    reference_fov = reference_masks[0].astype(float) + 0.1

    measurement_masks = np.zeros_like(reference_masks)
    measurement_masks[0] = apply_integer_image_translation(reference_masks[0], np.array([1, 2]), fill_value=False)
    measurement_fov = apply_integer_image_translation(reference_fov, np.array([1, 2]))

    reference_session = make_track2p_session("2024-05-01_a", reference_masks, fov=reference_fov)
    measurement_session = make_track2p_session("2024-05-02_a", measurement_masks, fov=measurement_fov)

    registration = register_measurement_plane_by_fov_translation(reference_session.plane_data, measurement_session.plane_data)
    registered_fov = registration.registered_measurement_plane.fov
    expected_registered_fov = apply_integer_image_translation(measurement_fov, np.array([-1, -2]))

    npt.assert_array_equal(registration.measurement_to_reference_shift_yx, np.array([-1, -2]))
    npt.assert_array_equal(registration.registered_measurement_plane.roi_masks, reference_session.plane_data.roi_masks)
    npt.assert_allclose(registered_fov, expected_registered_fov)

    # Zero-padded translation cannot recover nonzero FOV background outside the overlap.
    npt.assert_allclose(registered_fov[:-1, :-2], reference_session.plane_data.fov[:-1, :-2])
    npt.assert_array_equal(registered_fov[-1:, :], np.zeros_like(registered_fov[-1:, :]))
    npt.assert_array_equal(registered_fov[:, -2:], np.zeros_like(registered_fov[:, -2:]))


def test_build_fov_registered_session_pair_association_bundle_prefers_diagonal_matches(
    make_track2p_session,
    assert_diagonal_association,
):
    image_shape = (8, 10)

    reference_masks = np.zeros((2, *image_shape), dtype=bool)
    reference_masks[0, 1:3, 1:3] = True
    reference_masks[1, 4:6, 6:8] = True
    reference_fov = reference_masks.sum(axis=0, dtype=float)

    measurement_masks = np.zeros_like(reference_masks)
    for roi_index in range(reference_masks.shape[0]):
        measurement_masks[roi_index] = apply_integer_image_translation(reference_masks[roi_index], np.array([1, -1]), fill_value=False)
    measurement_fov = apply_integer_image_translation(reference_fov, np.array([1, -1]))

    reference_session = make_track2p_session("2024-05-01_a", reference_masks, fov=reference_fov)
    measurement_session = make_track2p_session("2024-05-02_a", measurement_masks, fov=measurement_fov)

    registered_bundle = build_fov_registered_session_pair_association_bundle(
        reference_session,
        measurement_session,
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )

    assert_diagonal_association(registered_bundle.association_bundle)


def test_build_fov_registered_consecutive_session_association_bundles(
    make_track2p_session,
):
    image_shape = (8, 10)
    masks_a = np.zeros((1, *image_shape), dtype=bool)
    masks_a[0, 2:4, 2:4] = True
    fov_a = masks_a[0].astype(float)

    masks_b = np.zeros_like(masks_a)
    masks_b[0] = apply_integer_image_translation(masks_a[0], np.array([1, 0]), fill_value=False)
    fov_b = apply_integer_image_translation(fov_a, np.array([1, 0]))

    masks_c = np.zeros_like(masks_a)
    masks_c[0] = apply_integer_image_translation(masks_b[0], np.array([0, 2]), fill_value=False)
    fov_c = apply_integer_image_translation(fov_b, np.array([0, 2]))

    sessions = [
        make_track2p_session("2024-05-01_a", masks_a, fov=fov_a),
        make_track2p_session("2024-05-02_a", masks_b, fov=fov_b),
        make_track2p_session("2024-05-03_a", masks_c, fov=fov_c),
    ]

    bundles = build_fov_registered_consecutive_session_association_bundles(
        sessions,
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )

    assert len(bundles) == 2
    for bundle in bundles:
        npt.assert_allclose(bundle.association_bundle.pairwise_components["iou"], np.ones((1, 1)))
