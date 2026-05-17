from __future__ import annotations

import numpy as np

from bayescatrack import CalciumPlaneData
from bayescatrack.fov_affine_registration import (
    apply_affine_roi_mask_warp,
    estimate_fov_affine_transform,
    register_measurement_plane_by_fov_affine,
)
from bayescatrack.fov_registration import apply_integer_image_translation
from bayescatrack.track2p_registration import register_plane_pair


def _spot_image(shape: tuple[int, int], centers: tuple[tuple[int, int], ...]) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    for y, x in centers:
        image[y - 1:y + 2, x - 1:x + 2] = 1.0
    return image


def test_apply_affine_roi_mask_warp_applies_translation():
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 3, 4] = True
    affine_xy = np.asarray([[1.0, 0.0, -2.0], [0.0, 1.0, 1.0]], dtype=float)

    registered = apply_affine_roi_mask_warp(masks, affine_xy, output_shape=(8, 8))

    expected = np.zeros_like(masks)
    expected[0, 4, 2] = True
    np.testing.assert_array_equal(registered, expected)


def test_fov_affine_estimate_contains_residual_metadata():
    reference = _spot_image((96, 96), ((20, 20), (20, 75), (72, 24), (75, 78)))
    measurement = apply_integer_image_translation(reference, [-4, 5], output_shape=(96, 96))

    estimate = estimate_fov_affine_transform(reference, measurement)

    assert estimate.matrix_xy.shape == (2, 3)
    assert estimate.inverse_matrix_xy.shape == (2, 3)
    assert estimate.tile_residual_norm.ndim == 1
    assert np.isfinite(estimate.fit_rmse)


def test_fov_affine_registration_recovers_translation_like_fallback():
    reference_fov = _spot_image((96, 96), ((20, 22), (28, 72), (68, 28), (72, 75)))
    measurement_fov = apply_integer_image_translation(reference_fov, [-3, 4], output_shape=(96, 96))
    reference_mask = reference_fov[None, :, :] > 0.0
    measurement_mask = apply_integer_image_translation(
        reference_mask[0], [-3, 4], output_shape=(96, 96)
    )[None, :, :] > 0
    reference_plane = CalciumPlaneData(reference_mask, fov=reference_fov, source="reference")
    measurement_plane = CalciumPlaneData(measurement_mask, fov=measurement_fov, source="measurement")

    registration = register_measurement_plane_by_fov_affine(reference_plane, measurement_plane)
    registered_mask = registration.registered_measurement_plane.roi_masks[0]

    assert registration.registered_measurement_plane.ops["registration_backend"] == "fov-affine"
    assert np.count_nonzero(registered_mask & reference_mask[0]) >= np.count_nonzero(reference_mask[0]) // 2


def test_register_plane_pair_affine_falls_back_to_fov_affine(monkeypatch):
    reference_fov = _spot_image((96, 96), ((20, 20), (24, 72), (70, 30), (74, 78)))
    measurement_fov = apply_integer_image_translation(reference_fov, [2, -3], output_shape=(96, 96))
    reference_plane = CalciumPlaneData(reference_fov[None, :, :] > 0.0, fov=reference_fov, source="reference")
    measurement_plane = CalciumPlaneData(measurement_fov[None, :, :] > 0.0, fov=measurement_fov, source="measurement")

    import bayescatrack.track2p_registration as registration_module

    def _raise_import_error():
        raise ImportError("missing Track2p backend")

    monkeypatch.setattr(registration_module, "_load_track2p_registration_backend", _raise_import_error)
    registered = register_plane_pair(reference_plane, measurement_plane, transform_type="affine")

    assert registered.ops["registration_backend"] == "fov-affine"
