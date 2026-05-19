from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from bayescatrack.association_guided_registration import (
    ASSOCIATION_GUIDED_NONRIGID_REGISTRATION_TRANSFORM_TYPES,
    register_measurement_plane_by_association_guided_nonrigid,
)
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.nonrigid_registration import (
    NONRIGID_REGISTRATION_TRANSFORM_TYPES,
    register_measurement_plane_by_nonrigid_fov,
)
from bayescatrack.track2p_registration import (
    REGISTRATION_TRANSFORM_TYPES,
    register_plane_pair,
)


def _spot_image(shape: tuple[int, int], center_yx: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape)
    y, x = center_yx
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 18.0)


def _plane(shape: tuple[int, int], center_yx: tuple[int, int]) -> CalciumPlaneData:
    mask = np.zeros((1, *shape), dtype=bool)
    y, x = center_yx
    y_start, y_stop = y - 2, y + 3
    x_start, x_stop = x - 2, x + 3
    mask[0, y_start:y_stop, x_start:x_stop] = True
    return CalciumPlaneData(
        roi_masks=mask,
        fov=_spot_image(shape, center_yx),
        roi_indices=np.asarray([0], dtype=int),
        source="synthetic",
    )


def _multi_spot_image(shape: tuple[int, int], centers_xy: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    image = np.zeros(shape, dtype=float)
    for x_value, y_value in np.asarray(centers_xy, dtype=float):
        image += np.exp(-((yy - y_value) ** 2 + (xx - x_value) ** 2) / 18.0)
    return image


def _disk_masks(
    centers_xy: np.ndarray,
    *,
    shape: tuple[int, int] = (80, 80),
    radius: float = 2.5,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    centers_xy = np.asarray(centers_xy, dtype=float)
    masks = np.zeros((centers_xy.shape[0], *shape), dtype=bool)
    for roi_index, (x_value, y_value) in enumerate(centers_xy):
        masks[roi_index] = (xx - x_value) ** 2 + (yy - y_value) ** 2 <= radius**2
    return masks


def _apply_affine_xy(points_xy: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    return points_xy @ matrix_xy[:, :2].T + matrix_xy[:, 2][None, :]


def _fit_affine_xy(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    design = np.column_stack((source_xy, np.ones(source_xy.shape[0], dtype=float)))
    coef, _, _, _ = np.linalg.lstsq(design, target_xy, rcond=None)
    return np.asarray(coef.T, dtype=float)


def _invert_affine_xy(matrix_xy: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix_xy, dtype=float)[:, :2]
    offset = np.asarray(matrix_xy, dtype=float)[:, 2]
    inverse_linear = np.linalg.inv(linear)
    return np.column_stack((inverse_linear, -inverse_linear @ offset))


def _median_link_error(candidate_xy: np.ndarray, reference_xy: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(candidate_xy - reference_xy, axis=1)))


def _growth_planes_and_estimate():
    shape = (80, 80)
    anchor_xy = np.asarray([10.0, 60.0], dtype=float)
    reference_xy = np.asarray(
        [
            [14.0, 56.0],
            [28.0, 54.0],
            [50.0, 52.0],
            [18.0, 38.0],
            [36.0, 36.0],
            [58.0, 34.0],
            [22.0, 20.0],
            [42.0, 18.0],
            [62.0, 14.0],
        ],
        dtype=float,
    )
    delta_xy = reference_xy - anchor_xy[None, :]
    normalized_distance = 0.5 * (
        delta_xy[:, 0] / 55.0 + (anchor_xy[1] - reference_xy[:, 1]) / 50.0
    )
    growth_scale = 1.08 + 0.22 * normalized_distance + 0.10 * normalized_distance**2
    measurement_xy = anchor_xy[None, :] + delta_xy / growth_scale[:, None]

    reference = CalciumPlaneData(
        roi_masks=_disk_masks(reference_xy, shape=shape),
        fov=_multi_spot_image(shape, reference_xy),
        roi_indices=np.arange(reference_xy.shape[0]),
        source="reference_growth",
    )
    measurement = CalciumPlaneData(
        roi_masks=_disk_masks(measurement_xy, shape=shape),
        fov=_multi_spot_image(shape, measurement_xy),
        roi_indices=np.arange(measurement_xy.shape[0]),
        source="measurement_growth",
    )
    affine_xy = _fit_affine_xy(measurement_xy, reference_xy)
    affine_registered_xy = _apply_affine_xy(measurement_xy, affine_xy)
    residual = affine_registered_xy - reference_xy
    estimate = SimpleNamespace(
        inverse_matrix_xy=_invert_affine_xy(affine_xy),
        tile_reference_xy=reference_xy,
        tile_measurement_xy=measurement_xy,
        tile_peak_correlation=np.ones(reference_xy.shape[0], dtype=float),
        fit_rmse=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        fallback_translation=False,
    )
    return reference, measurement, reference_xy, measurement_xy, affine_xy, estimate


def test_nonrigid_transform_names_are_registered() -> None:
    assert {"bspline", "tps", "local-affine-grid", "optical-flow"}.issubset(
        set(NONRIGID_REGISTRATION_TRANSFORM_TYPES)
    )
    assert set(NONRIGID_REGISTRATION_TRANSFORM_TYPES).issubset(
        set(REGISTRATION_TRANSFORM_TYPES)
    )
    assert set(ASSOCIATION_GUIDED_NONRIGID_REGISTRATION_TRANSFORM_TYPES).issubset(
        set(REGISTRATION_TRANSFORM_TYPES)
    )


def test_nonrigid_registration_routes_through_public_pair_registration() -> None:
    reference = _plane((72, 72), (36, 34))
    moving = _plane((72, 72), (32, 37))

    registered = register_plane_pair(reference, moving, transform_type="bspline")

    assert registered.ops is not None
    assert registered.ops["registration_backend"] == "bayescatrack-nonrigid"
    assert registered.ops["registration_transform_type"] == "bspline"
    assert registered.roi_masks.shape == reference.roi_masks.shape


def test_nonrigid_registration_accepts_tps_alias() -> None:
    reference = _plane((72, 72), (36, 34))
    moving = _plane((72, 72), (32, 37))

    registration = register_measurement_plane_by_nonrigid_fov(
        reference,
        moving,
        transform_type="thin-plate-spline",
    )

    assert registration.transform_type == "tps"
    assert registration.inverse_y.shape == reference.image_shape
    assert registration.inverse_x.shape == reference.image_shape


def test_tps_registration_handles_lower_left_anchor_upper_right_growth() -> None:
    reference, measurement, reference_xy, measurement_xy, affine_xy, estimate = (
        _growth_planes_and_estimate()
    )
    translation_xy = measurement_xy + np.mean(reference_xy - measurement_xy, axis=0)
    affine_registered_xy = _apply_affine_xy(measurement_xy, affine_xy)

    translation_error = _median_link_error(translation_xy, reference_xy)
    affine_error = _median_link_error(affine_registered_xy, reference_xy)

    with patch(
        "bayescatrack.nonrigid_registration.estimate_fov_affine_transform",
        return_value=estimate,
    ):
        registration = register_measurement_plane_by_nonrigid_fov(
            reference,
            measurement,
            transform_type="tps",
            grid_shape=(3, 3),
            min_tile_size=16,
        )

    registered_xy = registration.registered_measurement_plane.centroids(order="xy").T
    tps_error = _median_link_error(registered_xy, reference_xy)

    assert affine_error < 0.75 * translation_error
    assert tps_error < 0.75 * affine_error
    assert tps_error < 3.0
    assert registration.registered_measurement_plane.ops is not None
    assert (
        registration.registered_measurement_plane.ops["nonrigid_registration_backend"]
        == "thin-plate-spline-landmark-warp"
    )


def test_association_guided_tps_refines_from_unsupervised_pseudo_links() -> None:
    reference, measurement, reference_xy, measurement_xy, affine_xy, estimate = (
        _growth_planes_and_estimate()
    )
    affine_registered_xy = _apply_affine_xy(measurement_xy, affine_xy)
    affine_error = _median_link_error(affine_registered_xy, reference_xy)

    with patch(
        "bayescatrack.association_guided_registration.estimate_fov_affine_transform",
        return_value=estimate,
    ):
        registration = register_measurement_plane_by_association_guided_nonrigid(
            reference,
            measurement,
            transform_type="association-guided-tps",
            grid_shape=(3, 3),
            min_tile_size=16,
            iterations=2,
            min_pseudo_links=3,
            pseudo_link_cost_threshold=8.0,
        )

    registered_xy = registration.registered_measurement_plane.centroids(order="xy").T
    guided_error = _median_link_error(registered_xy, reference_xy)

    assert registration.selected_pseudo_links.shape[0] >= 3
    assert guided_error < 0.75 * affine_error
    assert guided_error < 3.0
    assert registration.registered_measurement_plane.ops is not None
    assert (
        registration.registered_measurement_plane.ops["registration_backend"]
        == "bayescatrack-association-guided-nonrigid"
    )
    assert (
        registration.registered_measurement_plane.ops["registration_transform_type"]
        == "association-guided-tps"
    )
