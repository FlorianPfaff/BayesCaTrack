from __future__ import annotations

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.nonrigid_registration import (
    NONRIGID_REGISTRATION_TRANSFORM_TYPES,
    register_measurement_plane_by_nonrigid_fov,
)
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES, register_plane_pair


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


def test_nonrigid_transform_names_are_registered() -> None:
    assert {"bspline", "tps", "local-affine-grid", "optical-flow"}.issubset(
        set(NONRIGID_REGISTRATION_TRANSFORM_TYPES)
    )
    assert set(NONRIGID_REGISTRATION_TRANSFORM_TYPES).issubset(
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
