from __future__ import annotations

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.nonrigid_registration import register_measurement_plane_by_nonrigid_fov


def _spot(shape: tuple[int, int], center_yx: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape)
    y, x = center_yx
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 18.0)


def _plane(centers_yx: list[tuple[int, int]]) -> CalciumPlaneData:
    shape = (72, 72)
    masks = np.zeros((len(centers_yx), *shape), dtype=bool)
    fov = np.zeros(shape, dtype=float)
    for index, (y, x) in enumerate(centers_yx):
        y_start, y_stop = y - 2, y + 3
        x_start, x_stop = x - 2, x + 3
        masks[index, y_start:y_stop, x_start:x_stop] = True
        fov += _spot(shape, (y, x))
    return CalciumPlaneData(
        roi_masks=masks,
        fov=fov,
        roi_indices=np.arange(len(centers_yx), dtype=int),
        source="synthetic",
    )


def test_bspline_uses_tensor_product_cubic_control_lattice_backend() -> None:
    reference_centers = [
        (14, 14),
        (14, 36),
        (14, 58),
        (36, 14),
        (36, 36),
        (36, 58),
        (58, 14),
        (58, 36),
        (58, 58),
    ]
    moving_centers = [
        (int(round(y * 0.98 + 1)), int(round(x * 1.04 - 2)))
        for y, x in reference_centers
    ]

    registration = register_measurement_plane_by_nonrigid_fov(
        _plane(reference_centers),
        _plane(moving_centers),
        transform_type="bspline",
        grid_shape=(3, 3),
        min_tile_size=12,
    )

    ops = registration.registered_measurement_plane.ops
    assert ops is not None
    assert ops["nonrigid_registration_backend"] == (
        "tensor-product-cubic-bspline-landmark-warp"
    )
    assert ops["nonrigid_registration_bspline_control_shape"] == (6, 6)
