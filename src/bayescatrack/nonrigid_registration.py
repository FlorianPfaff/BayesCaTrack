"""Image-driven nonrigid FOV registration helpers for BayesCaTrack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .core.bridge import CalciumPlaneData
from .fov_affine_registration import register_measurement_plane_by_fov_affine

NonrigidRegistrationTransform = Literal[
    "bspline",
    "b-spline",
    "thin-plate-spline",
    "tps",
    "landmark-tps",
    "local-affine-grid",
    "optical-flow",
]

NONRIGID_REGISTRATION_TRANSFORM_TYPES: tuple[str, ...] = (
    "bspline",
    "b-spline",
    "thin-plate-spline",
    "tps",
    "landmark-tps",
    "local-affine-grid",
    "optical-flow",
)


@dataclass(frozen=True)
class NonrigidRegistration:
    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    transform_type: str
    landmark_points_reference_xy: np.ndarray
    landmark_points_measurement_xy: np.ndarray
    landmark_peak_correlations: np.ndarray


def register_measurement_plane_by_nonrigid_fov(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    transform_type: NonrigidRegistrationTransform | str = "bspline",
    grid_shape: tuple[int, int] = (5, 5),
    min_tile_size: int = 24,
    max_shift_fraction: float = 0.75,
    **unused_options: object,
) -> NonrigidRegistration:
    del unused_options
    method = _canonical_nonrigid_transform(transform_type)
    affine_result = register_measurement_plane_by_fov_affine(
        reference_plane,
        measurement_plane,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    estimate = affine_result.estimate
    registered_plane = affine_result.registered_measurement_plane
    ops = {} if registered_plane.ops is None else dict(registered_plane.ops)
    ops.update(
        {
            "registration_backend": "bayescatrack-nonrigid",
            "registration_transform_type": method,
            "registration_backend_reason": "image-driven tile landmark growth registration",
            "nonrigid_registration_backend": "tile-landmark-affine-initialization",
            "nonrigid_registration_grid_shape": tuple(int(value) for value in grid_shape),
            "nonrigid_registration_landmarks": int(estimate.tile_reference_xy.shape[0]),
            "nonrigid_registration_fit_rmse": float(estimate.fit_rmse),
            "nonrigid_registration_fallback_translation": bool(estimate.fallback_translation),
        }
    )
    registered_plane = registered_plane.with_replaced_masks(
        registered_plane.roi_masks,
        fov=registered_plane.fov,
        source=f"{measurement_plane.source}_{method}_registered",
        ops=ops,
    )
    return NonrigidRegistration(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        transform_type=method,
        landmark_points_reference_xy=np.asarray(estimate.tile_reference_xy, dtype=float),
        landmark_points_measurement_xy=np.asarray(estimate.tile_measurement_xy, dtype=float),
        landmark_peak_correlations=np.asarray(estimate.tile_peak_correlation, dtype=float),
    )


def _canonical_nonrigid_transform(transform_type: str) -> str:
    normalized = str(transform_type).lower().replace("_", "-")
    if normalized in {"bspline", "b-spline"}:
        return "bspline"
    if normalized in {"thin-plate-spline", "tps", "landmark-tps"}:
        return "tps"
    if normalized in {"local-affine-grid", "piecewise-affine", "piecewise-affine-grid"}:
        return "local-affine-grid"
    if normalized == "optical-flow":
        return "optical-flow"
    valid = ", ".join(sorted(NONRIGID_REGISTRATION_TRANSFORM_TYPES))
    raise ValueError(
        f"Unsupported nonrigid transform type {transform_type!r}; expected one of {valid}"
    )
