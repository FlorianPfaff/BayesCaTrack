"""Image-driven dense nonrigid FOV registration helpers for BayesCaTrack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .core.bridge import CalciumPlaneData
from .fov_affine_registration import estimate_fov_affine_transform

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
    """Result of an image-driven dense nonrigid measurement-to-reference warp."""

    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    transform_type: str
    landmark_points_reference_xy: np.ndarray
    landmark_points_measurement_xy: np.ndarray
    landmark_peak_correlations: np.ndarray
    inverse_y: np.ndarray
    inverse_x: np.ndarray


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def register_measurement_plane_by_nonrigid_fov(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    transform_type: NonrigidRegistrationTransform | str = "bspline",
    grid_shape: tuple[int, int] = (5, 5),
    min_tile_size: int = 24,
    max_shift_fraction: float = 0.75,
    tps_regularization: float = 1.0e-3,
    bspline_regularization: float = 1.0e-2,
    optical_flow_iterations: int = 12,
    optical_flow_alpha: float = 25.0,
    **unused_options: object,
) -> NonrigidRegistration:
    """Register measurement ROIs into the reference FOV with an inverse dense warp."""

    del unused_options
    method = _canonical_nonrigid_transform(transform_type)
    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError("Both planes must provide FOV images for nonrigid registration")
    if grid_shape[0] < 2 or grid_shape[1] < 2:
        raise ValueError("grid_shape must contain at least two tiles per axis")
    if tps_regularization < 0.0:
        raise ValueError("tps_regularization must be non-negative")
    if bspline_regularization < 0.0:
        raise ValueError("bspline_regularization must be non-negative")
    if optical_flow_iterations < 0:
        raise ValueError("optical_flow_iterations must be non-negative")
    if optical_flow_alpha <= 0.0:
        raise ValueError("optical_flow_alpha must be strictly positive")

    reference = _finite_image(reference_plane.fov)
    measurement = _finite_image(measurement_plane.fov)
    estimate = estimate_fov_affine_transform(
        reference,
        measurement,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    output_shape = reference_plane.image_shape
    base_y, base_x = _affine_inverse_grid(estimate.inverse_matrix_xy, output_shape)
    bspline_control_shape = _bspline_control_shape(output_shape, grid_shape)

    reference_xy = np.asarray(estimate.tile_reference_xy, dtype=float)
    measurement_xy = np.asarray(estimate.tile_measurement_xy, dtype=float)
    if reference_xy.shape[0] >= 3:
        if method == "tps":
            inverse_y, inverse_x = _tps_inverse_grid(
                reference_xy,
                measurement_xy,
                output_shape,
                tps_regularization=tps_regularization,
                fallback_y=base_y,
                fallback_x=base_x,
            )
            backend = "thin-plate-spline-landmark-warp"
        elif method == "bspline":
            inverse_y, inverse_x = _bspline_inverse_grid(
                reference_xy,
                measurement_xy,
                output_shape,
                fallback_y=base_y,
                fallback_x=base_x,
                control_shape_yx=bspline_control_shape,
                regularization=bspline_regularization,
            )
            backend = "tensor-product-cubic-bspline-landmark-warp"
        else:
            nearest = 4 if method == "local-affine-grid" else None
            inverse_y, inverse_x = _idw_inverse_grid(
                reference_xy,
                measurement_xy,
                output_shape,
                fallback_y=base_y,
                fallback_x=base_x,
                nearest=nearest,
                smooth_iterations=0,
            )
            backend = (
                "local-landmark-grid-warp"
                if method == "local-affine-grid"
                else "smooth-landmark-displacement-warp"
            )
    else:
        inverse_y, inverse_x = base_y, base_x
        backend = "affine-fallback-insufficient-landmarks"

    if method == "optical-flow":
        inverse_y, inverse_x = _refine_inverse_grid_by_intensity_flow(
            reference,
            measurement,
            inverse_y,
            inverse_x,
            iterations=optical_flow_iterations,
            alpha=optical_flow_alpha,
        )
        backend = "landmark-warp-with-intensity-flow-refinement"

    registered_masks = _warp_mask_stack_nearest(
        measurement_plane.roi_masks,
        inverse_y,
        inverse_x,
        output_shape=output_shape,
    )
    registered_fov = _warp_image_bilinear(measurement, inverse_y, inverse_x)
    ops = {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    valid_fraction = float(np.mean(_valid_sample_mask(inverse_y, inverse_x, measurement.shape)))
    ops.update(
        {
            "registration_backend": "bayescatrack-nonrigid",
            "registration_transform_type": method,
            "registration_backend_reason": "image-driven dense inverse FOV warp",
            "nonrigid_registration_backend": backend,
            "nonrigid_registration_grid_shape": tuple(int(value) for value in grid_shape),
            "nonrigid_registration_landmarks": int(reference_xy.shape[0]),
            "nonrigid_registration_fit_rmse": float(estimate.fit_rmse),
            "nonrigid_registration_fallback_translation": bool(estimate.fallback_translation),
            "nonrigid_registration_inverse_warp_valid_fraction": valid_fraction,
            "nonrigid_registration_tps_regularization": float(tps_regularization),
            "nonrigid_registration_bspline_regularization": float(bspline_regularization),
            "nonrigid_registration_bspline_control_shape": tuple(
                int(value) for value in bspline_control_shape
            ),
            "nonrigid_registration_optical_flow_iterations": int(optical_flow_iterations),
            "nonrigid_registration_optical_flow_alpha": float(optical_flow_alpha),
        }
    )
    registered_plane = measurement_plane.with_replaced_masks(
        registered_masks,
        fov=registered_fov,
        source=f"{measurement_plane.source}_{method}_registered",
        ops=ops,
    )
    return NonrigidRegistration(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        transform_type=method,
        landmark_points_reference_xy=reference_xy,
        landmark_points_measurement_xy=measurement_xy,
        landmark_peak_correlations=np.asarray(estimate.tile_peak_correlation, dtype=float),
        inverse_y=inverse_y,
        inverse_x=inverse_x,
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


def _finite_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("FOV images must be two-dimensional")
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _affine_inverse_grid(
    inverse_matrix_xy: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(output_shape, dtype=float)
    query_xy = np.column_stack((xx.ravel(), yy.ravel()))
    source_xy = (
        query_xy @ np.asarray(inverse_matrix_xy[:, :2], dtype=float).T
        + np.asarray(inverse_matrix_xy[:, 2], dtype=float)[None, :]
    )
    return source_xy[:, 1].reshape(output_shape), source_xy[:, 0].reshape(output_shape)


def _idw_inverse_grid(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    output_shape: tuple[int, int],
    *,
    fallback_y: np.ndarray,
    fallback_x: np.ndarray,
    nearest: int | None = None,
    smooth_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(output_shape, dtype=float)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    displacement = np.asarray(measurement_xy, dtype=float) - np.asarray(reference_xy, dtype=float)
    result = np.empty_like(query)
    chunk_size = 65536
    smooth_scale = _landmark_spacing(reference_xy)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        chunk = query[start:stop]
        dist2 = np.sum((chunk[:, None, :] - reference_xy[None, :, :]) ** 2, axis=2)
        if nearest is not None and reference_xy.shape[0] > nearest:
            keep = np.argpartition(dist2, nearest - 1, axis=1)[:, :nearest]
            rows = np.arange(dist2.shape[0])[:, None]
            selected_dist2 = dist2[rows, keep]
            selected_disp = displacement[keep]
            weights = 1.0 / np.maximum(selected_dist2 + smooth_scale**2, 1.0e-6)
            disp = np.sum(weights[:, :, None] * selected_disp, axis=1) / np.sum(
                weights,
                axis=1,
            )[:, None]
        else:
            weights = 1.0 / np.maximum(dist2 + smooth_scale**2, 1.0e-6)
            disp = weights @ displacement / np.sum(weights, axis=1)[:, None]
        result[start:stop] = chunk + disp
    inv_x = result[:, 0].reshape(output_shape)
    inv_y = result[:, 1].reshape(output_shape)
    inv_y = _fill_invalid(inv_y, fallback_y)
    inv_x = _fill_invalid(inv_x, fallback_x)
    for _ in range(smooth_iterations):
        inv_y = _smooth_field(inv_y, fallback_y)
        inv_x = _smooth_field(inv_x, fallback_x)
    return inv_y, inv_x


def _bspline_control_shape(
    output_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    """Return a padded cubic B-spline control lattice shape in y/x order."""

    del output_shape
    return (
        max(4, int(grid_shape[0]) + 3),
        max(4, int(grid_shape[1]) + 3),
    )


# pylint: disable=too-many-arguments,too-many-locals
def _bspline_inverse_grid(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    output_shape: tuple[int, int],
    *,
    fallback_y: np.ndarray,
    fallback_x: np.ndarray,
    control_shape_yx: tuple[int, int],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    control_disp_y, control_disp_x = _fit_bspline_control_displacements(
        reference_xy,
        measurement_xy,
        output_shape,
        fallback_y=fallback_y,
        fallback_x=fallback_x,
        control_shape_yx=control_shape_yx,
        regularization=regularization,
    )
    yy, xx = np.indices(output_shape, dtype=float)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    inv_y_flat = np.empty(query.shape[0], dtype=float)
    inv_x_flat = np.empty(query.shape[0], dtype=float)
    chunk_size = 32768
    disp_y_flat = control_disp_y.ravel()
    disp_x_flat = control_disp_x.ravel()
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        chunk = query[start:stop]
        basis = _bspline_design_matrix(chunk, output_shape, control_shape_yx)
        inv_x_flat[start:stop] = chunk[:, 0] + basis @ disp_x_flat
        inv_y_flat[start:stop] = chunk[:, 1] + basis @ disp_y_flat
    inv_y = inv_y_flat.reshape(output_shape)
    inv_x = inv_x_flat.reshape(output_shape)
    return _fill_invalid(inv_y, fallback_y), _fill_invalid(inv_x, fallback_x)


# pylint: disable=too-many-arguments,too-many-locals
def _fit_bspline_control_displacements(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    output_shape: tuple[int, int],
    *,
    fallback_y: np.ndarray,
    fallback_x: np.ndarray,
    control_shape_yx: tuple[int, int],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference_xy = np.asarray(reference_xy, dtype=float)
    measurement_xy = np.asarray(measurement_xy, dtype=float)
    displacement_xy = measurement_xy - reference_xy
    design = _bspline_design_matrix(reference_xy, output_shape, control_shape_yx)
    control_xy = _bspline_control_positions(output_shape, control_shape_yx)
    fallback_control_x = _sample_scalar_field_bilinear(fallback_x, control_xy)
    fallback_control_y = _sample_scalar_field_bilinear(fallback_y, control_xy)
    fallback_disp_x = fallback_control_x - control_xy[:, 0]
    fallback_disp_y = fallback_control_y - control_xy[:, 1]
    penalty = float(max(regularization, 0.0))
    normal = design.T @ design + penalty * np.eye(design.shape[1])
    rhs_x = design.T @ displacement_xy[:, 0] + penalty * fallback_disp_x
    rhs_y = design.T @ displacement_xy[:, 1] + penalty * fallback_disp_y
    coef_x = _solve_or_lstsq(normal, rhs_x)
    coef_y = _solve_or_lstsq(normal, rhs_y)
    return coef_y.reshape(control_shape_yx), coef_x.reshape(control_shape_yx)


def _solve_or_lstsq(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _bspline_design_matrix(
    points_xy: np.ndarray,
    output_shape: tuple[int, int],
    control_shape_yx: tuple[int, int],
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    control_y, control_x = int(control_shape_yx[0]), int(control_shape_yx[1])
    control_coord_y, control_coord_x = _bspline_control_coordinates(
        points_xy,
        output_shape,
        control_shape_yx,
    )
    base_y = np.floor(control_coord_y).astype(int)
    base_x = np.floor(control_coord_x).astype(int)
    weights_y = _cubic_bspline_weights(control_coord_y - base_y)
    weights_x = _cubic_bspline_weights(control_coord_x - base_x)
    design = np.zeros((points_xy.shape[0], control_y * control_x), dtype=float)
    rows = np.arange(points_xy.shape[0])
    for offset_y in range(4):
        node_y = np.clip(base_y + offset_y - 1, 0, control_y - 1)
        for offset_x in range(4):
            node_x = np.clip(base_x + offset_x - 1, 0, control_x - 1)
            columns = node_y * control_x + node_x
            weights = weights_y[:, offset_y] * weights_x[:, offset_x]
            np.add.at(design, (rows, columns), weights)
    return design


def _bspline_control_coordinates(
    points_xy: np.ndarray,
    output_shape: tuple[int, int],
    control_shape_yx: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(output_shape[0]), int(output_shape[1])
    control_y, control_x = int(control_shape_yx[0]), int(control_shape_yx[1])
    scale_x = (control_x - 3) / max(width - 1, 1)
    scale_y = (control_y - 3) / max(height - 1, 1)
    return 1.0 + points_xy[:, 1] * scale_y, 1.0 + points_xy[:, 0] * scale_x


def _bspline_control_positions(
    output_shape: tuple[int, int],
    control_shape_yx: tuple[int, int],
) -> np.ndarray:
    height, width = int(output_shape[0]), int(output_shape[1])
    control_y, control_x = int(control_shape_yx[0]), int(control_shape_yx[1])
    node_y, node_x = np.indices((control_y, control_x), dtype=float)
    x = (node_x.ravel() - 1.0) * max(width - 1, 1) / max(control_x - 3, 1)
    y = (node_y.ravel() - 1.0) * max(height - 1, 1) / max(control_y - 3, 1)
    return np.column_stack((x, y))


def _sample_scalar_field_bilinear(field: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    x = np.clip(points_xy[:, 0], 0.0, max(field.shape[1] - 1, 0))
    y = np.clip(points_xy[:, 1], 0.0, max(field.shape[0] - 1, 0))
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, field.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, field.shape[1] - 1)
    wy = y - y0
    wx = x - x0
    return (
        (1.0 - wy) * (1.0 - wx) * field[y0, x0]
        + (1.0 - wy) * wx * field[y0, x1]
        + wy * (1.0 - wx) * field[y1, x0]
        + wy * wx * field[y1, x1]
    )


def _cubic_bspline_weights(fraction: np.ndarray) -> np.ndarray:
    fraction = np.asarray(fraction, dtype=float)
    return np.column_stack(
        (
            ((1.0 - fraction) ** 3) / 6.0,
            (3.0 * fraction**3 - 6.0 * fraction**2 + 4.0) / 6.0,
            (-3.0 * fraction**3 + 3.0 * fraction**2 + 3.0 * fraction + 1.0)
            / 6.0,
            fraction**3 / 6.0,
        )
    )


def _tps_inverse_grid(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    output_shape: tuple[int, int],
    *,
    tps_regularization: float,
    fallback_y: np.ndarray,
    fallback_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        params_x = _fit_tps(reference_xy, measurement_xy[:, 0], tps_regularization)
        params_y = _fit_tps(reference_xy, measurement_xy[:, 1], tps_regularization)
        yy, xx = np.indices(output_shape, dtype=float)
        query = np.column_stack((xx.ravel(), yy.ravel()))
        inv_x = _eval_tps(query, reference_xy, params_x).reshape(output_shape)
        inv_y = _eval_tps(query, reference_xy, params_y).reshape(output_shape)
        return _fill_invalid(inv_y, fallback_y), _fill_invalid(inv_x, fallback_x)
    except np.linalg.LinAlgError:
        return _idw_inverse_grid(
            reference_xy,
            measurement_xy,
            output_shape,
            fallback_y=fallback_y,
            fallback_x=fallback_x,
            nearest=None,
            smooth_iterations=2,
        )


def _fit_tps(points_xy: np.ndarray, values: np.ndarray, regularization: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    values = np.asarray(values, dtype=float)
    distances = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    kernel = _tps_kernel(distances)
    kernel += float(regularization) * np.eye(points_xy.shape[0])
    polynomial = np.column_stack(
        (np.ones(points_xy.shape[0], dtype=float), points_xy),
    )
    system = np.block(
        [
            [kernel, polynomial],
            [polynomial.T, np.zeros((3, 3), dtype=float)],
        ]
    )
    rhs = np.concatenate((values, np.zeros(3, dtype=float)))
    return np.linalg.solve(system, rhs)


def _eval_tps(query_xy: np.ndarray, control_xy: np.ndarray, params: np.ndarray) -> np.ndarray:
    n_control = control_xy.shape[0]
    distances = np.linalg.norm(query_xy[:, None, :] - control_xy[None, :, :], axis=2)
    polynomial = np.column_stack((np.ones(query_xy.shape[0], dtype=float), query_xy))
    return _tps_kernel(distances) @ params[:n_control] + polynomial @ params[n_control:]


def _tps_kernel(radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = radius**2 * np.log(radius)
    values[~np.isfinite(values)] = 0.0
    return values


def _refine_inverse_grid_by_intensity_flow(
    reference: np.ndarray,
    measurement: np.ndarray,
    inverse_y: np.ndarray,
    inverse_x: np.ndarray,
    *,
    iterations: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = _pad_or_crop(reference, inverse_y.shape)
    inv_y = np.asarray(inverse_y, dtype=float).copy()
    inv_x = np.asarray(inverse_x, dtype=float).copy()
    for _ in range(iterations):
        warped = _warp_image_bilinear(measurement, inv_y, inv_x)
        grad_y, grad_x = np.gradient(warped)
        residual = reference - warped
        denom = grad_x * grad_x + grad_y * grad_y + float(alpha) ** 2
        update_x = _smooth_array(residual * grad_x / denom)
        update_y = _smooth_array(residual * grad_y / denom)
        inv_x = np.clip(inv_x + update_x, 0.0, max(measurement.shape[1] - 1, 0))
        inv_y = np.clip(inv_y + update_y, 0.0, max(measurement.shape[0] - 1, 0))
    return inv_y, inv_x


def _warp_mask_stack_nearest(
    masks: np.ndarray,
    inverse_y: np.ndarray,
    inverse_x: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    mask_array = np.asarray(masks)
    result = np.zeros((mask_array.shape[0], *output_shape), dtype=mask_array.dtype)
    y = np.rint(inverse_y).astype(int)
    x = np.rint(inverse_x).astype(int)
    valid = _valid_sample_mask(inverse_y, inverse_x, mask_array.shape[1:])
    for roi_index, mask in enumerate(mask_array):
        sampled = np.zeros(output_shape, dtype=mask_array.dtype)
        sampled[valid] = mask[y[valid], x[valid]]
        result[roi_index] = sampled
    return result


def _warp_image_bilinear(image: np.ndarray, inverse_y: np.ndarray, inverse_x: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    output = np.zeros(inverse_y.shape, dtype=float)
    valid = _valid_sample_mask(inverse_y, inverse_x, image.shape)
    if not np.any(valid):
        return output
    y0 = np.floor(inverse_y[valid]).astype(int)
    x0 = np.floor(inverse_x[valid]).astype(int)
    y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    wy = inverse_y[valid] - y0
    wx = inverse_x[valid] - x0
    output[valid] = (
        (1.0 - wy) * (1.0 - wx) * image[y0, x0]
        + (1.0 - wy) * wx * image[y0, x1]
        + wy * (1.0 - wx) * image[y1, x0]
        + wy * wx * image[y1, x1]
    )
    return output


def _valid_sample_mask(inverse_y: np.ndarray, inverse_x: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    return (
        np.isfinite(inverse_y)
        & np.isfinite(inverse_x)
        & (inverse_y >= 0.0)
        & (inverse_x >= 0.0)
        & (inverse_y <= max(int(image_shape[0]) - 1, 0))
        & (inverse_x <= max(int(image_shape[1]) - 1, 0))
    )


def _fill_invalid(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(result)
    result[invalid] = np.asarray(fallback, dtype=float)[invalid]
    return result


def _smooth_field(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    smoothed = _smooth_array(values)
    invalid = ~np.isfinite(smoothed)
    smoothed[invalid] = fallback[invalid]
    return smoothed


def _smooth_array(values: np.ndarray) -> np.ndarray:
    return (
        4.0 * values
        + np.roll(values, 1, axis=0)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=1)
    ) / 8.0


def _landmark_spacing(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 1.0
    distances = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    distances = distances[distances > 0.0]
    if not distances.size:
        return 1.0
    return float(max(np.median(distances) * 0.25, 1.0))


def _pad_or_crop(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(output_shape, dtype=float)
    height = min(image.shape[0], output_shape[0])
    width = min(image.shape[1], output_shape[1])
    result[:height, :width] = image[:height, :width]
    return result
