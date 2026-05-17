from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CalciumPlaneData
from .fov_registration import estimate_integer_fov_shift


@dataclass(frozen=True)
class FovAffineEstimate:
    matrix_xy: np.ndarray
    inverse_matrix_xy: np.ndarray
    tile_reference_xy: np.ndarray
    tile_measurement_xy: np.ndarray
    tile_shift_yx: np.ndarray
    tile_peak_correlation: np.ndarray
    tile_residual_norm: np.ndarray
    fit_rmse: float
    fallback_translation: bool = False


@dataclass(frozen=True)
class FovAffineRegistration:
    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    estimate: FovAffineEstimate


def register_measurement_plane_by_fov_affine(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    subtract_mean: bool = True,
    grid_shape: tuple[int, int] = (3, 3),
    min_tile_size: int = 32,
    max_shift_fraction: float = 0.55,
) -> FovAffineRegistration:
    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError("Both planes must provide fov images for FOV-affine registration")
    estimate = estimate_fov_affine_transform(
        reference_plane.fov,
        measurement_plane.fov,
        subtract_mean=subtract_mean,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    masks = apply_affine_roi_mask_warp(
        measurement_plane.roi_masks,
        estimate.matrix_xy,
        output_shape=reference_plane.image_shape,
    )
    ops = {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    ops.update(
        {
            "registration_backend": "fov-affine",
            "registration_transform_type": "fov-affine",
            "registration_backend_reason": "tile-wise FOV phase-correlation affine fallback",
            "fov_registration_method": "tile_phase_correlation_affine",
            "fov_affine_matrix_xy": estimate.matrix_xy,
            "fov_affine_tile_count": int(estimate.tile_reference_xy.shape[0]),
            "fov_affine_fit_rmse": float(estimate.fit_rmse),
            "fov_affine_fallback_translation": bool(estimate.fallback_translation),
        }
    )
    registered_plane = measurement_plane.with_replaced_masks(
        masks,
        fov=reference_plane.fov,
        source=f"{measurement_plane.source}_fov_affine_registered",
        ops=ops,
    )
    return FovAffineRegistration(reference_plane, measurement_plane, registered_plane, estimate)


def estimate_fov_affine_transform(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
    grid_shape: tuple[int, int] = (3, 3),
    min_tile_size: int = 32,
    max_shift_fraction: float = 0.55,
) -> FovAffineEstimate:
    reference = np.asarray(reference_fov, dtype=float)
    measurement = np.asarray(measurement_fov, dtype=float)
    if reference.ndim != 2 or measurement.ndim != 2:
        raise ValueError("reference_fov and measurement_fov must be 2-D")
    shape = (max(reference.shape[0], measurement.shape[0]), max(reference.shape[1], measurement.shape[1]))
    reference = _pad(reference, shape)
    measurement = _pad(measurement, shape)
    ref_xy, meas_xy, shifts_yx, peaks = _tile_correspondences(
        reference,
        measurement,
        subtract_mean=subtract_mean,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    if ref_xy.shape[0] < 3 or np.linalg.matrix_rank(_design(meas_xy)) < 3:
        return _translation_estimate(reference, measurement, subtract_mean=subtract_mean)
    coef, _, _, _ = np.linalg.lstsq(_design(meas_xy), ref_xy, rcond=None)
    matrix_xy = np.asarray(coef.T, dtype=float)
    residual = _design(meas_xy) @ coef - ref_xy
    estimate = _make_estimate(matrix_xy, ref_xy, meas_xy, shifts_yx, peaks, residual, False)
    if ref_xy.shape[0] > 3:
        keep = _inlier_mask(np.linalg.norm(residual, axis=1))
        if np.count_nonzero(keep) >= 3 and np.linalg.matrix_rank(_design(meas_xy[keep])) >= 3:
            coef, _, _, _ = np.linalg.lstsq(_design(meas_xy[keep]), ref_xy[keep], rcond=None)
            matrix_xy = np.asarray(coef.T, dtype=float)
            residual = _design(meas_xy[keep]) @ coef - ref_xy[keep]
            estimate = _make_estimate(matrix_xy, ref_xy[keep], meas_xy[keep], shifts_yx[keep], peaks[keep], residual, False)
    return estimate


def apply_affine_roi_mask_warp(
    roi_masks: np.ndarray,
    matrix_xy: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    roi_masks = np.asarray(roi_masks)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if matrix_xy.shape != (2, 3):
        raise ValueError("matrix_xy must have shape (2, 3)")
    output = np.zeros((roi_masks.shape[0], int(output_shape[0]), int(output_shape[1])), dtype=roi_masks.dtype)
    linear = matrix_xy[:, :2]
    offset = matrix_xy[:, 2]
    for roi_index, mask in enumerate(roi_masks):
        yy, xx = np.nonzero(mask)
        if yy.size == 0:
            continue
        src_xy = np.column_stack((xx.astype(float), yy.astype(float)))
        dst_xy = src_xy @ linear.T + offset[None, :]
        x = np.rint(dst_xy[:, 0]).astype(int)
        y = np.rint(dst_xy[:, 1]).astype(int)
        valid = (x >= 0) & (x < output.shape[2]) & (y >= 0) & (y < output.shape[1])
        if not np.any(valid):
            continue
        if roi_masks.dtype == np.bool_:
            output[roi_index, y[valid], x[valid]] = True
        else:
            values = np.asarray(mask[yy[valid], xx[valid]], dtype=output.dtype)
            np.maximum.at(output[roi_index], (y[valid], x[valid]), values)
    return output


def invert_affine_xy(matrix_xy: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix_xy, dtype=float)[:, :2]
    offset = np.asarray(matrix_xy, dtype=float)[:, 2]
    try:
        inv_linear = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        inv_linear = np.linalg.pinv(linear)
    return np.column_stack((inv_linear, -inv_linear @ offset))


def _tile_correspondences(reference, measurement, *, subtract_mean, grid_shape, min_tile_size, max_shift_fraction):
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    y_edges = np.linspace(0, reference.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, reference.shape[1], cols + 1, dtype=int)
    ref_points, meas_points, shifts, peaks = [], [], [], []
    for row in range(rows):
        for col in range(cols):
            ys = slice(int(y_edges[row]), int(y_edges[row + 1]))
            xs = slice(int(x_edges[col]), int(x_edges[col + 1]))
            if ys.stop - ys.start < min_tile_size or xs.stop - xs.start < min_tile_size:
                continue
            ref_tile = reference[ys, xs]
            meas_tile = measurement[ys, xs]
            if _low_info(ref_tile) or _low_info(meas_tile):
                continue
            shift_yx, peak = estimate_integer_fov_shift(ref_tile, meas_tile, subtract_mean=subtract_mean)
            max_shift = max(abs(int(shift_yx[0])), abs(int(shift_yx[1])))
            if max_shift > max(ys.stop - ys.start, xs.stop - xs.start) * max_shift_fraction:
                continue
            center = np.asarray([0.5 * (xs.start + xs.stop - 1), 0.5 * (ys.start + ys.stop - 1)], dtype=float)
            shift_xy = np.asarray([float(shift_yx[1]), float(shift_yx[0])], dtype=float)
            ref_points.append(center)
            meas_points.append(center - shift_xy)
            shifts.append(shift_yx.astype(float))
            peaks.append(float(peak))
    if not ref_points:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    return np.vstack(ref_points), np.vstack(meas_points), np.vstack(shifts), np.asarray(peaks, dtype=float)


def _translation_estimate(reference, measurement, *, subtract_mean):
    shift_yx, peak = estimate_integer_fov_shift(reference, measurement, subtract_mean=subtract_mean)
    matrix_xy = np.asarray([[1.0, 0.0, float(shift_yx[1])], [0.0, 1.0, float(shift_yx[0])]], dtype=float)
    return _make_estimate(matrix_xy, np.zeros((0, 2)), np.zeros((0, 2)), np.asarray(shift_yx, dtype=float).reshape(1, 2), np.asarray([peak]), np.zeros((0, 2)), True)


def _make_estimate(matrix_xy, ref_xy, meas_xy, shifts_yx, peaks, residual, fallback):
    residual_norm = np.linalg.norm(residual, axis=1) if residual.size else np.zeros((0,), dtype=float)
    return FovAffineEstimate(
        matrix_xy,
        invert_affine_xy(matrix_xy),
        np.asarray(ref_xy, dtype=float),
        np.asarray(meas_xy, dtype=float),
        np.asarray(shifts_yx, dtype=float),
        np.asarray(peaks, dtype=float),
        residual_norm,
        float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else 0.0,
        fallback,
    )


def _design(xy):
    return np.column_stack((xy, np.ones((xy.shape[0],), dtype=float)))


def _inlier_mask(residual_norm):
    finite = residual_norm[np.isfinite(residual_norm)]
    if finite.size == 0:
        return np.ones_like(residual_norm, dtype=bool)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return np.isfinite(residual_norm) & (residual_norm <= max(med + 3.0 * 1.4826 * mad, np.percentile(finite, 75), 1.0))


def _low_info(tile):
    finite = np.asarray(tile, dtype=float)
    finite = finite[np.isfinite(finite)]
    return finite.size == 0 or float(np.std(finite)) <= 1.0e-8


def _pad(image, shape):
    if image.shape == shape:
        return image
    out = np.zeros(shape, dtype=image.dtype)
    out[: image.shape[0], : image.shape[1]] = image
    return out
