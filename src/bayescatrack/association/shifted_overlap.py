"""Local shift-search overlap costs for registered calcium-imaging ROIs.

The functions in this module implement the shifted-IoU diagnostic/cost proposed
for cases where a global registration leaves small local integer-pixel residuals.
Unlike dilation, every candidate benefits only from a coherent translation of the
whole measurement ROI, so shape selectivity is preserved in crowded fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.core import _bridge_impl  # pylint: disable=protected-access
from bayescatrack.core.bridge import CalciumPlaneData

PairwiseCostMethod = Callable[
    ..., np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]
]
SHIFTED_OVERLAP_KWARG_NAMES = frozenset(
    (
        "shifted_iou_radius",
        "use_shifted_iou_for_iou_cost",
        "shifted_iou_weight",
        "use_shifted_mask_cosine_for_mask_cosine_cost",
        "shifted_mask_cosine_weight",
        "shifted_iou_shift_penalty_weight",
        "shifted_iou_shift_penalty_scale",
    )
)


def pairwise_kwargs_use_shifted_overlap(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
) -> bool:
    """Return whether pairwise cost kwargs require shifted-overlap support."""

    if not pairwise_cost_kwargs:
        return False
    return any(key in pairwise_cost_kwargs for key in SHIFTED_OVERLAP_KWARG_NAMES)


def shifted_iou_pairwise_cost_matrix(
    original_method: PairwiseCostMethod,
    self: CalciumPlaneData,
    other: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Wrap ``CalciumPlaneData.build_pairwise_cost_matrix`` with shifted overlap.

    Recognized keyword arguments are removed before delegating to the original
    method, so this wrapper can be installed without changing the public core
    method signature.

    Parameters
    ----------
    shifted_iou_radius
        Maximum absolute integer shift, in pixels, applied to measurement masks.
        A value of zero disables this wrapper and preserves exact-IoU behavior.
    use_shifted_iou_for_iou_cost
        Replace the standard IoU cost term with best local shifted IoU.
    shifted_iou_weight
        Add an additional shifted-IoU cost term without replacing exact IoU.
    use_shifted_mask_cosine_for_mask_cosine_cost
        Replace the standard mask-cosine term with best local shifted cosine.
    shifted_mask_cosine_weight
        Add an additional shifted-mask-cosine cost term.
    shifted_iou_shift_penalty_weight
        Add a penalty proportional to the residual shift needed by the best
        shifted-IoU match. This keeps shifted IoU from treating large local
        shifts as equivalent to exact local registration.
    """

    shifted_iou_radius = _nonnegative_int(
        kwargs.pop("shifted_iou_radius", 0), name="shifted_iou_radius"
    )
    shifted_iou_weight = float(kwargs.pop("shifted_iou_weight", 0.0) or 0.0)
    shifted_mask_cosine_weight = float(
        kwargs.pop("shifted_mask_cosine_weight", 0.0) or 0.0
    )
    shifted_iou_shift_penalty_weight = float(
        kwargs.pop("shifted_iou_shift_penalty_weight", 0.0) or 0.0
    )
    shifted_iou_shift_penalty_scale = kwargs.pop(
        "shifted_iou_shift_penalty_scale", None
    )
    use_shifted_iou_for_iou_cost = bool(
        kwargs.pop("use_shifted_iou_for_iou_cost", False)
    )
    use_shifted_mask_cosine_for_mask_cosine_cost = bool(
        kwargs.pop("use_shifted_mask_cosine_for_mask_cosine_cost", False)
    )

    if shifted_iou_weight < 0.0:
        raise ValueError("shifted_iou_weight must be non-negative")
    if shifted_mask_cosine_weight < 0.0:
        raise ValueError("shifted_mask_cosine_weight must be non-negative")
    if shifted_iou_shift_penalty_weight < 0.0:
        raise ValueError("shifted_iou_shift_penalty_weight must be non-negative")
    if shifted_iou_shift_penalty_scale is not None:
        shifted_iou_shift_penalty_scale = float(shifted_iou_shift_penalty_scale)
        if shifted_iou_shift_penalty_scale <= 0.0:
            raise ValueError(
                "shifted_iou_shift_penalty_scale must be strictly positive"
            )

    return_components = bool(kwargs.get("return_components", False))
    uses_shifted_overlap = shifted_iou_radius > 0 and (
        return_components
        or use_shifted_iou_for_iou_cost
        or shifted_iou_weight > 0.0
        or use_shifted_mask_cosine_for_mask_cosine_cost
        or shifted_mask_cosine_weight > 0.0
        or shifted_iou_shift_penalty_weight > 0.0
    )
    if not uses_shifted_overlap:
        return original_method(self, other, **kwargs)

    similarity_epsilon = float(kwargs.get("similarity_epsilon", 1.0e-6))
    large_cost = float(kwargs.get("large_cost", 1.0e6))
    iou_weight = float(kwargs.get("iou_weight", 6.0) or 0.0)
    mask_cosine_weight = float(kwargs.get("mask_cosine_weight", 2.0) or 0.0)

    base_kwargs = dict(kwargs)
    if use_shifted_iou_for_iou_cost:
        base_kwargs["iou_weight"] = 0.0
    if use_shifted_mask_cosine_for_mask_cosine_cost:
        base_kwargs["mask_cosine_weight"] = 0.0
    base_kwargs["return_components"] = return_components

    base_result = original_method(self, other, **base_kwargs)
    if return_components:
        base_cost, components = base_result
        components = dict(components)
    else:
        base_cost = base_result
        components = {}

    needs_shifted_cosine = (
        return_components
        or use_shifted_mask_cosine_for_mask_cosine_cost
        or shifted_mask_cosine_weight > 0.0
    )
    shifted = pairwise_shifted_overlap_matrices(
        self.roi_masks,
        other.roi_masks,
        radius=shifted_iou_radius,
        include_mask_cosine=needs_shifted_cosine,
        similarity_epsilon=similarity_epsilon,
    )
    shifted_iou = shifted["shifted_iou"]
    shifted_iou_shift_norm = shifted["shifted_iou_shift_norm"]
    shifted_iou_cost = -np.log(np.clip(shifted_iou, similarity_epsilon, 1.0))
    shift_penalty_scale = (
        float(shifted_iou_shift_penalty_scale)
        if shifted_iou_shift_penalty_scale is not None
        else float(max(shifted_iou_radius, 1))
    )
    shifted_iou_shift_penalty_cost = shifted_iou_shift_norm / shift_penalty_scale
    if needs_shifted_cosine:
        shifted_cosine = shifted["shifted_mask_cosine_similarity"]
        shifted_cosine_cost = 1.0 - np.clip(shifted_cosine, 0.0, 1.0)
    else:
        shifted_cosine = np.zeros_like(shifted_iou, dtype=float)
        shifted_cosine_cost = np.zeros_like(shifted_iou, dtype=float)

    total_cost = np.asarray(base_cost, dtype=float).copy()
    if use_shifted_iou_for_iou_cost and iou_weight > 0.0:
        total_cost += iou_weight * shifted_iou_cost
    if shifted_iou_weight > 0.0:
        total_cost += shifted_iou_weight * shifted_iou_cost
    if use_shifted_mask_cosine_for_mask_cosine_cost and mask_cosine_weight > 0.0:
        total_cost += mask_cosine_weight * shifted_cosine_cost
    if shifted_mask_cosine_weight > 0.0:
        total_cost += shifted_mask_cosine_weight * shifted_cosine_cost
    if shifted_iou_shift_penalty_weight > 0.0:
        total_cost += shifted_iou_shift_penalty_weight * shifted_iou_shift_penalty_cost

    gated = np.asarray(
        components.get("gated", total_cost >= 0.5 * large_cost), dtype=bool
    )
    if gated.shape == total_cost.shape:
        total_cost = np.where(gated, large_cost, total_cost)
    total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)

    if not return_components:
        return total_cost

    exact_iou = components.get("iou")
    exact_cosine = components.get("mask_cosine_similarity")
    iou_for_cost = (
        shifted_iou
        if use_shifted_iou_for_iou_cost
        else np.asarray(
            exact_iou if exact_iou is not None else np.zeros_like(total_cost),
            dtype=float,
        )
    )
    mask_cosine_for_cost = (
        shifted_cosine
        if use_shifted_mask_cosine_for_mask_cosine_cost
        else np.asarray(
            exact_cosine if exact_cosine is not None else np.zeros_like(total_cost),
            dtype=float,
        )
    )
    components.update(
        {
            "pairwise_cost_matrix": total_cost,
            "shifted_iou": shifted_iou,
            "shifted_iou_cost": shifted_iou_cost,
            "shifted_iou_shift_y": shifted["shifted_iou_shift_y"],
            "shifted_iou_shift_x": shifted["shifted_iou_shift_x"],
            "shifted_iou_shift_norm": shifted_iou_shift_norm,
            "shifted_iou_shift_penalty_cost": shifted_iou_shift_penalty_cost,
            "shifted_mask_cosine_similarity": shifted_cosine,
            "shifted_mask_cosine_cost": shifted_cosine_cost,
            "shifted_iou_radius": np.full_like(
                total_cost, shifted_iou_radius, dtype=float
            ),
            "iou_for_cost": iou_for_cost,
            "mask_cosine_for_cost": mask_cosine_for_cost,
        }
    )
    return total_cost, components


def pairwise_shifted_overlap_matrices(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
    include_mask_cosine: bool = True,
    similarity_epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Return best local integer-shift IoU/cosine matrices for all ROI pairs."""

    radius = _nonnegative_int(radius, name="radius")
    if similarity_epsilon <= 0.0:
        raise ValueError("similarity_epsilon must be strictly positive")

    reference_array = np.asarray(reference_masks)
    measurement_array = np.asarray(measurement_masks)
    if reference_array.ndim != 3 or measurement_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    if reference_array.shape[1:] != measurement_array.shape[1:]:
        raise ValueError("Mask stacks must have matching spatial shapes")
    if not include_mask_cosine:
        return _pairwise_shifted_iou_from_support(
            reference_array,
            measurement_array,
            radius=radius,
        )

    cost_shape = (int(reference_array.shape[0]), int(measurement_array.shape[0]))
    best_iou: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_cosine: np.ndarray | None = (
        np.zeros(cost_shape, dtype=float) if include_mask_cosine else None
    )
    best_shift_y: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_shift_x: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_shift_norm: np.ndarray = np.zeros(cost_shape, dtype=float)

    for dy, dx in shift_offsets(radius):
        shifted_measurement = shift_mask_stack(measurement_array, dy=dy, dx=dx)
        iou = _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
            reference_array,
            shifted_measurement,
        )
        improved = iou > best_iou
        if np.any(improved):
            shift_norm = float(np.hypot(dy, dx))
            best_iou[improved] = iou[improved]
            best_shift_y[improved] = float(dy)
            best_shift_x[improved] = float(dx)
            best_shift_norm[improved] = shift_norm

        if include_mask_cosine:
            assert best_cosine is not None
            cosine = _bridge_impl._pairwise_mask_cosine_similarity(  # pylint: disable=protected-access
                reference_array,
                shifted_measurement,
                similarity_epsilon=similarity_epsilon,
            )
            np.maximum(best_cosine, cosine, out=best_cosine)

    result: dict[str, np.ndarray] = {
        "shifted_iou": best_iou,
        "shifted_iou_shift_y": best_shift_y,
        "shifted_iou_shift_x": best_shift_x,
        "shifted_iou_shift_norm": best_shift_norm,
    }
    if include_mask_cosine:
        assert best_cosine is not None
        result["shifted_mask_cosine_similarity"] = best_cosine
    return result


def _pairwise_shifted_iou_from_support(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
) -> dict[str, np.ndarray]:
    reference_support = _mask_support(reference_masks)
    measurement_support = _mask_support(measurement_masks)
    n_reference, height, width = reference_masks.shape
    n_measurement = int(measurement_masks.shape[0])
    cost_shape = (int(n_reference), n_measurement)
    best_iou: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_shift_y: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_shift_x: np.ndarray = np.zeros(cost_shape, dtype=float)
    best_shift_norm: np.ndarray = np.zeros(cost_shape, dtype=float)
    reference_order = np.argsort(reference_support.pixel, kind="stable")
    reference_pixel = reference_support.pixel[reference_order]
    reference_roi = reference_support.roi[reference_order]

    for dy, dx in shift_offsets(radius):
        shifted_y: np.ndarray = measurement_support.y + dy
        shifted_x: np.ndarray = measurement_support.x + dx
        valid = (
            (shifted_y >= 0)
            & (shifted_y < height)
            & (shifted_x >= 0)
            & (shifted_x < width)
        )
        if not np.any(valid):
            continue
        shifted_pixels = shifted_y[valid] * width + shifted_x[valid]
        shifted_rois = measurement_support.roi[valid]
        shifted_areas = np.bincount(
            shifted_rois,
            minlength=n_measurement,
        ).astype(float)
        intersections = _pairwise_binary_intersections_from_support(
            reference_pixel,
            reference_roi,
            shifted_pixels,
            shifted_rois,
            num_reference=int(n_reference),
            num_measurement=n_measurement,
            num_pixels=int(height * width),
            reference_is_sorted=True,
        )
        unions = (
            reference_support.areas[:, None] + shifted_areas[None, :] - intersections
        )
        iou: np.ndarray = np.zeros(cost_shape, dtype=float)
        valid_unions = unions > 0.0
        iou[valid_unions] = intersections[valid_unions] / unions[valid_unions]
        improved = iou > best_iou
        if np.any(improved):
            shift_norm = float(np.hypot(dy, dx))
            best_iou[improved] = iou[improved]
            best_shift_y[improved] = float(dy)
            best_shift_x[improved] = float(dx)
            best_shift_norm[improved] = shift_norm

    return {
        "shifted_iou": best_iou,
        "shifted_iou_shift_y": best_shift_y,
        "shifted_iou_shift_x": best_shift_x,
        "shifted_iou_shift_norm": best_shift_norm,
    }


@dataclass(frozen=True)
class _MaskSupport:
    roi: np.ndarray
    pixel: np.ndarray
    y: np.ndarray
    x: np.ndarray
    areas: np.ndarray


def _mask_support(masks: np.ndarray) -> _MaskSupport:
    mask_array = np.asarray(masks)
    n_rois, _, width = mask_array.shape
    flat_masks = mask_array.reshape(n_rois, -1)
    roi, pixel = np.nonzero(flat_masks > 0)
    roi = roi.astype(int, copy=False)
    pixel = pixel.astype(int, copy=False)
    y = (pixel // width).astype(int, copy=False)
    x = (pixel % width).astype(int, copy=False)
    areas = np.bincount(roi, minlength=int(n_rois)).astype(float)
    return _MaskSupport(roi=roi, pixel=pixel, y=y, x=x, areas=areas)


def _pairwise_binary_intersections_from_support(
    reference_pixel: np.ndarray,
    reference_roi: np.ndarray,
    measurement_pixel: np.ndarray,
    measurement_roi: np.ndarray,
    *,
    num_reference: int,
    num_measurement: int,
    num_pixels: int,
    reference_is_sorted: bool = False,
) -> np.ndarray:
    result: np.ndarray = np.zeros((num_reference, num_measurement), dtype=float)
    if reference_pixel.size == 0 or measurement_pixel.size == 0:
        return result

    unique_pixel_result = _bridge_impl._pairwise_unique_pixel_mask_dot(  # pylint: disable=protected-access
        reference_pixel,
        reference_roi,
        np.ones(reference_roi.shape[0], dtype=float),
        measurement_pixel,
        measurement_roi,
        np.ones(measurement_roi.shape[0], dtype=float),
        num_pixels=num_pixels,
        num_reference=num_reference,
        num_measurement=num_measurement,
    )
    if unique_pixel_result is not None:
        return unique_pixel_result

    if not reference_is_sorted:
        reference_order = np.argsort(reference_pixel, kind="stable")
        reference_pixel = reference_pixel[reference_order]
        reference_roi = reference_roi[reference_order]
    measurement_order = np.argsort(measurement_pixel, kind="stable")
    measurement_pixel = measurement_pixel[measurement_order]
    measurement_roi = measurement_roi[measurement_order]

    reference_index = 0
    measurement_index = 0
    while (
        reference_index < reference_pixel.size
        and measurement_index < measurement_pixel.size
    ):
        reference_current_pixel = reference_pixel[reference_index]
        measurement_current_pixel = measurement_pixel[measurement_index]
        if reference_current_pixel < measurement_current_pixel:
            reference_index = _advance_equal_values(reference_pixel, reference_index)
            continue
        if measurement_current_pixel < reference_current_pixel:
            measurement_index = _advance_equal_values(
                measurement_pixel, measurement_index
            )
            continue

        reference_stop = _advance_equal_values(reference_pixel, reference_index)
        measurement_stop = _advance_equal_values(measurement_pixel, measurement_index)
        result[
            np.ix_(
                reference_roi[reference_index:reference_stop],
                measurement_roi[measurement_index:measurement_stop],
            )
        ] += 1.0
        reference_index = reference_stop
        measurement_index = measurement_stop
    return result


def _advance_equal_values(values: np.ndarray, start_index: int) -> int:
    current_value = values[start_index]
    stop_index = start_index + 1
    while stop_index < values.size and values[stop_index] == current_value:
        stop_index += 1
    return stop_index


def shift_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Return offsets ordered so smaller shifts win exact IoU ties."""

    radius = _nonnegative_int(radius, name="radius")
    offsets = (
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    )
    return tuple(
        sorted(
            offsets,
            key=lambda offset: (
                offset[0] * offset[0] + offset[1] * offset[1],
                offset[0],
                offset[1],
            ),
        )
    )


def shift_mask_stack(masks: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    """Translate a stack of ROI masks with zero padding."""

    mask_array = np.asarray(masks)
    if mask_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    if dy == 0 and dx == 0:
        return mask_array

    _, height, width = mask_array.shape
    shifted = np.zeros_like(mask_array)
    if abs(dy) >= height or abs(dx) >= width:
        return shifted

    src_y = slice(max(0, -dy), min(height, height - dy))
    dst_y = slice(max(0, dy), min(height, height + dy))
    src_x = slice(max(0, -dx), min(width, width - dx))
    dst_x = slice(max(0, dx), min(width, width + dx))
    shifted[:, dst_y, dst_x] = mask_array[:, src_y, src_x]
    return shifted


def install_shifted_overlap_cost_patch() -> PairwiseCostMethod:
    """Install shifted-overlap kwargs support on ``CalciumPlaneData``.

    Returns the previous method so callers can restore it in a ``finally`` block.
    """

    original_method = CalciumPlaneData.build_pairwise_cost_matrix
    if getattr(original_method, "_bayescatrack_shifted_overlap_patch", False):
        return original_method

    def _patched_pairwise_cost(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        return shifted_iou_pairwise_cost_matrix(
            original_method,
            self,
            other,
            **kwargs,
        )

    setattr(_patched_pairwise_cost, "_bayescatrack_shifted_overlap_patch", True)
    setattr(_patched_pairwise_cost, "_bayescatrack_original", original_method)
    CalciumPlaneData.build_pairwise_cost_matrix = _patched_pairwise_cost  # type: ignore[method-assign]
    return original_method


def _ensure_finite_cost_matrix(
    cost_matrix: np.ndarray, *, large_cost: float
) -> np.ndarray:
    sanitized = np.asarray(cost_matrix, dtype=float).copy()
    invalid = ~np.isfinite(sanitized)
    if np.any(invalid):
        sanitized[invalid] = large_cost
    sanitized[sanitized < 0.0] = 0.0
    return sanitized


def _nonnegative_int(value: Any, *, name: str) -> int:
    try:
        integer_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer_value
