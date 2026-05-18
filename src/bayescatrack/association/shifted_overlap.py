"""Local shift-search overlap costs for registered calcium-imaging ROIs.

The functions in this module implement the shifted-IoU diagnostic/cost proposed
for cases where a global registration leaves small local integer-pixel residuals.
Unlike dilation, every candidate benefits only from a coherent translation of the
whole measurement ROI, so shape selectivity is preserved in crowded fields.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from bayescatrack.core import _bridge_impl  # pylint: disable=protected-access
from bayescatrack.core.bridge import CalciumPlaneData

PairwiseCostMethod = Callable[..., np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]]


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
    """

    shifted_iou_radius = _nonnegative_int(
        kwargs.pop("shifted_iou_radius", 0), name="shifted_iou_radius"
    )
    shifted_iou_weight = float(kwargs.pop("shifted_iou_weight", 0.0) or 0.0)
    shifted_mask_cosine_weight = float(
        kwargs.pop("shifted_mask_cosine_weight", 0.0) or 0.0
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

    uses_shifted_overlap = (
        shifted_iou_radius > 0
        and (
            use_shifted_iou_for_iou_cost
            or shifted_iou_weight > 0.0
            or use_shifted_mask_cosine_for_mask_cosine_cost
            or shifted_mask_cosine_weight > 0.0
        )
    )
    if not uses_shifted_overlap:
        return original_method(self, other, **kwargs)

    return_components = bool(kwargs.get("return_components", False))
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

    shifted = pairwise_shifted_overlap_matrices(
        self.roi_masks,
        other.roi_masks,
        radius=shifted_iou_radius,
        similarity_epsilon=similarity_epsilon,
    )
    shifted_iou = shifted["shifted_iou"]
    shifted_iou_cost = -np.log(np.clip(shifted_iou, similarity_epsilon, 1.0))
    shifted_cosine = shifted["shifted_mask_cosine_similarity"]
    shifted_cosine_cost = 1.0 - np.clip(shifted_cosine, 0.0, 1.0)

    total_cost = np.asarray(base_cost, dtype=float).copy()
    if use_shifted_iou_for_iou_cost and iou_weight > 0.0:
        total_cost += iou_weight * shifted_iou_cost
    if shifted_iou_weight > 0.0:
        total_cost += shifted_iou_weight * shifted_iou_cost
    if use_shifted_mask_cosine_for_mask_cosine_cost and mask_cosine_weight > 0.0:
        total_cost += mask_cosine_weight * shifted_cosine_cost
    if shifted_mask_cosine_weight > 0.0:
        total_cost += shifted_mask_cosine_weight * shifted_cosine_cost

    gated = np.asarray(components.get("gated", total_cost >= 0.5 * large_cost), dtype=bool)
    if gated.shape == total_cost.shape:
        total_cost = np.where(gated, large_cost, total_cost)
    total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)

    if not return_components:
        return total_cost

    exact_iou = components.get("iou")
    exact_cosine = components.get("mask_cosine_similarity")
    components.update(
        {
            "pairwise_cost_matrix": total_cost,
            "shifted_iou": shifted_iou,
            "shifted_iou_cost": shifted_iou_cost,
            "shifted_iou_shift_y": shifted["shifted_iou_shift_y"],
            "shifted_iou_shift_x": shifted["shifted_iou_shift_x"],
            "shifted_iou_shift_norm": shifted["shifted_iou_shift_norm"],
            "shifted_mask_cosine_similarity": shifted_cosine,
            "shifted_mask_cosine_cost": shifted_cosine_cost,
            "shifted_iou_radius": np.full_like(total_cost, shifted_iou_radius, dtype=float),
            "iou_for_cost": shifted_iou if use_shifted_iou_for_iou_cost else exact_iou,
            "mask_cosine_for_cost": (
                shifted_cosine
                if use_shifted_mask_cosine_for_mask_cosine_cost
                else exact_cosine
            ),
        }
    )
    return total_cost, components


def pairwise_shifted_overlap_matrices(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
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

    cost_shape = (int(reference_array.shape[0]), int(measurement_array.shape[0]))
    best_iou = np.zeros(cost_shape, dtype=float)
    best_cosine = np.zeros(cost_shape, dtype=float)
    best_shift_y = np.zeros(cost_shape, dtype=float)
    best_shift_x = np.zeros(cost_shape, dtype=float)
    best_shift_norm = np.zeros(cost_shape, dtype=float)

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

        cosine = _bridge_impl._pairwise_mask_cosine_similarity(  # pylint: disable=protected-access
            reference_array,
            shifted_measurement,
            similarity_epsilon=similarity_epsilon,
        )
        np.maximum(best_cosine, cosine, out=best_cosine)

    return {
        "shifted_iou": best_iou,
        "shifted_mask_cosine_similarity": best_cosine,
        "shifted_iou_shift_y": best_shift_y,
        "shifted_iou_shift_x": best_shift_x,
        "shifted_iou_shift_norm": best_shift_norm,
    }


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

    Returns the original method so callers can restore it in a ``finally`` block.
    """

    original_method = CalciumPlaneData.build_pairwise_cost_matrix

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

    CalciumPlaneData.build_pairwise_cost_matrix = _patched_pairwise_cost  # type: ignore[method-assign]
    return original_method


def _ensure_finite_cost_matrix(cost_matrix: np.ndarray, *, large_cost: float) -> np.ndarray:
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
