"""Local shift-search ROI-overlap costs for near-miss registered masks.

This module extends :meth:`CalciumPlaneData.build_pairwise_cost_matrix` with
integer translation search around already registered measurement ROIs.  Unlike
mask dilation, shifted overlap only rewards a candidate when a coherent local
translation improves overlap of the entire ROI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
import bayescatrack.core._bridge_impl as _bridge_impl


def registered_shifted_iou_cost_kwargs(
    *,
    similarity_epsilon: float = 1.0e-6,
    shifted_iou_radius: int = 4,
    shifted_iou_weight: float = 1.0,
    shifted_mask_cosine_weight: float = 0.0,
    best_shift_norm_weight: float = 0.0,
) -> dict[str, float | int]:
    """Return kwargs for a registered local shift-search IoU ablation.

    The preset deliberately disables exact IoU, centroid, feature, area, and
    cell-probability terms.  It then uses the best IoU obtained by locally
    translating the registered measurement ROI over integer pixel shifts.
    """

    return {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "soft_iou_weight": 0.0,
        "distance_transform_overlap_weight": 0.0,
        "shifted_iou_weight": float(shifted_iou_weight),
        "shifted_iou_radius": int(shifted_iou_radius),
        "shifted_mask_cosine_weight": float(shifted_mask_cosine_weight),
        "best_shift_norm_weight": float(best_shift_norm_weight),
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": float(similarity_epsilon),
    }


def install_shifted_overlap_costs() -> None:
    """Install shifted-overlap cost extensions and cost presets."""

    _install_cost_matrix_patch()
    _install_global_assignment_preset()
    _install_registration_qa_preset()


def _install_cost_matrix_patch() -> None:
    original = CalciumPlaneData.build_pairwise_cost_matrix
    if getattr(original, "_bayescatrack_shifted_overlap_patch", False):
        return

    # pylint: disable=too-many-arguments,too-many-locals
    def _build_pairwise_cost_matrix_with_shifted_overlap(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        *args: Any,
        shifted_iou_weight: float = 0.0,
        shifted_iou_radius: int = 0,
        shifted_mask_cosine_weight: float = 0.0,
        best_shift_norm_weight: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        shifted_iou_weight = float(shifted_iou_weight)
        shifted_iou_radius = int(shifted_iou_radius)
        shifted_mask_cosine_weight = float(shifted_mask_cosine_weight)
        best_shift_norm_weight = float(best_shift_norm_weight)
        if shifted_iou_weight < 0.0:
            raise ValueError("shifted_iou_weight must be non-negative")
        if shifted_iou_radius < 0:
            raise ValueError("shifted_iou_radius must be non-negative")
        if shifted_mask_cosine_weight < 0.0:
            raise ValueError("shifted_mask_cosine_weight must be non-negative")
        if best_shift_norm_weight < 0.0:
            raise ValueError("best_shift_norm_weight must be non-negative")

        return_components = bool(kwargs.pop("return_components", False))
        needs_shift_components = return_components and shifted_iou_radius > 0
        needs_shift_cost = (
            shifted_iou_weight > 0.0
            or shifted_mask_cosine_weight > 0.0
            or best_shift_norm_weight > 0.0
        )
        if not needs_shift_components and not needs_shift_cost:
            return original(
                self,
                other,
                *args,
                return_components=return_components,
                **kwargs,
            )

        base_cost, components = original(
            self,
            other,
            *args,
            return_components=True,
            **kwargs,
        )
        total_cost = np.asarray(base_cost, dtype=float).copy()
        similarity_epsilon = float(kwargs.get("similarity_epsilon", 1.0e-6))
        if similarity_epsilon <= 0.0:
            raise ValueError("similarity_epsilon must be strictly positive")

        shifted_overlap = _pairwise_shifted_overlap_matrices(
            self.roi_masks,
            other.roi_masks,
            radius=shifted_iou_radius,
            include_iou=True,
            include_mask_cosine=shifted_mask_cosine_weight > 0.0 or return_components,
            similarity_epsilon=similarity_epsilon,
        )
        shifted_iou = shifted_overlap["shifted_iou"]
        shifted_iou_cost = -np.log(np.clip(shifted_iou, similarity_epsilon, 1.0))
        if shifted_iou_weight > 0.0:
            total_cost += shifted_iou_weight * shifted_iou_cost

        shifted_mask_cosine_similarity = shifted_overlap[
            "shifted_mask_cosine_similarity"
        ]
        shifted_mask_cosine_cost = 1.0 - np.clip(
            shifted_mask_cosine_similarity,
            0.0,
            1.0,
        )
        if shifted_mask_cosine_weight > 0.0:
            total_cost += shifted_mask_cosine_weight * shifted_mask_cosine_cost

        best_shift_norm = shifted_overlap["best_shift_norm"]
        shift_normalizer = max(float(shifted_iou_radius), 1.0)
        best_shift_norm_cost = np.where(
            shifted_iou > 0.0,
            best_shift_norm / shift_normalizer,
            1.0,
        )
        if best_shift_norm_weight > 0.0:
            total_cost += best_shift_norm_weight * best_shift_norm_cost

        large_cost = float(kwargs.get("large_cost", 1.0e6))
        total_cost = _bridge_impl._ensure_finite_cost_matrix(  # pylint: disable=protected-access
            total_cost,
            large_cost=large_cost,
        )
        if not return_components:
            return total_cost
        components = dict(components)
        components.update(
            {
                "pairwise_cost_matrix": total_cost,
                "shifted_iou": shifted_iou,
                "shifted_iou_cost": shifted_iou_cost,
                "shifted_mask_cosine_similarity": shifted_mask_cosine_similarity,
                "shifted_mask_cosine_cost": shifted_mask_cosine_cost,
                "best_shift_norm": best_shift_norm,
                "best_shift_norm_cost": best_shift_norm_cost,
                "shifted_iou_radius": np.full_like(
                    total_cost,
                    shifted_iou_radius,
                    dtype=float,
                ),
            }
        )
        return total_cost, components

    setattr(
        _build_pairwise_cost_matrix_with_shifted_overlap,
        "_bayescatrack_shifted_overlap_patch",
        True,
    )
    setattr(
        _build_pairwise_cost_matrix_with_shifted_overlap,
        "_bayescatrack_original",
        original,
    )
    CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        _build_pairwise_cost_matrix_with_shifted_overlap
    )


def _install_global_assignment_preset() -> None:
    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    original = global_assignment._cost_kwargs_for_method
    if getattr(original, "_bayescatrack_shifted_overlap_patch", False):
        global_assignment.registered_shifted_iou_cost_kwargs = (
            registered_shifted_iou_cost_kwargs
        )
        return

    def _cost_kwargs_for_method_with_shifted_overlap(cost: str) -> dict[str, Any]:
        if cost == "registered-shifted-iou":
            return dict(registered_shifted_iou_cost_kwargs())
        return original(cost)  # type: ignore[arg-type]

    setattr(
        _cost_kwargs_for_method_with_shifted_overlap,
        "_bayescatrack_shifted_overlap_patch",
        True,
    )
    setattr(
        _cost_kwargs_for_method_with_shifted_overlap,
        "_bayescatrack_original",
        original,
    )
    global_assignment._cost_kwargs_for_method = _cost_kwargs_for_method_with_shifted_overlap
    global_assignment.registered_shifted_iou_cost_kwargs = (
        registered_shifted_iou_cost_kwargs
    )


def _install_registration_qa_preset() -> None:
    try:
        from bayescatrack.experiments import registration_qa_report
    except ImportError:  # pragma: no cover - optional CLI import path
        return

    original = registration_qa_report._cost_kwargs
    if getattr(original, "_bayescatrack_shifted_overlap_patch", False):
        return

    def _registration_qa_cost_kwargs_with_shifted_overlap(config: Any) -> dict[str, Any]:
        if getattr(config, "cost", None) == "registered-shifted-iou":
            kwargs = dict(registered_shifted_iou_cost_kwargs())
            kwargs.update(getattr(config, "pairwise_cost_kwargs", None) or {})
            return kwargs
        return original(config)

    setattr(
        _registration_qa_cost_kwargs_with_shifted_overlap,
        "_bayescatrack_shifted_overlap_patch",
        True,
    )
    setattr(
        _registration_qa_cost_kwargs_with_shifted_overlap,
        "_bayescatrack_original",
        original,
    )
    registration_qa_report._cost_kwargs = _registration_qa_cost_kwargs_with_shifted_overlap


def _pairwise_shifted_overlap_matrices(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
    include_iou: bool,
    include_mask_cosine: bool,
    similarity_epsilon: float,
) -> dict[str, np.ndarray]:
    """Return best local-shift IoU/cosine matrices and selected shift norms."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if similarity_epsilon <= 0.0:
        raise ValueError("similarity_epsilon must be strictly positive")

    reference_array = np.asarray(reference_masks)
    measurement_array = np.asarray(measurement_masks)
    if reference_array.ndim != 3 or measurement_array.ndim != 3:
        raise ValueError("Mask stacks must have shape (n_roi, height, width)")
    if reference_array.shape[1:] != measurement_array.shape[1:]:
        raise ValueError("Mask stacks must have matching spatial shapes")

    cost_shape = (reference_array.shape[0], measurement_array.shape[0])
    shifted_iou = np.zeros(cost_shape, dtype=float)
    shifted_mask_cosine_similarity = np.zeros(cost_shape, dtype=float)
    best_shift_norm = np.zeros(cost_shape, dtype=float)
    if reference_array.shape[0] == 0 or measurement_array.shape[0] == 0:
        return {
            "shifted_iou": shifted_iou,
            "shifted_mask_cosine_similarity": shifted_mask_cosine_similarity,
            "best_shift_norm": best_shift_norm,
        }

    for offset_y, offset_x in _shift_offsets(radius):
        shifted_measurement = _translate_mask_stack(
            measurement_array,
            shift_y=offset_y,
            shift_x=offset_x,
        )
        if include_iou:
            current_iou = _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
                reference_array,
                shifted_measurement,
            )
            update = current_iou > shifted_iou + 1.0e-12
            if np.any(update):
                shifted_iou[update] = current_iou[update]
                best_shift_norm[update] = float(np.hypot(offset_y, offset_x))
        if include_mask_cosine:
            current_cosine = _bridge_impl._pairwise_mask_cosine_similarity(  # pylint: disable=protected-access
                reference_array,
                shifted_measurement,
                similarity_epsilon=similarity_epsilon,
            )
            shifted_mask_cosine_similarity = np.maximum(
                shifted_mask_cosine_similarity,
                np.clip(current_cosine, 0.0, 1.0),
            )

    return {
        "shifted_iou": shifted_iou,
        "shifted_mask_cosine_similarity": shifted_mask_cosine_similarity,
        "best_shift_norm": best_shift_norm,
    }


def _shift_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    offsets = (
        (offset_y, offset_x)
        for offset_y in range(-radius, radius + 1)
        for offset_x in range(-radius, radius + 1)
    )
    return tuple(
        sorted(
            offsets,
            key=lambda offset: (
                offset[0] ** 2 + offset[1] ** 2,
                offset[0],
                offset[1],
            ),
        )
    )


def _translate_mask_stack(masks: np.ndarray, *, shift_y: int, shift_x: int) -> np.ndarray:
    """Translate a mask stack without wrapping pixels around image boundaries."""

    mask_array = np.asarray(masks)
    if mask_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    shifted = np.zeros_like(mask_array)
    _, height, width = mask_array.shape
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return shifted

    src_y = slice(max(0, -shift_y), min(height, height - shift_y))
    dst_y = slice(max(0, shift_y), min(height, height + shift_y))
    src_x = slice(max(0, -shift_x), min(width, width - shift_x))
    dst_x = slice(max(0, shift_x), min(width, width + shift_x))
    shifted[:, dst_y, dst_x] = mask_array[:, src_y, src_x]
    return shifted


__all__ = [
    "install_shifted_overlap_costs",
    "registered_shifted_iou_cost_kwargs",
]
