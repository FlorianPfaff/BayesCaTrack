"""Track2p benchmark wrapper using BayesCaTrack's FOV-affine registration."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_benchmark import (
    _config_from_args,
    _csv_fieldnames,
    build_arg_parser,
    run_track2p_benchmark,
    write_results,
)
from bayescatrack.fov_affine_registration import register_measurement_plane_by_fov_affine


def _register_plane_pair_with_fov_affine(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str = "affine",
) -> CalciumPlaneData:
    if transform_type == "fov-affine":
        return register_measurement_plane_by_fov_affine(
            reference_plane,
            moving_plane,
        ).registered_measurement_plane
    from bayescatrack.track2p_registration import register_plane_pair

    return register_plane_pair(
        reference_plane,
        moving_plane,
        transform_type=transform_type,
    )


def _enable_fov_affine_choice(parser: Any) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == "transform_type":
            action.choices = ("affine", "rigid", "fov-affine", "fov-translation", "none")
            action.default = "fov-affine"
            action.help = "Registration transform; fov-affine is the default for this wrapper"
            return
    raise RuntimeError("Could not find --transform-type action")


def _add_soft_iou_options(parser: Any) -> None:
    parser.add_argument(
        "--soft-iou-radius",
        type=int,
        default=0,
        help=(
            "Optional dilation radius in pixels for soft-overlap IoU costs. "
            "When positive, exact IoU is still reported, but the IoU cost uses "
            "max(exact IoU, dilated IoU) so near-miss masks are not hard zero-overlap candidates."
        ),
    )


def _soft_iou_pairwise_cost_matrix(
    original_method: Any,
    self: CalciumPlaneData,
    other: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    soft_iou_radius = int(kwargs.pop("soft_iou_radius", 0) or 0)
    iou_weight = float(kwargs.get("iou_weight", 6.0))
    if iou_weight <= 0.0:
        return original_method(self, other, **kwargs)

    return_components = bool(kwargs.get("return_components", False))
    similarity_epsilon = float(kwargs.get("similarity_epsilon", 1.0e-6))
    large_cost = float(kwargs.get("large_cost", 1.0e6))
    if soft_iou_radius <= 0 and _is_iou_only_cost(kwargs):
        exact_iou = _pairwise_iou_matrix_sparse(self.roi_masks, other.roi_masks)
        iou_cost = -np.log(np.clip(exact_iou, similarity_epsilon, 1.0))
        total_cost = iou_weight * iou_cost
        total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)
        if return_components:
            components = _iou_only_components(total_cost, exact_iou, iou_cost)
            return total_cost, components
        return total_cost

    base_kwargs = dict(kwargs)
    base_kwargs["iou_weight"] = 0.0
    base_kwargs["return_components"] = return_components
    base_result = original_method(self, other, **base_kwargs)
    if return_components:
        base_cost, components = base_result
    else:
        base_cost = base_result
        components = {}
    exact_iou = _pairwise_iou_matrix_sparse(self.roi_masks, other.roi_masks)
    soft_iou = _pairwise_dilated_iou_matrix(
        self.roi_masks,
        other.roi_masks,
        radius=soft_iou_radius,
    )
    effective_iou = np.maximum(exact_iou, soft_iou)
    iou_cost = -np.log(np.clip(effective_iou, similarity_epsilon, 1.0))
    total_cost = np.asarray(base_cost, dtype=float) + iou_weight * iou_cost
    if "gated" in components:
        total_cost = np.where(np.asarray(components["gated"], dtype=bool), large_cost, total_cost)
    total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)

    if return_components:
        components = {
            **components,
            "pairwise_cost_matrix": total_cost,
            "iou": exact_iou,
            "soft_iou": soft_iou,
            "effective_iou": effective_iou,
            "iou_cost": iou_cost,
            "soft_iou_radius": np.full_like(total_cost, soft_iou_radius, dtype=float),
        }
        return total_cost, components
    return total_cost


def _is_iou_only_cost(kwargs: dict[str, Any]) -> bool:
    return (
        float(kwargs.get("centroid_weight", 1.0)) == 0.0
        and kwargs.get("max_centroid_distance") is None
        and float(kwargs.get("mask_cosine_weight", 2.0)) == 0.0
        and float(kwargs.get("area_weight", 0.5)) == 0.0
        and float(kwargs.get("roi_feature_weight", 0.25)) == 0.0
        and float(kwargs.get("cell_probability_weight", 0.0)) == 0.0
    )


def _iou_only_components(
    total_cost: np.ndarray,
    iou_matrix: np.ndarray,
    iou_cost: np.ndarray,
) -> dict[str, np.ndarray]:
    zero_cost = np.zeros_like(total_cost, dtype=float)
    return {
        "pairwise_cost_matrix": total_cost,
        "centroid_distance": zero_cost,
        "centroid_cost": zero_cost,
        "iou": iou_matrix,
        "iou_cost": iou_cost,
        "mask_cosine_similarity": zero_cost,
        "mask_cosine_cost": zero_cost,
        "area_ratio_cost": zero_cost,
        "roi_feature_cost": zero_cost,
        "cell_probability_cost": zero_cost,
        "gated": np.zeros_like(total_cost, dtype=bool),
    }


def _pairwise_dilated_iou_matrix(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    reference_dilated = _dilate_mask_stack(reference_masks, radius=radius)
    measurement_dilated = _dilate_mask_stack(measurement_masks, radius=radius)
    return _pairwise_iou_matrix_sparse(reference_dilated, measurement_dilated)


def _pairwise_iou_matrix_sparse(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
) -> np.ndarray:
    try:
        from scipy import sparse  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - SciPy is available in benchmark environments.
        from bayescatrack.core import _bridge_impl  # pylint: disable=import-outside-toplevel,protected-access

        return _bridge_impl._pairwise_iou_matrix(reference_masks, measurement_masks)  # pylint: disable=protected-access

    reference_array = np.asarray(reference_masks) > 0
    measurement_array = np.asarray(measurement_masks) > 0
    if reference_array.shape[1:] != measurement_array.shape[1:]:
        raise ValueError("Mask stacks must have matching spatial shapes")

    n_reference = int(reference_array.shape[0])
    n_measurement = int(measurement_array.shape[0])
    if n_reference == 0 or n_measurement == 0:
        return np.zeros((n_reference, n_measurement), dtype=float)

    reference_flat = reference_array.reshape(n_reference, -1)
    measurement_flat = measurement_array.reshape(n_measurement, -1)
    reference_roi, reference_pixel = np.nonzero(reference_flat)
    measurement_roi, measurement_pixel = np.nonzero(measurement_flat)
    reference_sparse = sparse.csr_matrix(
        (
            np.ones(reference_roi.shape[0], dtype=np.float32),
            (reference_roi, reference_pixel),
        ),
        shape=reference_flat.shape,
        dtype=np.float32,
    )
    measurement_sparse = sparse.csr_matrix(
        (
            np.ones(measurement_roi.shape[0], dtype=np.float32),
            (measurement_roi, measurement_pixel),
        ),
        shape=measurement_flat.shape,
        dtype=np.float32,
    )
    intersections = (reference_sparse @ measurement_sparse.T).toarray()
    reference_areas = np.asarray(reference_sparse.sum(axis=1)).ravel()
    measurement_areas = np.asarray(measurement_sparse.sum(axis=1)).ravel()
    unions = reference_areas[:, None] + measurement_areas[None, :] - intersections
    iou = np.zeros_like(intersections, dtype=float)
    valid = unions > 0.0
    iou[valid] = intersections[valid] / unions[valid]
    return iou


def _dilate_mask_stack(masks: np.ndarray, *, radius: int) -> np.ndarray:
    mask_array = np.asarray(masks) > 0
    if mask_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    if radius < 0:
        raise ValueError("soft_iou_radius must be non-negative")
    if radius == 0 or mask_array.shape[0] == 0:
        return mask_array

    result = np.array(mask_array, copy=True)
    _, height, width = mask_array.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx > radius * radius:
                continue
            src_y = slice(max(0, -dy), min(height, height - dy))
            dst_y = slice(max(0, dy), min(height, height + dy))
            src_x = slice(max(0, -dx), min(width, width - dx))
            dst_x = slice(max(0, dx), min(width, width + dx))
            result[:, dst_y, dst_x] |= mask_array[:, src_y, src_x]
    return result


def _ensure_finite_cost_matrix(cost_matrix: np.ndarray, *, large_cost: float) -> np.ndarray:
    sanitized = np.asarray(cost_matrix, dtype=float).copy()
    invalid = ~np.isfinite(sanitized)
    if np.any(invalid):
        sanitized[invalid] = large_cost
    sanitized[sanitized < 0.0] = 0.0
    return sanitized


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    from bayescatrack.experiments.track2p_benchmark import format_benchmark_table

    print(format_benchmark_table(rows))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-fov-affine"
    _enable_fov_affine_choice(parser)
    _add_soft_iou_options(parser)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    if args.soft_iou_radius > 0:
        pairwise_cost_kwargs = dict(config.pairwise_cost_kwargs or {})
        pairwise_cost_kwargs["soft_iou_radius"] = int(args.soft_iou_radius)
        config = replace(config, pairwise_cost_kwargs=pairwise_cost_kwargs)

    import bayescatrack.association.pyrecest_global_assignment as assignment

    original_register = assignment.register_plane_pair
    original_pairwise_cost = CalciumPlaneData.build_pairwise_cost_matrix

    def _patched_pairwise_cost(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        return _soft_iou_pairwise_cost_matrix(
            original_pairwise_cost,
            self,
            other,
            **kwargs,
        )

    assignment.register_plane_pair = _register_plane_pair_with_fov_affine
    CalciumPlaneData.build_pairwise_cost_matrix = _patched_pairwise_cost  # type: ignore[method-assign]
    try:
        results = run_track2p_benchmark(config)
    finally:
        assignment.register_plane_pair = original_register
        CalciumPlaneData.build_pairwise_cost_matrix = original_pairwise_cost  # type: ignore[method-assign]

    rows = [result.to_dict() for result in results]
    for row in rows:
        row["registration_wrapper"] = "fov-affine"
        row["soft_iou_radius"] = int(args.soft_iou_radius)
    if args.output is not None:
        write_results(rows, Path(args.output), args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
