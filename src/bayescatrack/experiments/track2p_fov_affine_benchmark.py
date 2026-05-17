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
    if soft_iou_radius <= 0:
        return original_method(self, other, **kwargs)

    iou_weight = float(kwargs.get("iou_weight", 6.0))
    if iou_weight <= 0.0:
        return original_method(self, other, **kwargs)

    return_components = bool(kwargs.get("return_components", False))
    similarity_epsilon = float(kwargs.get("similarity_epsilon", 1.0e-6))
    large_cost = float(kwargs.get("large_cost", 1.0e6))

    base_kwargs = dict(kwargs)
    base_kwargs["iou_weight"] = 0.0
    base_kwargs["return_components"] = True
    base_cost, components = original_method(self, other, **base_kwargs)
    exact_iou = np.asarray(components["iou"], dtype=float)
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

    components = {
        **components,
        "pairwise_cost_matrix": total_cost,
        "soft_iou": soft_iou,
        "effective_iou": effective_iou,
        "iou_cost": iou_cost,
        "soft_iou_radius": np.full_like(total_cost, soft_iou_radius, dtype=float),
    }
    if return_components:
        return total_cost, components
    return total_cost


def _pairwise_dilated_iou_matrix(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    from bayescatrack.core import _bridge_impl  # pylint: disable=import-outside-toplevel,protected-access

    reference_dilated = _dilate_mask_stack(reference_masks, radius=radius)
    measurement_dilated = _dilate_mask_stack(measurement_masks, radius=radius)
    return _bridge_impl._pairwise_iou_matrix(reference_dilated, measurement_dilated)  # pylint: disable=protected-access


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
