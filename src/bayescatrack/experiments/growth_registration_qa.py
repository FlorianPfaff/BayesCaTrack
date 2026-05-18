"""Growth-aware spatial registration QA for Track2p-style benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
    roi_aware_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.core.bridge import Track2pSession, build_session_pair_association_bundle
from bayescatrack.experiments.oracle_affine_registration_qa import (
    ManualGTLink,
    OracleAffineFit,
    _fit_affine_xy,
    _linked_iou,
    _manual_gt_links,
    _mask_centroids_xy,
    _oracle_affine_registered_plane,
)
from bayescatrack.experiments.registration_qa_report import (
    _benchmark_config,
    _config_from_args as _registration_config_from_args,
    _csv_fieldnames,
    _format_value,
    build_arg_parser as _registration_qa_arg_parser,
)
from bayescatrack.experiments.track2p_benchmark import (
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair


@dataclass(frozen=True)
class GrowthRegistrationQAConfig:
    """Configuration for the growth-aware diagnostic."""

    registration: Any
    min_fit_links: int = 3
    require_full_rank: bool = True
    ridge: float = 0.0


def run_growth_registration_qa_report(config: GrowthRegistrationQAConfig) -> list[dict[str, Any]]:
    """Return one row per manual-GT link with spatial residual diagnostics."""

    if config.min_fit_links < 3:
        raise ValueError("min_fit_links must be at least 3")
    if config.registration.cost == "calibrated":
        raise ValueError("growth-registration-qa supports registered-iou and roi-aware costs only")

    subject_dirs = discover_subject_dirs(config.registration.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {config.registration.data}")

    benchmark_config = _benchmark_config(config.registration)
    rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        if config.registration.progress:
            print(f"growth-registration-qa: {subject_dir.name}", file=sys.stderr, flush=True)
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.registration.data, config=benchmark_config
        )
        _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=benchmark_config)
        sessions = _load_subject_sessions(subject_dir, benchmark_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference, curated_only=config.registration.curated_only
        )
        rows.extend(_audit_subject(subject_dir.name, sessions, reference_matrix, config))
    return rows


def summarize_growth_registration_qa(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate link rows by edge and source-image quadrant."""

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["cost"]),
                str(row["subject"]),
                str(row["source_session_name"]),
                str(row["target_session_name"]),
                str(row["source_quadrant"]),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (cost, subject, source_name, target_name, quadrant), group in sorted(grouped.items()):
        summary.append(
            {
                "cost": cost,
                "subject": subject,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "source_quadrant": quadrant,
                "n_gt_links": len(group),
                "transform_type": _mode(group, "transform_type"),
                "median_raw_iou": _stat(group, "raw_iou"),
                "median_registered_iou": _stat(group, "registered_iou"),
                "nonzero_registered_iou_rate": _mean_bool(group, "registered_iou_positive"),
                "median_raw_residual_norm": _stat(group, "raw_residual_norm"),
                "median_registered_residual_norm": _stat(group, "registered_residual_norm"),
                "p90_registered_residual_norm": _stat(group, "registered_residual_norm", 90),
                "median_gt_rank": _stat(group, "gt_rank"),
                "gt_admissible_rate": _mean_bool(group, "gt_candidate_admissible"),
                "empty_registered_mask_rate": _mean_bool(group, "target_empty_registered_mask"),
                "median_lower_left_distance_norm": _stat(group, "source_lower_left_distance_norm"),
                "oracle_affine_rmse": _stat(group, "oracle_affine_rmse"),
                "oracle_affine_scale_1": _stat(group, "oracle_affine_scale_1"),
                "oracle_affine_scale_2": _stat(group, "oracle_affine_scale_2"),
                "oracle_affine_shear_norm": _stat(group, "oracle_affine_shear_norm"),
                "affine_lower_left_to_upper_right_delta_norm": _stat(
                    group, "affine_lower_left_to_upper_right_delta_norm"
                ),
            }
        )
    return summary


def format_growth_registration_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format spatial summary rows as Markdown."""

    columns = [
        "cost",
        "subject",
        "source_session_name",
        "target_session_name",
        "source_quadrant",
        "n_gt_links",
        "transform_type",
        "median_registered_iou",
        "nonzero_registered_iou_rate",
        "median_registered_residual_norm",
        "p90_registered_residual_norm",
        "median_gt_rank",
        "gt_admissible_rate",
        "empty_registered_mask_rate",
        "oracle_affine_rmse",
        "affine_lower_left_to_upper_right_delta_norm",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_growth_registration_qa_results(
    rows: Sequence[Mapping[str, Any]], output: Path, fmt: str
) -> None:
    """Write rows as JSON, CSV, or table."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if fmt == "csv":
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output.write_text(format_growth_registration_qa_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Return CLI parser based on registration-qa options."""

    parser = _registration_qa_arg_parser()
    parser.prog = "bayescatrack benchmark growth-registration-qa"
    parser.description = (
        "Report spatially resolved growth/deformation registration QA on manual-GT links."
    )
    level_action = parser._option_string_actions["--level"]  # pylint: disable=protected-access
    level_action.choices = ("links", "spatial-summary")
    level_action.default = "spatial-summary"
    transform_action = parser._option_string_actions["--transform-type"]  # pylint: disable=protected-access
    transform_action.choices = ("affine", "rigid", "fov-translation", "none", "gt-affine-oracle")
    transform_action.default = "gt-affine-oracle"
    parser.add_argument("--min-fit-links", type=int, default=3)
    parser.add_argument("--allow-rank-deficient-fit", action="store_true")
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = GrowthRegistrationQAConfig(
        registration=_registration_config_from_args(args),
        min_fit_links=args.min_fit_links,
        require_full_rank=not args.allow_rank_deficient_fit,
        ridge=args.ridge,
    )
    rows: Sequence[Mapping[str, Any]] = run_growth_registration_qa_report(config)
    if args.level == "spatial-summary":
        rows = summarize_growth_registration_qa(rows)
    if args.output is not None:
        write_growth_registration_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(list(rows), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_growth_registration_qa_table(rows))
    return 0


def _audit_subject(
    subject: str,
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    config: GrowthRegistrationQAConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, target_index in session_edge_pairs(
        len(sessions), max_gap=config.registration.max_gap
    ):
        source_session = sessions[source_index]
        target_session = sessions[target_index]
        links = _manual_gt_links(
            source_session.plane_data,
            target_session.plane_data,
            reference_matrix,
            source_index,
            target_index,
        )
        if not links:
            continue
        fit: OracleAffineFit | None = None
        if config.registration.transform_type == "gt-affine-oracle":
            if len(links) < config.min_fit_links:
                continue
            fit = _fit_affine_xy(
                np.vstack([link.source_xy for link in links]),
                np.vstack([link.target_xy for link in links]),
                ridge=config.ridge,
                require_full_rank=config.require_full_rank,
            )
            registered_plane = _oracle_affine_registered_plane(
                source_session.plane_data, target_session.plane_data, fit
            )
        else:
            registered_plane = register_plane_pair(
                source_session.plane_data,
                target_session.plane_data,
                transform_type=config.registration.transform_type,
            )
        rows.extend(
            _edge_link_rows(
                subject,
                source_index,
                target_index,
                source_session,
                target_session,
                links,
                registered_plane,
                fit,
                config,
            )
        )
    return rows


def _edge_link_rows(
    subject: str,
    source_index: int,
    target_index: int,
    source_session: Track2pSession,
    target_session: Track2pSession,
    links: Sequence[ManualGTLink],
    registered_plane: Any,
    fit: OracleAffineFit | None,
    config: GrowthRegistrationQAConfig,
) -> list[dict[str, Any]]:
    source_locals = np.asarray([link.source_local for link in links], dtype=int)
    target_locals = np.asarray([link.target_local for link in links], dtype=int)
    source_xy = np.vstack([link.source_xy for link in links])
    raw_target_xy = np.vstack([link.target_xy for link in links])
    registered_xy_all = _mask_centroids_xy(registered_plane.roi_masks)
    registered_target_xy = registered_xy_all[target_locals]
    raw_iou = (
        _linked_iou(
            source_session.plane_data.roi_masks[source_locals],
            target_session.plane_data.roi_masks[target_locals],
        )
        if source_session.plane_data.image_shape == target_session.plane_data.image_shape
        else np.full(len(links), np.nan)
    )
    registered_iou = _linked_iou(
        source_session.plane_data.roi_masks[source_locals],
        registered_plane.roi_masks[target_locals],
    )
    cost_plane, empty_registered_rois = replace_empty_registered_masks(registered_plane)
    cost_matrix = _pairwise_cost_matrix(source_session, target_session, cost_plane, config)
    if empty_registered_rois.size:
        cost_matrix[:, empty_registered_rois] = float(_cost_kwargs(config).get("large_cost", 1.0e6))

    affine_values = _affine_summary(fit, source_session.plane_data.image_shape)
    rows: list[dict[str, Any]] = []
    for link_index, link in enumerate(links):
        source_local = link.source_local
        target_local = link.target_local
        cost_row = cost_matrix[source_local]
        gt_cost = float(cost_row[target_local])
        finite_costs = cost_row[np.isfinite(cost_row)]
        gt_rank = int(1 + np.count_nonzero(cost_row < gt_cost))
        below_threshold = True if config.registration.cost_threshold is None else bool(
            gt_cost <= float(config.registration.cost_threshold)
        )
        raw_residual = raw_target_xy[link_index] - source_xy[link_index]
        registered_residual = registered_target_xy[link_index] - source_xy[link_index]
        target_empty = bool(empty_registered_rois[target_local]) if empty_registered_rois.size else False
        rows.append(
            {
                "cost": config.registration.cost,
                "subject": subject,
                "source_session_index": source_index,
                "target_session_index": target_index,
                "source_session_name": source_session.session_name,
                "target_session_name": target_session.session_name,
                "session_gap": target_index - source_index,
                "transform_type": config.registration.transform_type,
                "source_local_index": int(source_local),
                "target_local_index": int(target_local),
                "source_x": float(source_xy[link_index, 0]),
                "source_y": float(source_xy[link_index, 1]),
                "source_quadrant": _quadrant(source_xy[link_index], source_session.plane_data.image_shape),
                "source_lower_left_distance_norm": _lower_left_distance_norm(
                    source_xy[link_index], source_session.plane_data.image_shape
                ),
                "raw_iou": float(raw_iou[link_index]),
                "registered_iou": float(registered_iou[link_index]),
                "registered_iou_positive": bool(registered_iou[link_index] > 0.0),
                "raw_residual_x": float(raw_residual[0]),
                "raw_residual_y": float(raw_residual[1]),
                "raw_residual_norm": float(np.linalg.norm(raw_residual)),
                "registered_residual_x": float(registered_residual[0]),
                "registered_residual_y": float(registered_residual[1]),
                "registered_residual_norm": float(np.linalg.norm(registered_residual)),
                "gt_cost": gt_cost,
                "gt_rank": gt_rank,
                "gt_cost_percentile": _cost_percentile(finite_costs, gt_cost),
                "target_empty_registered_mask": target_empty,
                "target_below_cost_threshold": below_threshold,
                "gt_candidate_admissible": (not target_empty) and below_threshold,
                **affine_values,
            }
        )
    return rows


def _pairwise_cost_matrix(
    source_session: Track2pSession,
    target_session: Track2pSession,
    registered_plane: Any,
    config: GrowthRegistrationQAConfig,
) -> np.ndarray:
    bundle = build_session_pair_association_bundle(
        source_session,
        target_session,
        measurement_plane_in_reference_frame=registered_plane,
        order=config.registration.order,
        weighted_centroids=config.registration.weighted_centroids,
        velocity_variance=config.registration.velocity_variance,
        regularization=config.registration.regularization,
        pairwise_cost_kwargs=_cost_kwargs(config),
        return_pairwise_components=False,
    )
    return np.asarray(bundle.pairwise_cost_matrix, dtype=float)


def _cost_kwargs(config: GrowthRegistrationQAConfig) -> dict[str, Any]:
    kwargs = (
        registered_iou_cost_kwargs()
        if config.registration.cost == "registered-iou"
        else roi_aware_cost_kwargs()
    )
    kwargs.update(config.registration.pairwise_cost_kwargs or {})
    return kwargs


def _quadrant(xy: np.ndarray, image_shape: tuple[int, int]) -> str:
    height, width = float(image_shape[0]), float(image_shape[1])
    vertical = "upper" if float(xy[1]) < height / 2.0 else "lower"
    horizontal = "left" if float(xy[0]) < width / 2.0 else "right"
    return f"{vertical}-{horizontal}"


def _lower_left_distance_norm(xy: np.ndarray, image_shape: tuple[int, int]) -> float:
    anchor = np.asarray([0.0, float(image_shape[0] - 1)], dtype=float)
    diagonal = float(np.hypot(max(image_shape[1] - 1, 1), max(image_shape[0] - 1, 1)))
    return float(np.linalg.norm(np.asarray(xy, dtype=float) - anchor) / diagonal)


def _affine_summary(fit: OracleAffineFit | None, image_shape: tuple[int, int]) -> dict[str, float | int]:
    if fit is None:
        return {
            "oracle_affine_n_links": 0,
            "oracle_affine_rmse": np.nan,
            "oracle_affine_scale_1": np.nan,
            "oracle_affine_scale_2": np.nan,
            "oracle_affine_shear_norm": np.nan,
            "affine_lower_left_to_upper_right_delta_norm": np.nan,
        }
    linear = np.asarray(fit.matrix_xy[:, :2], dtype=float)
    scales = np.linalg.svd(linear, compute_uv=False)
    shear_norm = float(np.linalg.norm(linear - np.diag(np.diag(linear))))
    height, width = int(image_shape[0]), int(image_shape[1])
    lower_left = np.asarray([0.0, float(height - 1)])
    upper_right = np.asarray([float(width - 1), 0.0])
    displacement_delta = _affine_displacement(fit.matrix_xy, upper_right) - _affine_displacement(
        fit.matrix_xy, lower_left
    )
    return {
        "oracle_affine_n_links": int(fit.residual_norm.size),
        "oracle_affine_rmse": fit.rms_residual,
        "oracle_affine_scale_1": float(scales[0]),
        "oracle_affine_scale_2": float(scales[-1]),
        "oracle_affine_shear_norm": shear_norm,
        "affine_lower_left_to_upper_right_delta_norm": float(np.linalg.norm(displacement_delta)),
    }


def _affine_displacement(matrix_xy: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    mapped = np.asarray(matrix_xy[:, :2]) @ point_xy + np.asarray(matrix_xy[:, 2])
    return mapped - point_xy


def _cost_percentile(finite_costs: np.ndarray, gt_cost: float) -> float:
    if not np.isfinite(gt_cost) or finite_costs.size <= 1:
        return float("nan")
    return float(100.0 * np.count_nonzero(finite_costs < gt_cost) / max(int(finite_costs.size) - 1, 1))


def _stat(group: Sequence[Mapping[str, Any]], key: str, percentile: float = 50.0) -> float:
    values = np.asarray([row.get(key, np.nan) for row in group], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    if percentile == 50.0:
        return float(np.median(values))
    return float(np.percentile(values, percentile))


def _mean_bool(group: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [row.get(key) for row in group if row.get(key) is not None]
    if not values:
        return float("nan")
    return float(np.mean([bool(value) for value in values]))


def _mode(group: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in group:
        counts[str(row.get(key, ""))] += 1
    if not counts:
        return ""
    return max(sorted(counts), key=lambda value: counts[value])


if __name__ == "__main__":
    raise SystemExit(main())
