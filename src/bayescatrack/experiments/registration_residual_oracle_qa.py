"""Registration residual and manual-GT oracle upper-bound diagnostics.

This module complements ``registration_qa_report`` and
``oracle_affine_registration_qa`` without changing their output schemas. It emits
per-manual-link signed residual vectors for raw, baseline-registered, and
manual-GT affine-oracle geometry, plus true-edge IoU rank/margin diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.core._bridge_impl import _pairwise_iou_matrix as _pairwise_iou_matrix  # pylint: disable=protected-access
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.association.pyrecest_global_assignment import session_edge_pairs
from bayescatrack.experiments.oracle_affine_registration_qa import (  # pylint: disable=protected-access
    _fit_affine_xy,
    _oracle_affine_registered_plane,
)
from bayescatrack.experiments.registration_qa_report import (  # pylint: disable=protected-access
    OutputFormat,
    RegistrationQAConfig,
    _benchmark_config,
    _csv_fieldnames,
    _format_value,
)
from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=protected-access
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair

ResidualOracleQALevel = Literal["summary", "links"]


@dataclass(frozen=True)
class RegistrationResidualOracleQAConfig:
    """Configuration for residual/oracle registration diagnostics."""

    registration: RegistrationQAConfig
    min_fit_links: int = 3
    require_full_rank: bool = True
    ridge: float = 0.0


def run_registration_residual_oracle_qa_report(
    config: RegistrationResidualOracleQAConfig,
) -> list[dict[str, Any]]:
    """Return one diagnostics row per manual-GT link and audited session edge."""

    if config.min_fit_links < 3:
        raise ValueError("min_fit_links must be at least 3")
    if config.ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    subject_dirs = discover_subject_dirs(config.registration.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.registration.data}"
        )

    benchmark_config = _benchmark_config(config.registration)
    rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        if config.registration.progress:
            print(
                f"registration-residual-oracle-qa: {subject_dir.name}",
                file=sys.stderr,
                flush=True,
            )
        rows.extend(_audit_subject(subject_dir, benchmark_config, config))
    return rows


def summarize_registration_residual_oracle_qa_links(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate link-level residual/oracle diagnostics by subject and edge."""

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["subject"]),
                str(row["source_session_name"]),
                str(row["target_session_name"]),
                int(row["session_gap"]),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (subject, source_name, target_name, session_gap), group in sorted(grouped.items()):
        summary.append(
            {
                "subject": subject,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "session_gap": session_gap,
                "n_gt_links": len(group),
                "baseline_transform_type": _mode(group, "baseline_transform_type"),
                "median_raw_iou": _stat(group, "raw_iou"),
                "median_baseline_iou": _stat(group, "baseline_iou"),
                "median_oracle_iou": _stat(group, "oracle_iou"),
                "baseline_iou_row_hit1_rate": _mean_bool(group, "baseline_iou_row_rank_is_1"),
                "oracle_iou_row_hit1_rate": _mean_bool(group, "oracle_iou_row_rank_is_1"),
                "baseline_iou_column_hit1_rate": _mean_bool(group, "baseline_iou_column_rank_is_1"),
                "oracle_iou_column_hit1_rate": _mean_bool(group, "oracle_iou_column_rank_is_1"),
                "baseline_iou_mutual_hit1_rate": _mean_bool(group, "baseline_iou_mutual_rank_is_1"),
                "oracle_iou_mutual_hit1_rate": _mean_bool(group, "oracle_iou_mutual_rank_is_1"),
                "median_baseline_iou_row_rank": _stat(group, "baseline_iou_row_rank"),
                "median_oracle_iou_row_rank": _stat(group, "oracle_iou_row_rank"),
                "median_baseline_iou_row_margin": _stat(group, "baseline_iou_row_margin"),
                "median_oracle_iou_row_margin": _stat(group, "oracle_iou_row_margin"),
                "median_baseline_residual_norm": _stat(group, "baseline_residual_norm"),
                "median_oracle_residual_norm": _stat(group, "oracle_residual_norm"),
                "median_abs_baseline_residual_x": _stat(group, "baseline_abs_residual_x"),
                "median_abs_baseline_residual_y": _stat(group, "baseline_abs_residual_y"),
                "median_abs_baseline_radial_residual": _stat(
                    group,
                    "baseline_abs_residual_radial_component",
                ),
                "median_abs_baseline_tangential_residual": _stat(
                    group,
                    "baseline_abs_residual_tangential_component",
                ),
                "oracle_fit_rms_residual": _stat(group, "oracle_fit_rms_residual"),
            }
        )
    return summary


def format_registration_residual_oracle_qa_table(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Format residual/oracle summary rows as a compact Markdown table."""

    columns = [
        "subject",
        "source_session_name",
        "target_session_name",
        "n_gt_links",
        "baseline_transform_type",
        "median_baseline_iou",
        "median_oracle_iou",
        "baseline_iou_row_hit1_rate",
        "oracle_iou_row_hit1_rate",
        "baseline_iou_mutual_hit1_rate",
        "oracle_iou_mutual_hit1_rate",
        "median_baseline_iou_row_rank",
        "median_oracle_iou_row_rank",
        "median_baseline_iou_row_margin",
        "median_oracle_iou_row_margin",
        "median_baseline_residual_norm",
        "median_abs_baseline_radial_residual",
        "median_abs_baseline_tangential_residual",
        "oracle_fit_rms_residual",
    ]
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def write_registration_residual_oracle_qa_results(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write residual/oracle diagnostics as JSON, CSV, or Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(
        format_registration_residual_oracle_qa_table(rows) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the standalone CLI parser for this diagnostic module."""

    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.registration_residual_oracle_qa",
        description=(
            "Report signed registration residuals and manual-GT affine-oracle IoU rank upper bounds."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=(
            "affine",
            "rigid",
            "fov-translation",
            "fov-affine",
            "bspline",
            "b-spline",
            "thin-plate-spline",
            "tps",
            "landmark-tps",
            "local-affine-grid",
            "optical-flow",
            "none",
        ),
    )
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--min-fit-links", type=int, default=3)
    parser.add_argument("--allow-rank-deficient-fit", action="store_true")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--level", default="summary", choices=("summary", "links"))
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", default="table", choices=("table", "json", "csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the residual/oracle diagnostic CLI."""

    args = build_arg_parser().parse_args(argv)
    rows: Sequence[Mapping[str, Any]] = run_registration_residual_oracle_qa_report(
        _config_from_args(args)
    )
    if args.level == "summary":
        rows = summarize_registration_residual_oracle_qa_links(rows)
    if args.output is not None:
        write_registration_residual_oracle_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(list(rows), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_registration_residual_oracle_qa_table(rows))
    return 0


def _config_from_args(args: argparse.Namespace) -> RegistrationResidualOracleQAConfig:
    return RegistrationResidualOracleQAConfig(
        registration=RegistrationQAConfig(
            data=args.data,
            reference=args.reference,
            reference_kind=cast(ReferenceKind, args.reference_kind),
            allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
            curated_only=args.curated_only,
            plane_name=args.plane_name,
            input_format=args.input_format,
            max_gap=args.max_gap,
            transform_type=args.transform_type,
            include_behavior=args.include_behavior,
            include_non_cells=args.include_non_cells,
            cell_probability_threshold=args.cell_probability_threshold,
            weighted_masks=args.weighted_masks,
            exclude_overlapping_pixels=args.exclude_overlapping_pixels,
            weighted_centroids=args.weighted_centroids,
            progress=args.progress,
        ),
        min_fit_links=args.min_fit_links,
        require_full_rank=not args.allow_rank_deficient_fit,
        ridge=args.ridge,
    )


def _audit_subject(
    subject_dir: Path,
    benchmark_config: Track2pBenchmarkConfig,
    config: RegistrationResidualOracleQAConfig,
) -> list[dict[str, Any]]:
    reference = _load_reference_for_subject(
        subject_dir,
        data_root=config.registration.data,
        config=benchmark_config,
    )
    _validate_reference_for_benchmark(
        reference,
        subject_dir=subject_dir,
        config=benchmark_config,
    )
    sessions = _load_subject_sessions(subject_dir, benchmark_config)
    _validate_reference_roi_indices(reference, sessions)
    reference_matrix = _reference_matrix(
        reference,
        curated_only=config.registration.curated_only,
    )

    rows: list[dict[str, Any]] = []
    for source_index, target_index in session_edge_pairs(
        len(sessions),
        max_gap=config.registration.max_gap,
    ):
        links = _manual_gt_links(
            sessions[source_index].plane_data,
            sessions[target_index].plane_data,
            reference_matrix,
            source_index,
            target_index,
            weighted_centroids=config.registration.weighted_centroids,
        )
        if len(links) < config.min_fit_links:
            continue
        rows.extend(
            _audit_edge(
                subject_dir.name,
                sessions[source_index].session_name,
                sessions[target_index].session_name,
                source_index,
                target_index,
                links,
                sessions[source_index].plane_data,
                sessions[target_index].plane_data,
                config,
            )
        )
    return rows


def _audit_edge(
    subject: str,
    source_name: str,
    target_name: str,
    source_index: int,
    target_index: int,
    links: Sequence[tuple[int, int, int, int, np.ndarray, np.ndarray]],
    source_plane: CalciumPlaneData,
    target_plane: CalciumPlaneData,
    config: RegistrationResidualOracleQAConfig,
) -> list[dict[str, Any]]:
    source_xy = np.vstack([link[4] for link in links])
    target_xy = np.vstack([link[5] for link in links])
    fit = _fit_affine_xy(
        source_xy,
        target_xy,
        ridge=config.ridge,
        require_full_rank=config.require_full_rank,
    )
    baseline_plane = register_plane_pair(
        source_plane,
        target_plane,
        transform_type=config.registration.transform_type,
    )
    oracle_plane = _oracle_affine_registered_plane(source_plane, target_plane, fit)

    source_locals = np.asarray([link[0] for link in links], dtype=int)
    target_locals = np.asarray([link[1] for link in links], dtype=int)
    source_rois = np.asarray([link[2] for link in links], dtype=int)
    target_rois = np.asarray([link[3] for link in links], dtype=int)

    raw_target_centroids = _plane_centroids_xy(
        target_plane,
        weighted=config.registration.weighted_centroids,
    )
    baseline_target_centroids = _mask_centroids_xy(baseline_plane.roi_masks)
    oracle_target_centroids = _mask_centroids_xy(oracle_plane.roi_masks)

    raw_iou_matrix = _maybe_pairwise_iou(source_plane.roi_masks, target_plane.roi_masks)
    baseline_iou_matrix = _pairwise_iou_matrix(source_plane.roi_masks, baseline_plane.roi_masks)
    oracle_iou_matrix = _pairwise_iou_matrix(source_plane.roi_masks, oracle_plane.roi_masks)

    rows: list[dict[str, Any]] = []
    for link_index, (source_local, target_local, source_roi, target_roi) in enumerate(
        zip(source_locals, target_locals, source_rois, target_rois, strict=False)
    ):
        source_centroid_xy = source_xy[link_index]
        raw_target_xy = raw_target_centroids[target_local]
        baseline_target_xy = baseline_target_centroids[target_local]
        oracle_target_xy = oracle_target_centroids[target_local]
        baseline_rank = _iou_rank_metrics_for_link(
            baseline_iou_matrix,
            int(source_local),
            int(target_local),
        )
        oracle_rank = _iou_rank_metrics_for_link(
            oracle_iou_matrix,
            int(source_local),
            int(target_local),
        )
        rows.append(
            {
                "subject": subject,
                "source_session_index": source_index,
                "target_session_index": target_index,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "session_gap": target_index - source_index,
                "baseline_transform_type": config.registration.transform_type,
                "link_index": link_index,
                "source_roi": int(source_roi),
                "target_roi": int(target_roi),
                "source_centroid_x": float(source_centroid_xy[0]),
                "source_centroid_y": float(source_centroid_xy[1]),
                "raw_target_centroid_x": float(raw_target_xy[0]),
                "raw_target_centroid_y": float(raw_target_xy[1]),
                "baseline_target_centroid_x": float(baseline_target_xy[0]),
                "baseline_target_centroid_y": float(baseline_target_xy[1]),
                "oracle_target_centroid_x": float(oracle_target_xy[0]),
                "oracle_target_centroid_y": float(oracle_target_xy[1]),
                "raw_iou": _matrix_value(raw_iou_matrix, source_local, target_local),
                "baseline_iou": float(baseline_iou_matrix[source_local, target_local]),
                "oracle_iou": float(oracle_iou_matrix[source_local, target_local]),
                **_centroid_residual_metrics(
                    "raw_",
                    source_centroid_xy,
                    raw_target_xy,
                    image_shape=source_plane.image_shape,
                ),
                **_centroid_residual_metrics(
                    "baseline_",
                    source_centroid_xy,
                    baseline_target_xy,
                    image_shape=source_plane.image_shape,
                ),
                **_centroid_residual_metrics(
                    "oracle_",
                    source_centroid_xy,
                    oracle_target_xy,
                    image_shape=source_plane.image_shape,
                ),
                **_prefix_metrics("baseline_", baseline_rank),
                **_prefix_metrics("oracle_", oracle_rank),
                "oracle_fit_n_links": int(fit.residual_norm.size),
                "oracle_fit_rank": int(fit.rank),
                "oracle_fit_condition": float(fit.condition),
                "oracle_fit_rms_residual": float(fit.rms_residual),
                "oracle_fit_median_residual": _finite_median(fit.residual_norm),
            }
        )
    return rows


def _manual_gt_links(
    source_plane: CalciumPlaneData,
    target_plane: CalciumPlaneData,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    *,
    weighted_centroids: bool,
) -> list[tuple[int, int, int, int, np.ndarray, np.ndarray]]:
    source_lookup = _roi_lookup(source_plane)
    target_lookup = _roi_lookup(target_plane)
    source_centroids = _plane_centroids_xy(source_plane, weighted=weighted_centroids)
    target_centroids = _plane_centroids_xy(target_plane, weighted=weighted_centroids)
    links: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if not (_is_present_roi(source_roi) and _is_present_roi(target_roi)):
            continue
        source_roi_int = int(cast(Any, source_roi))
        target_roi_int = int(cast(Any, target_roi))
        if source_roi_int not in source_lookup or target_roi_int not in target_lookup:
            continue
        source_local = source_lookup[source_roi_int]
        target_local = target_lookup[target_roi_int]
        links.append(
            (
                source_local,
                target_local,
                source_roi_int,
                target_roi_int,
                source_centroids[source_local],
                target_centroids[target_local],
            )
        )
    return links


def _roi_lookup(plane: CalciumPlaneData) -> dict[int, int]:
    indices = (
        np.arange(plane.n_rois, dtype=int)
        if plane.roi_indices is None
        else np.asarray(plane.roi_indices, dtype=int).reshape(-1)
    )
    return {int(roi_index): local_index for local_index, roi_index in enumerate(indices)}


def _plane_centroids_xy(plane: CalciumPlaneData, *, weighted: bool) -> np.ndarray:
    return plane.centroids(order="xy", weighted=weighted).T


def _mask_centroids_xy(masks: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(masks)
    centroids = np.full((mask_array.shape[0], 2), np.nan, dtype=float)
    for index, mask in enumerate(mask_array):
        yy, xx = np.nonzero(mask)
        if yy.size:
            centroids[index] = (float(np.mean(xx)), float(np.mean(yy)))
    return centroids


def _maybe_pairwise_iou(source_masks: np.ndarray, target_masks: np.ndarray) -> np.ndarray:
    if np.asarray(source_masks).shape[1:] != np.asarray(target_masks).shape[1:]:
        return np.full(
            (np.asarray(source_masks).shape[0], np.asarray(target_masks).shape[0]),
            np.nan,
            dtype=float,
        )
    return _pairwise_iou_matrix(source_masks, target_masks)


def _matrix_value(matrix: np.ndarray, row: int, column: int) -> float:
    value = np.asarray(matrix, dtype=float)[row, column]
    return float(value) if np.isfinite(value) else float("nan")


def _iou_rank_metrics_for_link(
    iou_matrix: np.ndarray,
    source_local: int,
    target_local: int,
) -> dict[str, float | bool | int]:
    true_iou = float(iou_matrix[source_local, target_local])
    row_scores = np.asarray(iou_matrix[source_local], dtype=float)
    column_scores = np.asarray(iou_matrix[:, target_local], dtype=float)
    row_rank = int(1 + np.count_nonzero(row_scores > true_iou))
    column_rank = int(1 + np.count_nonzero(column_scores > true_iou))
    row_margin = _true_score_margin(true_iou, np.delete(row_scores, target_local))
    column_margin = _true_score_margin(true_iou, np.delete(column_scores, source_local))
    return {
        "iou_row_rank": row_rank,
        "iou_column_rank": column_rank,
        "iou_row_rank_is_1": row_rank == 1,
        "iou_column_rank_is_1": column_rank == 1,
        "iou_mutual_rank_is_1": row_rank == 1 and column_rank == 1,
        "iou_row_margin": row_margin,
        "iou_column_margin": column_margin,
    }


def _true_score_margin(true_score: float, false_scores: np.ndarray) -> float:
    finite_false = np.asarray(false_scores, dtype=float)
    finite_false = finite_false[np.isfinite(finite_false)]
    if not finite_false.size:
        return float("nan")
    return float(true_score - np.max(finite_false))


def _prefix_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _centroid_residual_metrics(
    prefix: str,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    metric_names = (
        "residual_x",
        "residual_y",
        "residual_norm",
        "residual_angle_rad",
        "residual_radial_component",
        "residual_tangential_component",
        "abs_residual_x",
        "abs_residual_y",
        "abs_residual_radial_component",
        "abs_residual_tangential_component",
    )
    source_xy = np.asarray(source_xy, dtype=float).reshape(2)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    if not (np.all(np.isfinite(source_xy)) and np.all(np.isfinite(target_xy))):
        return {f"{prefix}{name}": np.nan for name in metric_names}

    residual_xy = target_xy - source_xy
    radial_xy = source_xy - np.asarray(
        [0.5 * (int(image_shape[1]) - 1), 0.5 * (int(image_shape[0]) - 1)],
        dtype=float,
    )
    radial_norm = float(np.linalg.norm(radial_xy))
    if radial_norm <= np.spacing(1.0):
        radial_component = np.nan
        tangential_component = np.nan
    else:
        radial_unit = radial_xy / radial_norm
        tangential_unit = np.asarray([-radial_unit[1], radial_unit[0]], dtype=float)
        radial_component = float(np.dot(residual_xy, radial_unit))
        tangential_component = float(np.dot(residual_xy, tangential_unit))

    return {
        f"{prefix}residual_x": float(residual_xy[0]),
        f"{prefix}residual_y": float(residual_xy[1]),
        f"{prefix}residual_norm": float(np.linalg.norm(residual_xy)),
        f"{prefix}residual_angle_rad": float(np.arctan2(residual_xy[1], residual_xy[0])),
        f"{prefix}residual_radial_component": radial_component,
        f"{prefix}residual_tangential_component": tangential_component,
        f"{prefix}abs_residual_x": float(abs(residual_xy[0])),
        f"{prefix}abs_residual_y": float(abs(residual_xy[1])),
        f"{prefix}abs_residual_radial_component": (
            float(abs(radial_component)) if np.isfinite(radial_component) else np.nan
        ),
        f"{prefix}abs_residual_tangential_component": (
            float(abs(tangential_component))
            if np.isfinite(tangential_component)
            else np.nan
        ),
    }


def _is_present_roi(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(cast(Any, value)) >= 0
    except (TypeError, ValueError):
        return False


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _stat(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    if not values.size:
        return np.nan
    return float(np.median(values))


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return np.nan
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def _mode(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    if not counts:
        return ""
    return max(counts, key=lambda value: counts[value])


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return float(np.median(finite))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
