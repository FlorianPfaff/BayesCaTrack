"""Manual-GT oracle affine registration geometry diagnostic."""

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

from bayescatrack.association.pyrecest_global_assignment import session_edge_pairs
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.registration_qa_report import (
    RegistrationQAConfig,
    _benchmark_config,
    _config_from_args as _registration_config_from_args,
    _csv_fieldnames,
    _format_value,
    _linked_source_rois,
    _roi_lookup,
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
class OracleAffineQAConfig:
    """Configuration for the manual-GT oracle affine diagnostic."""

    registration: RegistrationQAConfig
    min_fit_links: int = 3
    require_full_rank: bool = True
    ridge: float = 0.0


@dataclass(frozen=True)
class OracleAffineFit:
    """Moving-to-reference affine fit source_xy ~= A target_xy + b."""

    matrix_xy: np.ndarray
    residual_xy: np.ndarray
    residual_norm: np.ndarray
    rank: int
    condition: float

    @property
    def rms_residual(self) -> float:
        if not self.residual_norm.size:
            return float("nan")
        return float(np.sqrt(np.mean(self.residual_norm**2)))


@dataclass(frozen=True)
class ManualGTLink:
    """A present manual-GT link between two session-local ROIs."""

    track_index: int
    source_roi: int
    target_roi: int
    source_local: int
    target_local: int
    source_xy: np.ndarray
    target_xy: np.ndarray


def run_oracle_affine_qa_report(config: OracleAffineQAConfig) -> list[dict[str, Any]]:
    """Return per-link baseline-vs-oracle true-link geometry metrics."""

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
            print(f"oracle-affine-qa: {subject_dir.name}", file=sys.stderr, flush=True)
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.registration.data, config=benchmark_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=benchmark_config
        )
        sessions = _load_subject_sessions(subject_dir, benchmark_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference, curated_only=config.registration.curated_only
        )
        for source_index, target_index in session_edge_pairs(
            len(sessions), max_gap=config.registration.max_gap
        ):
            linked_source_rois = _linked_source_rois(
                reference_matrix, source_index, target_index
            )
            if not linked_source_rois:
                continue
            source_session = sessions[source_index]
            target_session = sessions[target_index]
            links = _manual_gt_links(
                source_session.plane_data,
                target_session.plane_data,
                reference_matrix,
                source_index,
                target_index,
            )
            if len(links) < config.min_fit_links:
                continue
            source_xy = np.vstack([link.source_xy for link in links])
            target_xy = np.vstack([link.target_xy for link in links])
            fit = _fit_affine_xy(
                source_xy,
                target_xy,
                ridge=config.ridge,
                require_full_rank=config.require_full_rank,
            )
            baseline = register_plane_pair(
                source_session.plane_data,
                target_session.plane_data,
                transform_type=config.registration.transform_type,
            )
            oracle = _oracle_affine_registered_plane(
                source_session.plane_data, target_session.plane_data, fit
            )
            rows.extend(
                _edge_rows(
                    subject_dir.name,
                    source_session.session_name,
                    target_session.session_name,
                    source_index,
                    target_index,
                    links,
                    source_session.plane_data,
                    target_session.plane_data,
                    baseline,
                    oracle,
                    fit,
                    config.registration.transform_type,
                )
            )
    return rows


def summarize_oracle_affine_qa(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate oracle-affine link rows by subject and session edge."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["subject"]),
                int(row["source_session_index"]),
                int(row["target_session_index"]),
            )
        ].append(row)
    return [_oracle_summary_row(group) for _, group in sorted(grouped.items())]


def format_oracle_affine_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if rows and "n_gt_links" not in rows[0]:
        return _format_oracle_affine_link_table(rows)

    columns = [
        "subject",
        "source_session_name",
        "target_session_name",
        "n_gt_links",
        "baseline_transform_type",
        "median_baseline_iou",
        "nonzero_baseline_iou_rate",
        "median_oracle_iou",
        "nonzero_oracle_iou_rate",
        "median_iou_gain",
        "median_baseline_centroid_distance",
        "median_oracle_centroid_distance",
        "median_residual_norm_gain",
        "p90_oracle_centroid_distance",
        "oracle_fit_rms_residual",
        "oracle_fit_median_residual",
        "oracle_affine_det",
        "oracle_affine_scale_1",
        "oracle_affine_scale_2",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_oracle_affine_qa_results(
    rows: Sequence[Mapping[str, Any]], output: Path, fmt: str
) -> None:
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
    output.write_text(format_oracle_affine_qa_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _registration_qa_arg_parser()
    parser.prog = "bayescatrack benchmark oracle-affine-qa"
    parser.description = (
        "Compare baseline registration to a manual-GT oracle affine warp."
    )
    _add_transform_choice(parser, "fov-affine")
    parser.add_argument("--min-fit-links", type=int, default=3)
    parser.add_argument("--allow-rank-deficient-fit", action="store_true")
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.level == "backend-audit":
        raise ValueError("oracle-affine-qa supports --level summary or --level links")
    config = OracleAffineQAConfig(
        registration=_registration_config_from_args(args),
        min_fit_links=args.min_fit_links,
        require_full_rank=not args.allow_rank_deficient_fit,
        ridge=args.ridge,
    )
    rows: Sequence[Mapping[str, Any]] = run_oracle_affine_qa_report(config)
    if args.level == "summary":
        rows = summarize_oracle_affine_qa(rows)
    if args.output is not None:
        write_oracle_affine_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(list(rows), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_oracle_affine_qa_table(rows))
    return 0


def _manual_gt_links(
    source_plane: CalciumPlaneData,
    target_plane: CalciumPlaneData,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
) -> list[ManualGTLink]:
    source_lookup = _roi_lookup(type("Session", (), {"plane_data": source_plane})())
    target_lookup = _roi_lookup(type("Session", (), {"plane_data": target_plane})())
    source_centroids = source_plane.centroids(order="xy").T
    target_centroids = target_plane.centroids(order="xy").T
    links: list[ManualGTLink] = []
    for track_index, track in enumerate(reference_matrix):
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_roi_int = int(source_roi)
        target_roi_int = int(target_roi)
        if source_roi_int not in source_lookup or target_roi_int not in target_lookup:
            continue
        source_local = source_lookup[source_roi_int]
        target_local = target_lookup[target_roi_int]
        links.append(
            ManualGTLink(
                track_index=track_index,
                source_roi=source_roi_int,
                target_roi=target_roi_int,
                source_local=source_local,
                target_local=target_local,
                source_xy=source_centroids[source_local],
                target_xy=target_centroids[target_local],
            )
        )
    return links


def _fit_affine_xy(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    ridge: float = 0.0,
    require_full_rank: bool = True,
) -> OracleAffineFit:
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if (
        source_xy.shape != target_xy.shape
        or source_xy.ndim != 2
        or source_xy.shape[1] != 2
    ):
        raise ValueError("source_xy and target_xy must both have shape (n, 2)")
    design = np.column_stack((target_xy, np.ones(target_xy.shape[0], dtype=float)))
    rank = int(np.linalg.matrix_rank(design))
    singular_values = np.linalg.svd(design, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else np.inf
    )
    if require_full_rank and rank < 3:
        raise ValueError("Manual-GT affine design matrix is rank deficient")
    if ridge > 0.0:
        coef = np.linalg.solve(
            design.T @ design + ridge * np.eye(3), design.T @ source_xy
        )
    else:
        coef, _, _, _ = np.linalg.lstsq(design, source_xy, rcond=None)
    residual = design @ coef - source_xy
    return OracleAffineFit(
        matrix_xy=coef.T,
        residual_xy=residual,
        residual_norm=np.linalg.norm(residual, axis=1),
        rank=rank,
        condition=condition,
    )


def _oracle_affine_registered_plane(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    fit: OracleAffineFit,
) -> CalciumPlaneData:
    masks = _warp_masks_by_affine_xy(
        np.asarray(moving_plane.roi_masks) > 0,
        fit.matrix_xy,
        reference_plane.image_shape,
    )
    fov = (
        reference_plane.fov
        if reference_plane.fov is not None
        else np.zeros(reference_plane.image_shape, dtype=float)
    )
    ops = {} if moving_plane.ops is None else dict(moving_plane.ops)
    ops.update(
        {
            "registration_backend": "manual-gt-oracle-affine",
            "registration_transform_type": "oracle-affine",
            "oracle_affine_matrix_xy": fit.matrix_xy.tolist(),
        }
    )
    return moving_plane.with_replaced_masks(
        masks, fov=fov, source=f"{moving_plane.source}_oracle_affine", ops=ops
    )


def _warp_masks_by_affine_xy(
    masks: np.ndarray, matrix_xy: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    masks = np.asarray(masks)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    if masks.ndim != 3 or matrix_xy.shape != (2, 3):
        raise ValueError("Expected masks (n,h,w) and affine matrix (2,3)")
    inverse_linear = np.linalg.inv(matrix_xy[:, :2])
    offset = matrix_xy[:, 2]
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    yy: np.ndarray
    xx: np.ndarray
    yy, xx = np.indices((out_h, out_w), dtype=float)
    source_xy = np.stack((xx.ravel(), yy.ravel()), axis=1)
    target_xy = (source_xy - offset[None, :]) @ inverse_linear.T
    tx = np.rint(target_xy[:, 0]).astype(int)
    ty = np.rint(target_xy[:, 1]).astype(int)
    valid = (tx >= 0) & (tx < masks.shape[2]) & (ty >= 0) & (ty < masks.shape[1])
    warped = np.zeros((masks.shape[0], out_h * out_w), dtype=masks.dtype)
    warped[:, valid] = masks[:, ty[valid], tx[valid]]
    return warped.reshape((masks.shape[0], out_h, out_w))


def _edge_rows(
    subject: str,
    source_name: str,
    target_name: str,
    source_index: int,
    target_index: int,
    links: Sequence[ManualGTLink],
    source_plane: CalciumPlaneData,
    raw_target_plane: CalciumPlaneData,
    baseline_plane: CalciumPlaneData,
    oracle_plane: CalciumPlaneData,
    fit: OracleAffineFit,
    baseline_transform_type: str,
) -> list[dict[str, Any]]:
    source_locals = np.asarray([link.source_local for link in links], dtype=int)
    target_locals = np.asarray([link.target_local for link in links], dtype=int)
    baseline_iou = _linked_iou(
        source_plane.roi_masks[source_locals], baseline_plane.roi_masks[target_locals]
    )
    oracle_iou = _linked_iou(
        source_plane.roi_masks[source_locals], oracle_plane.roi_masks[target_locals]
    )
    raw_iou = (
        _linked_iou(
            source_plane.roi_masks[source_locals],
            raw_target_plane.roi_masks[target_locals],
        )
        if source_plane.image_shape == raw_target_plane.image_shape
        else np.full(len(links), np.nan)
    )
    raw_centroids = _mask_centroids_xy(raw_target_plane.roi_masks)
    baseline_centroids = _mask_centroids_xy(baseline_plane.roi_masks)
    oracle_centroids = _mask_centroids_xy(oracle_plane.roi_masks)
    center_xy = _image_center_xy(source_plane.image_shape)
    linear = fit.matrix_xy[:, :2]
    scales = np.linalg.svd(linear, compute_uv=False)
    affine_metadata = {
        "oracle_fit_n_links": fit.residual_norm.size,
        "oracle_fit_rank": fit.rank,
        "oracle_fit_condition": fit.condition,
        "oracle_fit_rms_residual": fit.rms_residual,
        "oracle_fit_median_residual": _finite_median(fit.residual_norm),
        "oracle_affine_det": float(np.linalg.det(linear)),
        "oracle_affine_scale_1": float(scales[0]),
        "oracle_affine_scale_2": float(scales[-1]),
        "oracle_affine_tx": float(fit.matrix_xy[0, 2]),
        "oracle_affine_ty": float(fit.matrix_xy[1, 2]),
    }
    rows: list[dict[str, Any]] = []
    for index, link in enumerate(links):
        raw_metrics = _residual_metrics(
            link.source_xy, raw_centroids[link.target_local], center_xy
        )
        baseline_metrics = _residual_metrics(
            link.source_xy,
            baseline_centroids[link.target_local],
            center_xy,
        )
        oracle_metrics = _residual_metrics(
            link.source_xy,
            oracle_centroids[link.target_local],
            center_xy,
        )
        rows.append(
            {
                "subject": subject,
                "source_session_index": source_index,
                "target_session_index": target_index,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "session_gap": target_index - source_index,
                "track_index": link.track_index,
                "source_roi": link.source_roi,
                "target_roi": link.target_roi,
                "source_local": link.source_local,
                "target_local": link.target_local,
                "baseline_transform_type": baseline_transform_type,
                "source_x": float(link.source_xy[0]),
                "source_y": float(link.source_xy[1]),
                "raw_target_x": float(raw_centroids[link.target_local, 0]),
                "raw_target_y": float(raw_centroids[link.target_local, 1]),
                "baseline_target_x": float(baseline_centroids[link.target_local, 0]),
                "baseline_target_y": float(baseline_centroids[link.target_local, 1]),
                "oracle_target_x": float(oracle_centroids[link.target_local, 0]),
                "oracle_target_y": float(oracle_centroids[link.target_local, 1]),
                "raw_iou": float(raw_iou[index]),
                "baseline_iou": float(baseline_iou[index]),
                "oracle_iou": float(oracle_iou[index]),
                "iou_gain": float(oracle_iou[index] - baseline_iou[index]),
                **_prefixed_metrics("raw", raw_metrics),
                **_prefixed_metrics("baseline", baseline_metrics),
                **_prefixed_metrics("oracle", oracle_metrics),
                "residual_norm_gain": float(
                    baseline_metrics["norm"] - oracle_metrics["norm"]
                ),
                "fit_residual_x": float(fit.residual_xy[index, 0]),
                "fit_residual_y": float(fit.residual_xy[index, 1]),
                "fit_residual_norm": float(fit.residual_norm[index]),
                **affine_metadata,
            }
        )
    return rows


def _oracle_summary_row(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = group[0]
    return {
        "subject": first["subject"],
        "source_session_index": first["source_session_index"],
        "target_session_index": first["target_session_index"],
        "source_session_name": first["source_session_name"],
        "target_session_name": first["target_session_name"],
        "session_gap": first["session_gap"],
        "n_gt_links": len(group),
        "baseline_transform_type": first["baseline_transform_type"],
        "median_raw_iou": _row_stat(group, "raw_iou"),
        "median_baseline_iou": _row_stat(group, "baseline_iou"),
        "median_oracle_iou": _row_stat(group, "oracle_iou"),
        "median_iou_gain": _row_stat(group, "iou_gain"),
        "nonzero_baseline_iou_rate": _row_positive_rate(group, "baseline_iou"),
        "nonzero_oracle_iou_rate": _row_positive_rate(group, "oracle_iou"),
        "median_baseline_centroid_distance": _row_stat(
            group,
            "baseline_residual_norm",
        ),
        "median_oracle_centroid_distance": _row_stat(
            group,
            "oracle_residual_norm",
        ),
        "p90_oracle_centroid_distance": _row_stat(
            group,
            "oracle_residual_norm",
            90,
        ),
        "median_residual_norm_gain": _row_stat(group, "residual_norm_gain"),
        "median_oracle_residual_x": _row_stat(group, "oracle_residual_x"),
        "median_oracle_residual_y": _row_stat(group, "oracle_residual_y"),
        "median_oracle_residual_radial": _row_stat(
            group,
            "oracle_residual_radial",
        ),
        "median_oracle_residual_tangential": _row_stat(
            group,
            "oracle_residual_tangential",
        ),
        "oracle_fit_n_links": first["oracle_fit_n_links"],
        "oracle_fit_rank": first["oracle_fit_rank"],
        "oracle_fit_condition": first["oracle_fit_condition"],
        "oracle_fit_rms_residual": first["oracle_fit_rms_residual"],
        "oracle_fit_median_residual": first["oracle_fit_median_residual"],
        "oracle_affine_det": first["oracle_affine_det"],
        "oracle_affine_scale_1": first["oracle_affine_scale_1"],
        "oracle_affine_scale_2": first["oracle_affine_scale_2"],
        "oracle_affine_tx": first["oracle_affine_tx"],
        "oracle_affine_ty": first["oracle_affine_ty"],
    }


def _format_oracle_affine_link_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "subject",
        "source_session_name",
        "target_session_name",
        "track_index",
        "source_roi",
        "target_roi",
        "baseline_iou",
        "oracle_iou",
        "iou_gain",
        "baseline_residual_norm",
        "oracle_residual_norm",
        "residual_norm_gain",
        "oracle_residual_x",
        "oracle_residual_y",
        "fit_residual_norm",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def _residual_metrics(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    center_xy: np.ndarray,
) -> dict[str, float]:
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    residual = target_xy - source_xy
    norm = float(np.linalg.norm(residual))
    radial_axis = source_xy - center_xy
    radial_norm = float(np.linalg.norm(radial_axis))
    if radial_norm > np.spacing(1.0):
        radial_unit = radial_axis / radial_norm
    else:
        radial_unit = np.asarray((0.0, 0.0), dtype=float)
    tangent_unit = np.asarray((-radial_unit[1], radial_unit[0]), dtype=float)
    return {
        "x": float(residual[0]),
        "y": float(residual[1]),
        "norm": norm,
        "angle": float(np.arctan2(residual[1], residual[0])),
        "radial": float(np.dot(residual, radial_unit)),
        "tangential": float(np.dot(residual, tangent_unit)),
    }


def _prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_residual_{key}": float(value) for key, value in metrics.items()}


def _image_center_xy(image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0), dtype=float)


def _linked_iou(source_masks: np.ndarray, target_masks: np.ndarray) -> np.ndarray:
    source = np.asarray(source_masks) > 0
    target = np.asarray(target_masks) > 0
    intersection = np.sum(source & target, axis=(1, 2), dtype=float)
    union = np.sum(source | target, axis=(1, 2), dtype=float)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )


def _mask_centroids_xy(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    centroids = np.full((masks.shape[0], 2), np.nan, dtype=float)
    for index, mask in enumerate(masks):
        yy, xx = np.nonzero(mask)
        if yy.size:
            centroids[index] = (float(np.mean(xx)), float(np.mean(yy)))
    return centroids


def _row_stat(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    percentile: float | None = None,
) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    if percentile is None:
        return float(np.median(values))
    return float(np.percentile(values, percentile))


def _row_positive_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.mean(values > 0.0))


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return float(np.median(finite))


def _add_transform_choice(parser: argparse.ArgumentParser, value: str) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest != "transform_type" or action.choices is None:
            continue
        choices = tuple(action.choices)
        if value not in choices:
            action.choices = (*choices, value)
        return


if __name__ == "__main__":
    raise SystemExit(main())
