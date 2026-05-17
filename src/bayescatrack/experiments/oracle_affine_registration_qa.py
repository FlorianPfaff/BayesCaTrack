"""Manual-GT oracle affine registration geometry diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
)
from bayescatrack.experiments.registration_qa_report import (
    _config_from_args as _registration_config_from_args,
)
from bayescatrack.experiments.registration_qa_report import (
    _csv_fieldnames,
    _format_value,
    _linked_source_rois,
    _roi_lookup,
)
from bayescatrack.experiments.registration_qa_report import (
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
    residual_norm: np.ndarray
    rank: int
    condition: float

    @property
    def rms_residual(self) -> float:
        if not self.residual_norm.size:
            return float("nan")
        return float(np.sqrt(np.mean(self.residual_norm**2)))


def run_oracle_affine_qa_report(config: OracleAffineQAConfig) -> list[dict[str, Any]]:
    """Return per-edge baseline-vs-oracle true-link geometry metrics."""

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
            source_xy = np.vstack([link[2] for link in links])
            target_xy = np.vstack([link[3] for link in links])
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
            rows.append(
                _edge_row(
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
    """Return rows unchanged; kept for API symmetry with registration QA."""

    return [dict(row) for row in rows]


def format_oracle_affine_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
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
        "median_baseline_centroid_distance",
        "median_oracle_centroid_distance",
        "oracle_fit_rms_residual",
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
    rows = run_oracle_affine_qa_report(config)
    if args.output is not None:
        write_oracle_affine_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(rows, indent=2))
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
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    source_lookup = _roi_lookup(type("Session", (), {"plane_data": source_plane})())
    target_lookup = _roi_lookup(type("Session", (), {"plane_data": target_plane})())
    source_centroids = source_plane.centroids(order="xy").T
    target_centroids = target_plane.centroids(order="xy").T
    links: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_roi_int = int(source_roi)
        target_roi_int = int(target_roi)
        if source_roi_int not in source_lookup or target_roi_int not in target_lookup:
            continue
        links.append(
            (
                source_lookup[source_roi_int],
                target_lookup[target_roi_int],
                source_centroids[source_lookup[source_roi_int]],
                target_centroids[target_lookup[target_roi_int]],
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


def _edge_row(
    subject: str,
    source_name: str,
    target_name: str,
    source_index: int,
    target_index: int,
    links: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    source_plane: CalciumPlaneData,
    raw_target_plane: CalciumPlaneData,
    baseline_plane: CalciumPlaneData,
    oracle_plane: CalciumPlaneData,
    fit: OracleAffineFit,
    baseline_transform_type: str,
) -> dict[str, Any]:
    source_locals = np.asarray([link[0] for link in links], dtype=int)
    target_locals = np.asarray([link[1] for link in links], dtype=int)
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
    baseline_distance = _linked_centroid_distance(
        source_plane, baseline_plane, source_locals, target_locals
    )
    oracle_distance = _linked_centroid_distance(
        source_plane, oracle_plane, source_locals, target_locals
    )
    linear = fit.matrix_xy[:, :2]
    scales = np.linalg.svd(linear, compute_uv=False)
    return {
        "subject": subject,
        "source_session_index": source_index,
        "target_session_index": target_index,
        "source_session_name": source_name,
        "target_session_name": target_name,
        "session_gap": target_index - source_index,
        "n_gt_links": len(links),
        "baseline_transform_type": baseline_transform_type,
        "median_raw_iou": _finite_median(raw_iou),
        "median_baseline_iou": _finite_median(baseline_iou),
        "median_oracle_iou": _finite_median(oracle_iou),
        "nonzero_baseline_iou_rate": _positive_rate(baseline_iou),
        "nonzero_oracle_iou_rate": _positive_rate(oracle_iou),
        "median_baseline_centroid_distance": _finite_median(baseline_distance),
        "median_oracle_centroid_distance": _finite_median(oracle_distance),
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


def _linked_centroid_distance(
    source_plane: CalciumPlaneData,
    target_plane: CalciumPlaneData,
    source_locals: np.ndarray,
    target_locals: np.ndarray,
) -> np.ndarray:
    source_xy = _mask_centroids_xy(source_plane.roi_masks)[source_locals]
    target_xy = _mask_centroids_xy(target_plane.roi_masks)[target_locals]
    return np.linalg.norm(source_xy - target_xy, axis=1)


def _mask_centroids_xy(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    centroids = np.full((masks.shape[0], 2), np.nan, dtype=float)
    for index, mask in enumerate(masks):
        yy, xx = np.nonzero(mask)
        if yy.size:
            centroids[index] = (float(np.mean(xx)), float(np.mean(yy)))
    return centroids


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return float(np.median(finite))


def _positive_rate(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return float(np.mean(finite > 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
