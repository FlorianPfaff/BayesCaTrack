"""Growth-field analysis for longitudinal calcium-imaging tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.core.bridge import Track2pSession, load_track2p_subject
from bayescatrack.ground_truth_eval import TrackTable, load_track_table_csv

CenterMode = Literal["tracked-centroid", "loaded-roi-centroid", "fov-center"]
OutputFormat = Literal["csv", "json", "markdown", "table"]


@dataclass(frozen=True)
class RadialDisplacementRow:
    """One tracked cell displacement projected onto a radial axis."""

    track_index: int
    source_session: str
    target_session: str
    source_session_index: int
    target_session_index: int
    source_roi: int
    target_roi: int
    source_x: float
    source_y: float
    target_x: float
    target_y: float
    center_x: float
    center_y: float
    displacement_x: float
    displacement_y: float
    radial_unit_x: float
    radial_unit_y: float
    radial_displacement: float
    tangential_displacement: float
    displacement_norm: float
    radial_alignment: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a stable CSV/JSON row."""

        return {
            "track_index": int(self.track_index),
            "source_session": self.source_session,
            "target_session": self.target_session,
            "source_session_index": int(self.source_session_index),
            "target_session_index": int(self.target_session_index),
            "source_roi": int(self.source_roi),
            "target_roi": int(self.target_roi),
            "source_x": float(self.source_x),
            "source_y": float(self.source_y),
            "target_x": float(self.target_x),
            "target_y": float(self.target_y),
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "displacement_x": float(self.displacement_x),
            "displacement_y": float(self.displacement_y),
            "radial_unit_x": float(self.radial_unit_x),
            "radial_unit_y": float(self.radial_unit_y),
            "radial_displacement": float(self.radial_displacement),
            "tangential_displacement": float(self.tangential_displacement),
            "displacement_norm": float(self.displacement_norm),
            "radial_alignment": (
                None if self.radial_alignment is None else float(self.radial_alignment)
            ),
        }


@dataclass(frozen=True)
class RadialGrowthSummary:
    """Aggregate radial displacement for one target session."""

    source_session: str
    target_session: str
    source_session_index: int
    target_session_index: int
    n_tracks: int
    outward_tracks: int
    inward_tracks: int
    zero_radial_tracks: int
    outward_fraction: float
    mean_radial_displacement: float
    median_radial_displacement: float
    mean_displacement_norm: float
    mean_radial_alignment: float | None
    outward_sign_p_value: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a stable CSV/JSON row."""

        return {
            "source_session": self.source_session,
            "target_session": self.target_session,
            "source_session_index": int(self.source_session_index),
            "target_session_index": int(self.target_session_index),
            "n_tracks": int(self.n_tracks),
            "outward_tracks": int(self.outward_tracks),
            "inward_tracks": int(self.inward_tracks),
            "zero_radial_tracks": int(self.zero_radial_tracks),
            "outward_fraction": float(self.outward_fraction),
            "mean_radial_displacement": float(self.mean_radial_displacement),
            "median_radial_displacement": float(self.median_radial_displacement),
            "mean_displacement_norm": float(self.mean_displacement_norm),
            "mean_radial_alignment": (
                None
                if self.mean_radial_alignment is None
                else float(self.mean_radial_alignment)
            ),
            "outward_sign_p_value": (
                None
                if self.outward_sign_p_value is None
                else float(self.outward_sign_p_value)
            ),
        }


@dataclass(frozen=True)
class AffineGrowthSummary:
    """Least-squares affine growth model for one target session."""

    source_session: str
    target_session: str
    source_session_index: int
    target_session_index: int
    n_tracks: int
    rank: int
    matrix_xx: float
    matrix_xy: float
    matrix_yx: float
    matrix_yy: float
    translation_x: float
    translation_y: float
    determinant: float
    isotropic_scale: float
    singular_value_min: float
    singular_value_max: float
    residual_rmse: float
    fixed_point_x: float | None
    fixed_point_y: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a stable CSV/JSON row."""

        return {
            "source_session": self.source_session,
            "target_session": self.target_session,
            "source_session_index": int(self.source_session_index),
            "target_session_index": int(self.target_session_index),
            "n_tracks": int(self.n_tracks),
            "rank": int(self.rank),
            "matrix_xx": float(self.matrix_xx),
            "matrix_xy": float(self.matrix_xy),
            "matrix_yx": float(self.matrix_yx),
            "matrix_yy": float(self.matrix_yy),
            "translation_x": float(self.translation_x),
            "translation_y": float(self.translation_y),
            "determinant": float(self.determinant),
            "isotropic_scale": float(self.isotropic_scale),
            "singular_value_min": float(self.singular_value_min),
            "singular_value_max": float(self.singular_value_max),
            "residual_rmse": float(self.residual_rmse),
            "fixed_point_x": (
                None if self.fixed_point_x is None else float(self.fixed_point_x)
            ),
            "fixed_point_y": (
                None if self.fixed_point_y is None else float(self.fixed_point_y)
            ),
        }


def radial_displacement_rows(
    sessions: Sequence[Track2pSession],
    track_matrix: Any,
    *,
    source_session: int = 0,
    target_sessions: Sequence[int] | None = None,
    center: CenterMode | Sequence[float] = "tracked-centroid",
    order: str = "xy",
    weighted_centroids: bool = False,
    min_radius: float = 1.0e-9,
) -> list[RadialDisplacementRow]:
    """Project tracked-cell displacements onto radial directions from a center."""

    sessions = tuple(sessions)
    matrix = _normalize_track_matrix(track_matrix, n_sessions=len(sessions))
    source_session = _validate_session_index(source_session, len(sessions))
    target_sessions = _target_sessions(
        n_sessions=len(sessions),
        source_session=source_session,
        target_sessions=target_sessions,
    )
    centroid_lookups = _roi_centroid_lookups(
        sessions, order=order, weighted_centroids=weighted_centroids
    )

    rows: list[RadialDisplacementRow] = []
    for target_session in target_sessions:
        pairs = _matched_track_points(
            matrix,
            centroid_lookups,
            source_session=source_session,
            target_session=target_session,
        )
        if not pairs:
            continue
        center_xy = _resolve_center(
            center,
            sessions=sessions,
            source_session=source_session,
            source_points=np.vstack([pair[3] for pair in pairs]),
            source_lookup=centroid_lookups[source_session],
            order=order,
        )
        for track_index, source_roi, target_roi, source_xy, target_xy in pairs:
            radius_vector = source_xy - center_xy
            radius = float(np.linalg.norm(radius_vector))
            if radius <= min_radius:
                continue
            radial_unit = radius_vector / radius
            displacement = target_xy - source_xy
            radial_displacement = float(np.dot(displacement, radial_unit))
            tangential_displacement = float(
                radial_unit[0] * displacement[1] - radial_unit[1] * displacement[0]
            )
            displacement_norm = float(np.linalg.norm(displacement))
            radial_alignment = (
                None
                if displacement_norm <= 0.0
                else float(radial_displacement / displacement_norm)
            )
            rows.append(
                RadialDisplacementRow(
                    track_index=track_index,
                    source_session=sessions[source_session].session_name,
                    target_session=sessions[target_session].session_name,
                    source_session_index=source_session,
                    target_session_index=target_session,
                    source_roi=source_roi,
                    target_roi=target_roi,
                    source_x=float(source_xy[0]),
                    source_y=float(source_xy[1]),
                    target_x=float(target_xy[0]),
                    target_y=float(target_xy[1]),
                    center_x=float(center_xy[0]),
                    center_y=float(center_xy[1]),
                    displacement_x=float(displacement[0]),
                    displacement_y=float(displacement[1]),
                    radial_unit_x=float(radial_unit[0]),
                    radial_unit_y=float(radial_unit[1]),
                    radial_displacement=radial_displacement,
                    tangential_displacement=tangential_displacement,
                    displacement_norm=displacement_norm,
                    radial_alignment=radial_alignment,
                )
            )
    return rows


def radial_growth_summaries(
    rows: Sequence[RadialDisplacementRow],
) -> list[RadialGrowthSummary]:
    """Summarize radial displacement rows by target session."""

    grouped: dict[tuple[int, int, str, str], list[RadialDisplacementRow]] = {}
    for row in rows:
        key = (
            row.source_session_index,
            row.target_session_index,
            row.source_session,
            row.target_session,
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[RadialGrowthSummary] = []
    for (
        source_index,
        target_index,
        source_name,
        target_name,
    ), group_rows in sorted(grouped.items(), key=lambda item: item[0][:2]):
        radial = np.asarray(
            [row.radial_displacement for row in group_rows], dtype=float
        )
        norms = np.asarray([row.displacement_norm for row in group_rows], dtype=float)
        alignments = np.asarray(
            [
                row.radial_alignment
                for row in group_rows
                if row.radial_alignment is not None
            ],
            dtype=float,
        )
        outward = int(np.sum(radial > 0.0))
        inward = int(np.sum(radial < 0.0))
        zero = int(radial.size - outward - inward)
        nonzero = outward + inward
        summaries.append(
            RadialGrowthSummary(
                source_session=source_name,
                target_session=target_name,
                source_session_index=source_index,
                target_session_index=target_index,
                n_tracks=int(radial.size),
                outward_tracks=outward,
                inward_tracks=inward,
                zero_radial_tracks=zero,
                outward_fraction=_safe_ratio(outward, int(radial.size)),
                mean_radial_displacement=float(np.mean(radial)) if radial.size else 0.0,
                median_radial_displacement=(
                    float(np.median(radial)) if radial.size else 0.0
                ),
                mean_displacement_norm=float(np.mean(norms)) if norms.size else 0.0,
                mean_radial_alignment=(
                    None if alignments.size == 0 else float(np.mean(alignments))
                ),
                outward_sign_p_value=(
                    None if nonzero == 0 else _one_sided_binomial_tail(outward, nonzero)
                ),
            )
        )
    return summaries


def affine_growth_summaries(
    sessions: Sequence[Track2pSession],
    track_matrix: Any,
    *,
    source_session: int = 0,
    target_sessions: Sequence[int] | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
) -> list[AffineGrowthSummary]:
    """Fit least-squares affine maps from one source session to target sessions."""

    sessions = tuple(sessions)
    matrix = _normalize_track_matrix(track_matrix, n_sessions=len(sessions))
    source_session = _validate_session_index(source_session, len(sessions))
    target_sessions = _target_sessions(
        n_sessions=len(sessions),
        source_session=source_session,
        target_sessions=target_sessions,
    )
    centroid_lookups = _roi_centroid_lookups(
        sessions, order=order, weighted_centroids=weighted_centroids
    )

    summaries: list[AffineGrowthSummary] = []
    for target_session in target_sessions:
        pairs = _matched_track_points(
            matrix,
            centroid_lookups,
            source_session=source_session,
            target_session=target_session,
        )
        if len(pairs) < 3:
            continue
        source_points = np.vstack([pair[3] for pair in pairs])
        target_points = np.vstack([pair[4] for pair in pairs])
        summaries.append(
            _fit_affine_summary(
                source_points,
                target_points,
                source_session=sessions[source_session].session_name,
                target_session=sessions[target_session].session_name,
                source_session_index=source_session,
                target_session_index=target_session,
            )
        )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the growth-analysis CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack growth",
        description="Analyze global displacement and growth fields from longitudinal track tables.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_common_args(
        subparsers.add_parser(
            "radial",
            help="Project tracked-cell displacements onto radial directions",
        )
    )
    affine = subparsers.add_parser(
        "affine",
        help="Fit global affine expansion maps between sessions",
    )
    _add_subject_track_args(affine)
    _add_output_args(affine)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the growth-analysis CLI."""

    args = build_arg_parser().parse_args(argv)
    sessions, tracks = _load_cli_inputs(args)
    target_sessions = _parse_target_sessions(args.target_sessions)
    if args.command == "radial":
        center: CenterMode | Sequence[float] = args.center
        if args.center_x is not None or args.center_y is not None:
            if args.center_x is None or args.center_y is None:
                raise ValueError("--center-x and --center-y must be provided together")
            center = (float(args.center_x), float(args.center_y))
        rows = radial_displacement_rows(
            sessions,
            tracks.tracks,
            source_session=args.source_session,
            target_sessions=target_sessions,
            center=center,
            order=args.order,
            weighted_centroids=args.weighted_centroids,
        )
        radial_summaries = radial_growth_summaries(rows)
        if args.rows_output is not None:
            _write_dict_rows(
                [row.to_dict() for row in rows],
                args.rows_output,
                "csv",
            )
        _write_result(
            [summary.to_dict() for summary in radial_summaries],
            args.output,
            args.format,
            title="Radial Growth Summary",
        )
        return 0

    affine_summaries = affine_growth_summaries(
        sessions,
        tracks.tracks,
        source_session=args.source_session,
        target_sessions=target_sessions,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
    )
    _write_result(
        [summary.to_dict() for summary in affine_summaries],
        args.output,
        args.format,
        title="Affine Growth Summary",
    )
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    _add_subject_track_args(parser)
    parser.add_argument(
        "--center",
        choices=("tracked-centroid", "loaded-roi-centroid", "fov-center"),
        default="tracked-centroid",
        help="Center used for radial displacement directions",
    )
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-track radial displacement rows",
    )
    _add_output_args(parser)


def _add_subject_track_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subject", required=True, type=Path, help="Track2p-style subject directory"
    )
    parser.add_argument(
        "--tracks",
        required=True,
        type=Path,
        help="CSV track table with Suite2p ROI indices",
    )
    parser.add_argument(
        "--plane", dest="plane_name", default="plane0", help="Plane name such as plane0"
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format for loading sessions",
    )
    parser.add_argument("--source-session", type=int, default=0)
    parser.add_argument(
        "--target-sessions",
        default=None,
        help="Comma-separated target session indices; defaults to every non-source session",
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument(
        "--include-non-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output path"
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "table", "json", "csv"),
        default="markdown",
    )


def _load_cli_inputs(
    args: argparse.Namespace,
) -> tuple[list[Track2pSession], TrackTable]:
    sessions = load_track2p_subject(
        args.subject,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=False,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
    )
    if not sessions:
        raise ValueError(f"No sessions were loaded from {args.subject}")
    tracks = load_track_table_csv(
        args.tracks,
        session_names=tuple(session.session_name for session in sessions),
    )
    return sessions, tracks


def _parse_target_sessions(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _roi_centroid_lookups(
    sessions: Sequence[Track2pSession],
    *,
    order: str,
    weighted_centroids: bool,
) -> tuple[dict[int, np.ndarray], ...]:
    lookups: list[dict[int, np.ndarray]] = []
    for session in sessions:
        plane = session.plane_data
        centroids = plane.centroids(order=order, weighted=weighted_centroids).T
        roi_indices = (
            np.asarray(plane.roi_indices, dtype=int)
            if plane.roi_indices is not None
            else np.arange(plane.n_rois, dtype=int)
        )
        lookup: dict[int, np.ndarray] = {}
        for detection_index, roi_index in enumerate(roi_indices):
            roi_index = int(roi_index)
            if roi_index in lookup:
                raise ValueError(
                    f"Session {session.session_name!r} has duplicate ROI index {roi_index}"
                )
            lookup[roi_index] = np.asarray(centroids[detection_index], dtype=float)
        lookups.append(lookup)
    return tuple(lookups)


def _matched_track_points(
    matrix: np.ndarray,
    centroid_lookups: Sequence[Mapping[int, np.ndarray]],
    *,
    source_session: int,
    target_session: int,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    pairs: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    source_lookup = centroid_lookups[source_session]
    target_lookup = centroid_lookups[target_session]
    for track_index, row in enumerate(matrix):
        source_roi = _optional_roi(row[source_session])
        target_roi = _optional_roi(row[target_session])
        if source_roi is None or target_roi is None:
            continue
        if source_roi not in source_lookup or target_roi not in target_lookup:
            continue
        pairs.append(
            (
                int(track_index),
                int(source_roi),
                int(target_roi),
                np.asarray(source_lookup[source_roi], dtype=float),
                np.asarray(target_lookup[target_roi], dtype=float),
            )
        )
    return pairs


def _resolve_center(
    center: CenterMode | Sequence[float],
    *,
    sessions: Sequence[Track2pSession],
    source_session: int,
    source_points: np.ndarray,
    source_lookup: Mapping[int, np.ndarray],
    order: str,
) -> np.ndarray:
    if not isinstance(center, str):
        center_array = np.asarray(center, dtype=float)
        if center_array.shape != (2,):
            raise ValueError("Explicit center must contain exactly two coordinates")
        return center_array
    if center == "tracked-centroid":
        return np.mean(source_points, axis=0)
    if center == "loaded-roi-centroid":
        if not source_lookup:
            raise ValueError("Cannot compute loaded ROI centroid without ROIs")
        return np.mean(np.vstack(list(source_lookup.values())), axis=0)
    if center == "fov-center":
        height, width = sessions[source_session].plane_data.image_shape
        if order == "xy":
            return np.asarray([(width - 1) / 2.0, (height - 1) / 2.0], dtype=float)
        return np.asarray([(height - 1) / 2.0, (width - 1) / 2.0], dtype=float)
    raise ValueError(f"Unsupported center mode: {center!r}")


def _fit_affine_summary(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_session: str,
    target_session: str,
    source_session_index: int,
    target_session_index: int,
) -> AffineGrowthSummary:
    design = np.column_stack(
        [source_points[:, 0], source_points[:, 1], np.ones(source_points.shape[0])]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, target_points, rcond=None)
    predicted = design @ coefficients
    residual = target_points - predicted
    matrix = np.asarray(
        [
            [coefficients[0, 0], coefficients[1, 0]],
            [coefficients[0, 1], coefficients[1, 1]],
        ],
        dtype=float,
    )
    translation = np.asarray(coefficients[2, :], dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    determinant = float(np.linalg.det(matrix))
    fixed_point = _affine_fixed_point(matrix, translation)
    return AffineGrowthSummary(
        source_session=source_session,
        target_session=target_session,
        source_session_index=source_session_index,
        target_session_index=target_session_index,
        n_tracks=int(source_points.shape[0]),
        rank=int(rank),
        matrix_xx=float(matrix[0, 0]),
        matrix_xy=float(matrix[0, 1]),
        matrix_yx=float(matrix[1, 0]),
        matrix_yy=float(matrix[1, 1]),
        translation_x=float(translation[0]),
        translation_y=float(translation[1]),
        determinant=determinant,
        isotropic_scale=float(math.sqrt(abs(determinant))),
        singular_value_min=float(np.min(singular_values)),
        singular_value_max=float(np.max(singular_values)),
        residual_rmse=float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        fixed_point_x=None if fixed_point is None else float(fixed_point[0]),
        fixed_point_y=None if fixed_point is None else float(fixed_point[1]),
    )


def _affine_fixed_point(
    matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray | None:
    system = np.eye(2) - matrix
    if abs(float(np.linalg.det(system))) < 1.0e-12:
        return None
    return np.linalg.solve(system, translation)


def _target_sessions(
    *,
    n_sessions: int,
    source_session: int,
    target_sessions: Sequence[int] | None,
) -> tuple[int, ...]:
    if target_sessions is None:
        return tuple(index for index in range(n_sessions) if index != source_session)
    targets = tuple(
        _validate_session_index(index, n_sessions) for index in target_sessions
    )
    if source_session in targets:
        raise ValueError("target_sessions must not include source_session")
    return targets


def _validate_session_index(index: int, n_sessions: int) -> int:
    index = int(index)
    if index < 0 or index >= n_sessions:
        raise IndexError(
            f"session index {index} out of bounds for {n_sessions} sessions"
        )
    return index


def _normalize_track_matrix(track_matrix: Any, *, n_sessions: int) -> np.ndarray:
    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")
    if matrix.shape[1] != n_sessions:
        raise ValueError(
            f"track_matrix has {matrix.shape[1]} sessions, but {n_sessions} sessions were loaded"
        )
    return matrix


def _optional_roi(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        roi = int(value)
    elif isinstance(value, (float, np.floating)):
        if np.isnan(float(value)):
            return None
        roi = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            roi = int(text)
        except ValueError:
            return None
    else:
        return None
    if roi < 0:
        return None
    return roi


def _one_sided_binomial_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    successes = int(successes)
    trials = int(trials)
    if trials <= 200:
        numerator = sum(
            math.comb(trials, count) for count in range(successes, trials + 1)
        )
        return float(numerator / (2**trials))
    mean = trials / 2.0
    sd = math.sqrt(trials / 4.0)
    z = (successes - 0.5 - mean) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _write_result(
    rows: Sequence[Mapping[str, object]],
    output_path: Path | None,
    output_format: OutputFormat,
    *,
    title: str,
) -> None:
    if output_path is None:
        if output_format == "json":
            print(json.dumps(list(rows), indent=2))
        elif output_format == "csv":
            writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        else:
            print(_format_markdown(rows, title=title))
        return
    _write_dict_rows(rows, output_path, output_format, title=title)


def _write_dict_rows(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    output_format: OutputFormat,
    *,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(
        _format_markdown(rows, title="Growth Summary" if title is None else title)
        + "\n",
        encoding="utf-8",
    )


def _format_markdown(rows: Sequence[Mapping[str, object]], *, title: str) -> str:
    if not rows:
        return f"## {title}\n\nNo rows."
    fieldnames = _fieldnames(rows)
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(fieldname)) for fieldname in fieldnames)
            + " |"
        )
    return "\n".join(lines)


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    preferred = [
        "source_session",
        "target_session",
        "source_session_index",
        "target_session_index",
        "n_tracks",
        "outward_tracks",
        "inward_tracks",
        "zero_radial_tracks",
        "outward_fraction",
        "mean_radial_displacement",
        "median_radial_displacement",
        "mean_displacement_norm",
        "mean_radial_alignment",
        "outward_sign_p_value",
        "rank",
        "matrix_xx",
        "matrix_xy",
        "matrix_yx",
        "matrix_yy",
        "translation_x",
        "translation_y",
        "determinant",
        "isotropic_scale",
        "singular_value_min",
        "singular_value_max",
        "residual_rmse",
        "fixed_point_x",
        "fixed_point_y",
    ]
    row_keys = {str(key) for row in rows for key in row}
    return [key for key in preferred if key in row_keys] + sorted(
        row_keys - set(preferred)
    )


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
