"""Reproducible Track2p benchmark harness for BayesCaTrack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import CalibratedAssociationModel
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    solve_global_assignment_for_sessions,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import (
    Track2pSession,
    find_track2p_session_dirs,
    load_track2p_subject,
)
from bayescatrack.evaluation.track2p_metrics import (
    normalize_track_matrix,
    score_track_matrices,
)
from bayescatrack.ground_truth_eval import load_track2p_ground_truth_csv
from bayescatrack.matching import build_track_rows_from_matches
from bayescatrack.reference import (
    Track2pReference,
    load_aligned_subject_reference,
    load_track2p_reference,
)

ReferenceKind = Literal["auto", "manual-gt", "track2p-output", "aligned-subject-rows"]
BenchmarkMethod = Literal["track2p-baseline", "global-assignment", "oracle-gt-links"]
BenchmarkSplit = Literal["subject", "leave-one-subject-out"]
OutputFormat = Literal["table", "json", "csv"]
GROUND_TRUTH_CSV_NAME = "ground_truth.csv"
GROUND_TRUTH_REFERENCE_SOURCE = "ground_truth_csv"
ALIGNED_REFERENCE_SOURCE = "aligned_subject_rows"
TRACK2P_REFERENCE_SOURCES = frozenset(
    {"track2p_output_suite2p_indices", "track2p_output_match_mat"}
)


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Track2pBenchmarkConfig:
    """Configuration for one Track2p benchmark run."""

    data: Path
    method: BenchmarkMethod
    split: BenchmarkSplit = "subject"
    plane_name: str = "plane0"
    input_format: str = "auto"
    reference: Path | None = None
    reference_kind: ReferenceKind = "auto"
    allow_track2p_as_reference_for_smoke_test: bool = False
    curated_only: bool = False
    seed_session: int = 0
    restrict_to_reference_seed_rois: bool = True
    cost: AssociationCost = "registered-iou"
    max_gap: int = 2
    transform_type: str = "affine"
    start_cost: float = 5.0
    end_cost: float = 5.0
    gap_penalty: float = 1.0
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    progress: bool = False


@dataclass(frozen=True)
class SubjectBenchmarkResult:
    """One subject-level benchmark result."""

    subject: str
    variant: str
    method: BenchmarkMethod
    scores: Mapping[str, float | int | str]
    n_sessions: int
    reference_source: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "subject": self.subject,
            "variant": self.variant,
            "method": self.method,
            "n_sessions": self.n_sessions,
            "reference_source": self.reference_source,
            **dict(self.scores),
        }


class ProgressReporter:
    """Small stderr progress reporter that keeps machine-readable stdout clean."""

    def __init__(self, total: int, *, enabled: bool, label: str) -> None:
        self.total = max(int(total), 1)
        self.enabled = bool(enabled)
        self.label = label
        self.current = 0

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        self.current = min(self.current + 1, self.total)
        filled = int(round(20 * self.current / self.total))
        bar = "#" * filled + "-" * (20 - filled)
        percent = 100.0 * self.current / self.total
        print(
            f"{self.label} [{bar}] {self.current}/{self.total} ({percent:5.1f}%) {message}",
            file=sys.stderr,
            flush=True,
        )


def run_track2p_benchmark(
    config: Track2pBenchmarkConfig,
) -> list[SubjectBenchmarkResult]:
    """Run a Track2p benchmark over one subject directory or a dataset root."""

    if config.split == "leave-one-subject-out":
        if config.method != "global-assignment" or config.cost != "calibrated":
            raise ValueError(
                "LOSO calibration requires method='global-assignment' and cost='calibrated'"
            )
        from bayescatrack.experiments.track2p_loso_calibration import (
            run_track2p_loso_calibration,
        )

        return run_track2p_loso_calibration(config).to_benchmark_results()
    if config.cost == "calibrated":
        raise ValueError("cost='calibrated' requires split='leave-one-subject-out'")

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="benchmark"
    )
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
            _validate_reference_roi_indices(
                reference, _load_subject_sessions(subject_dir, config)
            )
        predicted_matrix, variant = _predict_subject_tracks(
            subject_dir, config, reference=reference
        )
        scores = _score_prediction_against_reference(
            predicted_matrix, reference, config=config
        )
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=variant,
                method=config.method,
                scores=scores,
                n_sessions=reference.n_sessions,
                reference_source=reference.source,
            )
        )
    return results


def discover_subject_dirs(data_path: str | Path) -> list[Path]:
    """Find Track2p subject directories beneath ``data_path``."""

    root = Path(data_path)
    if _looks_like_subject_dir(root):
        return [root]
    subjects = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and _looks_like_subject_dir(child)
    ]
    return subjects


def format_benchmark_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Format benchmark rows as the first paper-facing Markdown table."""

    columns = [
        "variant",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        values = [_format_table_value(row.get(column, "")) for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def write_results(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write benchmark rows as JSON, CSV, or Markdown table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        fieldnames = _csv_fieldnames(rows)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_benchmark_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p",
        description="Run Track2p baseline and global-assignment ablations on Track2p-style datasets.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("track2p-baseline", "global-assignment", "oracle-gt-links"),
        help="Benchmark variant to run",
    )
    parser.add_argument(
        "--split",
        default="subject",
        choices=("subject", "leave-one-subject-out"),
        help="Evaluation split policy",
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
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground_truth.csv file, ground-truth root, subject directory, or track2p folder",
    )
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        help="Declared reference type; manual-gt is required for paper-facing runs",
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test",
        action="store_true",
        help="Permit Track2p/aligned-row references for plumbing smoke tests only",
    )
    parser.add_argument(
        "--curated-only",
        action="store_true",
        help="Evaluate only reference tracks marked curated",
    )
    parser.add_argument(
        "--seed-session",
        type=int,
        default=0,
        help="Reference seed session used for sparse-GT filtering",
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score only predicted tracks whose seed-session ROI is in the reference seed set",
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware", "calibrated"),
        help="Pairwise cost used by global assignment",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum forward session gap for global-assignment edges",
    )
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
        help="Track2p registration transform type",
    )
    parser.add_argument(
        "--start-cost", type=float, default=5.0, help="PyRecEst track start cost"
    )
    parser.add_argument(
        "--end-cost", type=float, default=5.0, help="PyRecEst track end cost"
    )
    parser.add_argument(
        "--gap-penalty", type=float, default=1.0, help="Penalty per skipped session"
    )
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=6.0,
        help="Maximum adjusted edge cost admitted by the solver",
    )
    parser.add_argument(
        "--no-cost-threshold",
        action="store_true",
        help="Disable the solver edge-cost threshold",
    )
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load behaviour arrays when present",
    )
    parser.add_argument(
        "--include-non-cells",
        action="store_true",
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=0.5,
        help="Suite2p iscell probability threshold",
    )
    parser.add_argument(
        "--weighted-masks",
        action="store_true",
        help="Use Suite2p lam weights while reconstructing masks",
    )
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )
    parser.add_argument(
        "--order", default="xy", choices=("xy", "yx"), help="Coordinate order for costs"
    )
    parser.add_argument(
        "--weighted-centroids",
        action="store_true",
        help="Use weighted centroids where masks contain weights",
    )
    parser.add_argument(
        "--velocity-variance",
        type=float,
        default=25.0,
        help="Velocity variance for association bundle state moments",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
        help="Position covariance regularization",
    )
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print benchmark progress to stderr",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output file"
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Stdout/output format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    results = run_track2p_benchmark(config)
    rows = [result.to_dict() for result in results]

    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _predict_subject_tracks(
    subject_dir: Path,
    config: Track2pBenchmarkConfig,
    *,
    reference: Track2pReference | None = None,
) -> tuple[np.ndarray, str]:
    if config.method == "track2p-baseline":
        track2p_dir = subject_dir / "track2p"
        if track2p_dir.exists():
            baseline = load_track2p_reference(track2p_dir, plane_name=config.plane_name)
        else:
            baseline = load_aligned_subject_reference(
                subject_dir,
                plane_name=config.plane_name,
                input_format=config.input_format,
                include_behavior=config.include_behavior,
                include_non_cells=config.include_non_cells,
                cell_probability_threshold=config.cell_probability_threshold,
                weighted_masks=config.weighted_masks,
                exclude_overlapping_pixels=config.exclude_overlapping_pixels,
            )
        return normalize_track_matrix(baseline.suite2p_indices), "Track2p default"

    if config.method == "oracle-gt-links":
        if reference is None:
            raise ValueError("oracle-gt-links requires a loaded reference")
        return (
            oracle_ground_truth_link_tracks(
                reference,
                curated_only=config.curated_only,
                seed_session=config.seed_session,
            ),
            "Oracle GT consecutive links",
        )

    if config.method != "global-assignment":
        raise ValueError(f"Unsupported benchmark method: {config.method!r}")

    sessions = _load_subject_sessions(subject_dir, config)
    assignment = solve_configured_global_assignment(sessions, config)
    predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    return predicted, _variant_name(config.cost)


def oracle_ground_truth_link_tracks(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_session: int = 0,
) -> np.ndarray:
    """Build an oracle prediction by stitching consecutive GT pairwise links.

    This diagnostic variant does not copy the reference track matrix directly.
    It converts each consecutive session pair into explicit GT ROI links and
    then uses the normal BayesCaTrack row-stitching helper. If complete-track F1
    is poor for this oracle, the failure is in track-row assembly, ROI indexing,
    or scoring rather than in registration or association costs.
    """

    reference_matrix = _reference_matrix(reference, curated_only=curated_only)
    start_roi_indices = sorted(
        _reference_seed_roi_set(reference_matrix, seed_session=seed_session)
    )

    if reference.n_sessions == 1:
        return np.asarray(start_roi_indices, dtype=int).reshape(-1, 1)

    consecutive_matches = [
        reference.pairwise_matches(
            session_index,
            session_index + 1,
            curated_only=curated_only,
        )
        for session_index in range(reference.n_sessions - 1)
    ]
    return build_track_rows_from_matches(
        reference.session_names,
        consecutive_matches,
        start_roi_indices=start_roi_indices,
        start_session_index=seed_session,
        fill_value=-1,
    )


def solve_configured_global_assignment(
    sessions: Sequence[Track2pSession],
    config: Track2pBenchmarkConfig,
    *,
    cost: AssociationCost | None = None,
    calibrated_model: CalibratedAssociationModel | None = None,
) -> GlobalAssignmentRun:
    """Run global assignment using the benchmark configuration knobs."""

    return solve_global_assignment_for_sessions(
        sessions,
        max_gap=config.max_gap,
        cost=config.cost if cost is None else cost,
        calibrated_model=calibrated_model,
        transform_type=config.transform_type,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
    )


def _variant_name(cost: AssociationCost) -> str:
    if cost == "registered-iou":
        return "Same costs + global assignment"
    if cost == "calibrated":
        return "Calibrated costs + global assignment"
    return "BayesCaTrack costs + global assignment"


# pylint: disable=too-many-return-statements,too-many-branches
def _load_reference_for_subject(
    subject_dir: Path, *, data_root: Path, config: Track2pBenchmarkConfig
) -> Track2pReference:
    data_root = Path(data_root)
    if config.reference is None:
        if config.reference_kind == "manual-gt":
            default_ground_truth_path = subject_dir / GROUND_TRUTH_CSV_NAME
            if default_ground_truth_path.exists():
                return _load_ground_truth_csv_reference(
                    default_ground_truth_path, subject_dir=subject_dir
                )
            raise ValueError(
                f"--reference-kind manual-gt was requested, but {default_ground_truth_path} does not exist"
            )
        if config.reference_kind == "track2p-output":
            track2p_dir = subject_dir / "track2p"
            if track2p_dir.exists():
                return load_track2p_reference(track2p_dir, plane_name=config.plane_name)
            raise ValueError(
                f"--reference-kind track2p-output was requested, but {track2p_dir} does not exist"
            )
        if config.reference_kind == "aligned-subject-rows":
            return _load_aligned_reference_for_config(subject_dir, config)

        default_ground_truth_path = subject_dir / GROUND_TRUTH_CSV_NAME
        if default_ground_truth_path.exists():
            return _load_ground_truth_csv_reference(
                default_ground_truth_path, subject_dir=subject_dir
            )
        track2p_dir = subject_dir / "track2p"
        if track2p_dir.exists():
            return load_track2p_reference(track2p_dir, plane_name=config.plane_name)
        return _load_aligned_reference_for_config(subject_dir, config)

    reference_root = Path(config.reference)
    if config.reference_kind == "manual-gt":
        if reference_root.is_file():
            return _load_ground_truth_csv_reference(
                reference_root, subject_dir=subject_dir
            )
        ground_truth_path = _resolve_ground_truth_csv_path(
            subject_dir, data_root=data_root, reference_root=reference_root
        )
        if ground_truth_path is None:
            raise ValueError(
                "--reference-kind manual-gt was requested, but no ground_truth.csv could be resolved "
                f"for subject {subject_dir.name!r} under {reference_root}"
            )
        return _load_ground_truth_csv_reference(
            ground_truth_path, subject_dir=subject_dir
        )

    if config.reference_kind == "track2p-output":
        reference_path = _resolve_track2p_reference_path(
            subject_dir, data_root=data_root, reference_root=reference_root
        )
        if reference_path is None:
            raise ValueError(
                "--reference-kind track2p-output was requested, but no Track2p reference could be resolved "
                f"for subject {subject_dir.name!r} under {reference_root}"
            )
        return load_track2p_reference(reference_path, plane_name=config.plane_name)

    if config.reference_kind == "aligned-subject-rows":
        return _load_aligned_reference_for_config(subject_dir, config)

    ground_truth_path = _resolve_ground_truth_csv_path(
        subject_dir, data_root=data_root, reference_root=reference_root
    )
    if ground_truth_path is not None:
        return _load_ground_truth_csv_reference(
            ground_truth_path, subject_dir=subject_dir
        )

    reference_path = _resolve_track2p_reference_path(
        subject_dir, data_root=data_root, reference_root=reference_root
    )
    if reference_path is not None:
        return load_track2p_reference(reference_path, plane_name=config.plane_name)
    return _load_aligned_reference_for_config(subject_dir, config)


def _validate_reference_for_benchmark(
    reference: Track2pReference,
    *,
    subject_dir: Path,
    config: Track2pBenchmarkConfig,
) -> None:
    if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
        return
    if config.allow_track2p_as_reference_for_smoke_test:
        return
    raise ValueError(
        f"Subject {subject_dir.name!r} resolved reference source {reference.source!r}, "
        "which is not independent manual ground truth. Provide --reference pointing at "
        "ground_truth.csv or a ground-truth root with --reference-kind manual-gt. "
        "For plumbing checks only, pass --allow-track2p-as-reference-for-smoke-test."
    )


def _resolve_ground_truth_csv_path(
    subject_dir: Path, *, data_root: Path, reference_root: Path
) -> Path | None:
    candidates: list[Path] = []
    if reference_root.is_file():
        candidates.append(reference_root)
    else:
        candidates.extend(
            [
                reference_root / GROUND_TRUTH_CSV_NAME,
                reference_root / subject_dir.name / GROUND_TRUTH_CSV_NAME,
            ]
        )
        relative_subject: Path | None
        try:
            relative_subject = subject_dir.relative_to(data_root)
        except ValueError:
            relative_subject = None
        if relative_subject is not None:
            candidates.append(reference_root / relative_subject / GROUND_TRUTH_CSV_NAME)

    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.name.casefold() == GROUND_TRUTH_CSV_NAME
        ):
            return candidate
    return None


def _resolve_track2p_reference_path(
    subject_dir: Path, *, data_root: Path, reference_root: Path
) -> Path | None:
    candidates = [
        reference_root,
        reference_root / subject_dir.name,
        reference_root / subject_dir.name / "track2p",
        reference_root / "track2p",
    ]
    relative_subject: Path | None
    try:
        relative_subject = subject_dir.relative_to(data_root)
    except ValueError:
        relative_subject = None
    if relative_subject is not None:
        candidates.extend(
            [
                reference_root / relative_subject,
                reference_root / relative_subject / "track2p",
            ]
        )
    for candidate in candidates:
        if (candidate / "track_ops.npy").exists() or (
            candidate / "track2p" / "track_ops.npy"
        ).exists():
            return candidate
    return None


def _load_ground_truth_csv_reference(
    ground_truth_path: Path, *, subject_dir: Path
) -> Track2pReference:
    session_names = tuple(
        session_dir.name for session_dir in find_track2p_session_dirs(subject_dir)
    )
    if not session_names:
        raise ValueError(
            f"No Track2p-style sessions were found for ground-truth reference {ground_truth_path}"
        )

    track_table = load_track2p_ground_truth_csv(ground_truth_path)
    if track_table.session_names != session_names:
        if set(track_table.session_names) != set(session_names):
            raise ValueError(
                f"{ground_truth_path} session columns {track_table.session_names!r} do not match subject sessions {session_names!r}"
            )
        track_table = track_table.aligned_to(session_names)

    return Track2pReference(
        session_names=track_table.session_names,
        suite2p_indices=track_table.tracks,
        curated_mask=np.ones((track_table.n_tracks,), dtype=bool),
        source=GROUND_TRUTH_REFERENCE_SOURCE,
    )


def _score_prediction_against_reference(
    predicted_matrix: np.ndarray,
    reference: Track2pReference,
    *,
    config: Track2pBenchmarkConfig,
) -> dict[str, float | int]:
    reference_matrix = _reference_matrix(reference, curated_only=config.curated_only)
    predicted = normalize_track_matrix(predicted_matrix)
    if predicted.shape[1] != reference_matrix.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    predicted_before_filter = int(predicted.shape[0])
    reference_seed_rois: set[int] = set()
    if config.restrict_to_reference_seed_rois:
        reference_seed_rois = _reference_seed_roi_set(
            reference_matrix, seed_session=config.seed_session
        )
        predicted = _filter_tracks_by_seed_rois(
            predicted,
            reference_seed_rois,
            seed_session=config.seed_session,
        )
        reference_matrix = _filter_tracks_by_seed_rois(
            reference_matrix,
            reference_seed_rois,
            seed_session=config.seed_session,
        )

    scores = _with_recomputed_f1_scores(
        score_track_matrices(predicted, reference_matrix)
    )
    if config.restrict_to_reference_seed_rois:
        scores = {
            **scores,
            "seed_session": int(config.seed_session),
            "reference_seed_rois": int(len(reference_seed_rois)),
            "evaluated_prediction_tracks": int(predicted.shape[0]),
            "dropped_prediction_tracks": int(
                predicted_before_filter - predicted.shape[0]
            ),
        }
    return scores


def _with_recomputed_f1_scores(
    scores: Mapping[str, float | int],
) -> dict[str, float | int]:
    repaired_scores = dict(scores)
    for prefix in ("pairwise", "complete_track"):
        tp = int(repaired_scores.get(f"{prefix}_true_positives", 0))
        fp = int(repaired_scores.get(f"{prefix}_false_positives", 0))
        fn = int(repaired_scores.get(f"{prefix}_false_negatives", 0))
        repaired_scores[f"{prefix}_f1"] = _f1_from_counts(tp, fp, fn)
    return repaired_scores


def _f1_from_counts(
    true_positives: int, false_positives: int, false_negatives: int
) -> float:
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 1.0
    return float(2 * true_positives / denominator)


def _filter_tracks_by_seed_rois(
    predicted_matrix: np.ndarray,
    seed_rois: set[int],
    *,
    seed_session: int,
) -> np.ndarray:
    if not seed_rois:
        return predicted_matrix[:0]
    keep = [
        _is_valid_roi_index(row[seed_session]) and int(row[seed_session]) in seed_rois
        for row in predicted_matrix
    ]
    return predicted_matrix[np.asarray(keep, dtype=bool)]


def _reference_seed_roi_set(
    reference_matrix: np.ndarray, *, seed_session: int
) -> set[int]:
    if seed_session < 0 or seed_session >= reference_matrix.shape[1]:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {reference_matrix.shape[1]} sessions"
        )
    return {
        int(cast(Any, value))
        for value in reference_matrix[:, seed_session]
        if _is_valid_roi_index(value)
    }


def _is_valid_roi_index(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(cast(Any, value)) >= 0
    except (TypeError, ValueError):
        return False


def _load_aligned_reference_for_config(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> Track2pReference:
    return load_aligned_subject_reference(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
    )


def _load_subject_sessions(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> list[Track2pSession]:
    return load_track2p_subject(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
    )


def _validate_reference_roi_indices(
    reference: Track2pReference, sessions: Sequence[Track2pSession]
) -> None:
    sessions = tuple(sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            f"Reference {reference.source!r} has {reference.n_sessions} sessions, but the loaded subject has {len(sessions)} sessions"
        )

    session_names = tuple(session.session_name for session in sessions)
    if session_names != reference.session_names:
        raise ValueError(
            f"Reference {reference.source!r} session order {reference.session_names!r} does not match loaded sessions {session_names!r}"
        )

    for session_index, session in enumerate(sessions):
        available_indices = _loaded_suite2p_index_set(session)
        referenced_indices = {
            int(value)
            for value in reference.suite2p_indices[:, session_index]
            if _is_valid_roi_index(value)
        }
        missing_indices = sorted(referenced_indices - available_indices)
        if missing_indices:
            preview = ", ".join(str(value) for value in missing_indices[:10])
            suffix = (
                ""
                if len(missing_indices) <= 10
                else f", ... ({len(missing_indices)} total)"
            )
            raise ValueError(
                "Reference ROI indices are absent from loaded session "
                f"{session.session_name!r}: {preview}{suffix}. "
                "This usually means the reference uses Suite2p stat.npy row indices, "
                "but the benchmark loaded a filtered ROI set. Re-run with --include-non-cells "
                "or adjust --cell-probability-threshold if this is intentional."
            )


def _loaded_suite2p_index_set(session: Track2pSession) -> set[int]:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return set(range(session.plane_data.n_rois))
    return {int(value) for value in np.asarray(roi_indices, dtype=int).reshape(-1)}


def _reference_matrix(reference: Track2pReference, *, curated_only: bool) -> np.ndarray:
    matrix = normalize_track_matrix(reference.suite2p_indices)
    if not curated_only:
        return matrix
    if reference.curated_mask is None:
        raise ValueError(
            "--curated-only was requested, but the reference has no curation mask"
        )
    return matrix[np.asarray(reference.curated_mask, dtype=bool)]


def _looks_like_subject_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "track2p").exists():
        return True
    return bool(find_track2p_session_dirs(path))


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    return Track2pBenchmarkConfig(
        data=args.data,
        method=args.method,
        split=args.split,
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=args.cost,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )


def _write_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_benchmark_table(rows))


def _csv_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
        "seed_session",
        "reference_seed_rois",
        "evaluated_prediction_tracks",
        "dropped_prediction_tracks",
        "training_examples",
        "positive_examples",
        "negative_examples",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_table_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
