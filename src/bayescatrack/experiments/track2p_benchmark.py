"""Reproducible Track2p benchmark harness for BayesCaTrack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bayescatrack.association.calibrated_costs import CalibratedAssociationModel
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    solve_global_assignment_for_sessions,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession, find_track2p_session_dirs, load_track2p_subject
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix, score_track_matrices
from bayescatrack.reference import Track2pReference, load_aligned_subject_reference, load_track2p_reference

BenchmarkMethod = Literal["track2p-baseline", "global-assignment"]
BenchmarkSplit = Literal["subject", "leave-one-subject-out"]
OutputFormat = Literal["table", "json", "csv"]


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
    curated_only: bool = False
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


def run_track2p_benchmark(config: Track2pBenchmarkConfig) -> list[SubjectBenchmarkResult]:
    """Run a Track2p benchmark over one subject directory or a dataset root."""

    if config.split == "leave-one-subject-out":
        if config.method != "global-assignment" or config.cost != "calibrated":
            raise ValueError("LOSO calibration requires method='global-assignment' and cost='calibrated'")
        from bayescatrack.experiments.track2p_loso_calibration import run_track2p_loso_calibration

        return run_track2p_loso_calibration(config).to_benchmark_results()
    if config.cost == "calibrated":
        raise ValueError("cost='calibrated' requires split='leave-one-subject-out'")

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {config.data}")

    results: list[SubjectBenchmarkResult] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(subject_dir, data_root=config.data, config=config)
        reference_matrix = _reference_matrix(reference, curated_only=config.curated_only)
        predicted_matrix, variant = _predict_subject_tracks(subject_dir, config)
        scores = score_track_matrices(predicted_matrix, reference_matrix)
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=variant,
                method=config.method,
                scores=scores,
                n_sessions=reference_matrix.shape[1],
                reference_source=reference.source,
            )
        )
    return results


def discover_subject_dirs(data_path: str | Path) -> list[Path]:
    """Find Track2p subject directories beneath ``data_path``."""

    root = Path(data_path)
    if _looks_like_subject_dir(root):
        return [root]
    subjects = [child for child in sorted(root.iterdir()) if child.is_dir() and _looks_like_subject_dir(child)]
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


def write_results(rows: Sequence[dict[str, float | int | str]], output_path: Path, output_format: OutputFormat) -> None:
    """Write benchmark rows as JSON, CSV, or Markdown table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--data", required=True, type=Path, help="Track2p dataset root or one subject directory")
    parser.add_argument(
        "--method",
        required=True,
        choices=("track2p-baseline", "global-assignment"),
        help="Benchmark variant to run",
    )
    parser.add_argument(
        "--split",
        default="subject",
        choices=("subject", "leave-one-subject-out"),
        help="Evaluation split policy",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0", help="Plane name such as plane0")
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
        help="Optional ground-truth root, subject directory, or track2p folder",
    )
    parser.add_argument("--curated-only", action="store_true", help="Evaluate only reference tracks marked curated")
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware", "calibrated"),
        help="Pairwise cost used by global assignment",
    )
    parser.add_argument("--max-gap", type=int, default=2, help="Maximum forward session gap for global-assignment edges")
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "none"),
        help="Track2p registration transform type",
    )
    parser.add_argument("--start-cost", type=float, default=5.0, help="PyRecEst track start cost")
    parser.add_argument("--end-cost", type=float, default=5.0, help="PyRecEst track end cost")
    parser.add_argument("--gap-penalty", type=float, default=1.0, help="Penalty per skipped session")
    parser.add_argument("--cost-threshold", type=float, default=6.0, help="Maximum adjusted edge cost admitted by the solver")
    parser.add_argument("--no-cost-threshold", action="store_true", help="Disable the solver edge-cost threshold")
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load behaviour arrays when present",
    )
    parser.add_argument("--include-non-cells", action="store_true", help="Keep Suite2p ROIs that fail iscell filtering")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5, help="Suite2p iscell probability threshold")
    parser.add_argument("--weighted-masks", action="store_true", help="Use Suite2p lam weights while reconstructing masks")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"), help="Coordinate order for costs")
    parser.add_argument("--weighted-centroids", action="store_true", help="Use weighted centroids where masks contain weights")
    parser.add_argument("--velocity-variance", type=float, default=25.0, help="Velocity variance for association bundle state moments")
    parser.add_argument("--regularization", type=float, default=1.0e-6, help="Position covariance regularization")
    parser.add_argument("--pairwise-cost-kwargs-json", default=None, help="JSON object merged into pairwise cost kwargs")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file")
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table", help="Stdout/output format")
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


def _predict_subject_tracks(subject_dir: Path, config: Track2pBenchmarkConfig) -> tuple[np.ndarray, str]:
    if config.method == "track2p-baseline":
        baseline = load_track2p_reference(subject_dir / "track2p", plane_name=config.plane_name)
        return normalize_track_matrix(baseline.suite2p_indices), "Track2p default"

    sessions = load_track2p_subject(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
    )
    assignment = solve_configured_global_assignment(sessions, config)
    predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    return predicted, _variant_name(config.cost)


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


def _load_reference_for_subject(subject_dir: Path, *, data_root: Path, config: Track2pBenchmarkConfig) -> Track2pReference:
    if config.reference is None:
        track2p_dir = subject_dir / "track2p"
        if track2p_dir.exists():
            return load_track2p_reference(track2p_dir, plane_name=config.plane_name)
        return load_aligned_subject_reference(subject_dir, plane_name=config.plane_name, input_format=config.input_format)

    reference_path = _resolve_reference_path(subject_dir, data_root=data_root, reference_root=config.reference)
    if reference_path is not None:
        return load_track2p_reference(reference_path, plane_name=config.plane_name)
    return load_aligned_subject_reference(subject_dir, plane_name=config.plane_name, input_format=config.input_format)


def _resolve_reference_path(subject_dir: Path, *, data_root: Path, reference_root: Path) -> Path | None:
    del data_root
    candidates = [
        reference_root,
        reference_root / subject_dir.name,
        reference_root / subject_dir.name / "track2p",
        reference_root / "track2p",
    ]
    for candidate in candidates:
        if (candidate / "track_ops.npy").exists() or (candidate / "track2p" / "track_ops.npy").exists():
            return candidate
    return None


def _reference_matrix(reference: Track2pReference, *, curated_only: bool) -> np.ndarray:
    matrix = normalize_track_matrix(reference.suite2p_indices)
    if not curated_only:
        return matrix
    if reference.curated_mask is None:
        raise ValueError("--curated-only was requested, but the reference has no curation mask")
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
        curated_only=args.curated_only,
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
    )


def _write_stdout(rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat) -> None:
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
