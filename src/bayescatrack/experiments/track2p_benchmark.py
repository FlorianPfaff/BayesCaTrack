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
from bayescatrack.association.edge_thresholds import (
    EdgeThresholdPolicy,
    compute_manual_oracle_edge_cost_thresholds,
)
from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    build_registered_pairwise_costs,
    session_edge_pairs,
    solve_global_assignment_from_pairwise_costs,
    solve_global_assignment_for_sessions,
    solve_track2p_style_propagation_for_sessions,
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
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES

ReferenceKind = Literal["auto", "manual-gt", "track2p-output", "aligned-subject-rows"]
BenchmarkMethod = Literal[
    "track2p-baseline",
    "track2p-clone",
    "global-assignment",
    "track2p-style-propagation",
    "oracle-gt-links",
    "oracle-gt-solver",
    "oracle-gt-consecutive-solver",
]
BenchmarkSplit = Literal["subject", "leave-one-subject-out"]
Track2pThresholdMethod = Literal["otsu", "min"]
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
    track2p_iou_dist_threshold: float = 16.0
    track2p_threshold_method: Track2pThresholdMethod = "otsu"
    track2p_threshold_remove_zeros: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    edge_threshold_policy: EdgeThresholdPolicy = "none"
    edge_threshold_otsu_bins: int = 256
    edge_threshold_otsu_max_cost: float | None = None
    oracle_match_cost: float = 0.0
    oracle_nonmatch_cost: float = 1.0e6
    higher_order_triplet_weight: float = 0.0
    higher_order_support_top_k: int = 8
    higher_order_support_cost_cap: float = 4.0
    higher_order_max_penalty: float = 2.0
    higher_order_large_cost: float = 1.0e6
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
        if config.method != "global-assignment" or config.cost not in {
            "calibrated",
            "monotone-ranked",
        }:
            raise ValueError(
                "LOSO calibration requires method='global-assignment' and "
                "cost='calibrated' or cost='monotone-ranked'"
            )
        from bayescatrack.experiments.track2p_loso_calibration import (
            run_track2p_loso_calibration,
        )

        return run_track2p_loso_calibration(config).to_benchmark_results()
    if config.cost in {"calibrated", "monotone-ranked"}:
        raise ValueError(f"cost={config.cost!r} requires split='leave-one-subject-out'")

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
                reference, _load_reference_validation_sessions(subject_dir, config)
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


def add_higher_order_consistency_arguments(parser: argparse.ArgumentParser) -> None:
    """Add paper-facing higher-order triplet-consistency CLI knobs."""

    parser.add_argument(
        "--higher-order-triplet-weight",
        "--triplet-weight",
        dest="higher_order_triplet_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for triplet-support penalties added to pairwise costs before "
            "global assignment. The default 0 disables higher-order consistency."
        ),
    )
    parser.add_argument(
        "--higher-order-support-top-k",
        "--triplet-support-top-k",
        dest="higher_order_support_top_k",
        type=int,
        default=8,
        help=(
            "Maximum number of admissible supporting ROIs kept per shared "
            "third-session ROI when approximating triplet support."
        ),
    )
    parser.add_argument(
        "--higher-order-support-cost-cap",
        "--triplet-support-cost-cap",
        dest="higher_order_support_cost_cap",
        type=float,
        default=4.0,
        help=(
            "Maximum pairwise cost considered as third-session support; unsupported "
            "edges receive a penalty."
        ),
    )
    parser.add_argument(
        "--higher-order-max-penalty",
        "--triplet-max-penalty",
        dest="higher_order_max_penalty",
        type=float,
        default=2.0,
        help="Maximum unweighted triplet-consistency penalty added to one edge.",
    )
    parser.add_argument(
        "--higher-order-large-cost",
        "--triplet-large-cost",
        dest="higher_order_large_cost",
        type=float,
        default=1.0e6,
        help=(
            "Cost value treated as an inadmissible/gated edge by the "
            "triplet-consistency penalty."
        ),
    )


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
        choices=(
            "track2p-baseline",
            "track2p-clone",
            "global-assignment",
            "track2p-style-propagation",
            "oracle-gt-links",
            "oracle-gt-solver",
            "oracle-gt-consecutive-solver",
        ),
        help=(
            "Benchmark variant to run. The oracle-gt-solver variants use manual-GT "
            "edges as pairwise costs but still exercise the global solver."
        ),
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
        choices=(
            "registered-iou",
            "roi-aware",
            "calibrated",
            "monotone-ranked",
        ),
        help="Pairwise cost or LOSO-trained association model used by global assignment",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help=(
            "Maximum forward session gap for global-assignment edges; "
            "Track2p-style propagation always uses consecutive edges"
        ),
    )
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=REGISTRATION_TRANSFORM_TYPES,
        help="Registration transform type",
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
        "--track2p-iou-dist-threshold",
        type=float,
        default=16.0,
        help="Track2p-clone centroid-distance gate for computing pairwise IoU",
    )
    parser.add_argument(
        "--track2p-threshold-method",
        choices=("otsu", "min"),
        default="otsu",
        help="Track2p-clone assigned-IoU thresholding method",
    )
    parser.add_argument(
        "--track2p-threshold-remove-zeros",
        action="store_true",
        help="Remove zero assigned IoUs before computing the Track2p-clone threshold",
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
        "--edge-threshold-policy",
        default="none",
        choices=("none", "otsu", "manual-oracle"),
        help=(
            "Optional per-session-edge pre-solver thresholding. 'otsu' is "
            "unsupervised; 'manual-oracle' chooses the F1-optimal edge "
            "threshold from manual ground truth for diagnostics only."
        ),
    )
    parser.add_argument(
        "--edge-threshold-otsu-bins",
        type=int,
        default=256,
        help="Histogram bins used by --edge-threshold-policy otsu",
    )
    parser.add_argument(
        "--edge-threshold-otsu-max-cost",
        type=float,
        default=None,
        help="Optional maximum finite cost included in Otsu threshold histograms",
    )
    parser.add_argument(
        "--oracle-match-cost",
        type=float,
        default=0.0,
        help="Pairwise cost assigned to manual-GT links in oracle solver ablations",
    )
    parser.add_argument(
        "--oracle-nonmatch-cost",
        type=float,
        default=1.0e6,
        help="Pairwise cost assigned to non-GT links in oracle solver ablations",
    )
    add_higher_order_consistency_arguments(parser)
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

    if config.method == "track2p-clone":
        from bayescatrack.experiments.internal_track2p_clone import (
            run_internal_track2p_clone,
        )

        clone_result = run_internal_track2p_clone(
            subject_dir,
            plane_name=config.plane_name,
            input_format=config.input_format,
            include_behavior=config.include_behavior,
            transform_type=config.transform_type,
            iscell_threshold=config.cell_probability_threshold,
            iou_dist_threshold=config.track2p_iou_dist_threshold,
            threshold_method=config.track2p_threshold_method,
            threshold_remove_zeros=config.track2p_threshold_remove_zeros,
        )
        return (
            normalize_track_matrix(clone_result.suite2p_indices),
            "Internal Track2p clone",
        )

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

    if config.method == "track2p-style-propagation":
        sessions = _load_subject_sessions(subject_dir, config)
        seed_detection_indices = None
        if reference is not None and config.restrict_to_reference_seed_rois:
            reference_matrix = _reference_matrix(
                reference,
                curated_only=config.curated_only,
            )
            reference_seed_rois = _reference_seed_roi_set(
                reference_matrix,
                seed_session=config.seed_session,
            )
            seed_detection_indices = _detection_indices_for_suite2p_rois(
                sessions[config.seed_session],
                reference_seed_rois,
            )
        assignment = solve_configured_track2p_style_propagation(
            sessions, config, seed_detection_indices=seed_detection_indices
        )
        predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
        return predicted, _track2p_style_variant_name(config.cost)

    if config.method != "global-assignment":
        if config.method in {"oracle-gt-solver", "oracle-gt-consecutive-solver"}:
            if reference is None:
                raise ValueError(f"{config.method} requires a loaded reference")
            sessions = _load_subject_sessions(subject_dir, config)
            if reference.n_sessions == 1:
                return (
                    oracle_ground_truth_link_tracks(
                        reference,
                        curated_only=config.curated_only,
                        seed_session=config.seed_session,
                    ),
                    "Oracle GT solver costs (single session)",
                )
            oracle_max_gap = (
                1
                if config.method == "oracle-gt-consecutive-solver"
                else config.max_gap
            )
            assignment = oracle_ground_truth_solver_assignment(
                reference,
                sessions,
                curated_only=config.curated_only,
                max_gap=oracle_max_gap,
                start_cost=config.start_cost,
                end_cost=config.end_cost,
                gap_penalty=config.gap_penalty,
                cost_threshold=config.cost_threshold,
                match_cost=config.oracle_match_cost,
                nonmatch_cost=config.oracle_nonmatch_cost,
            )
            predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
            return predicted, _oracle_solver_variant_name(config.method, oracle_max_gap)
        raise ValueError(f"Unsupported benchmark method: {config.method!r}")

    sessions = _load_subject_sessions(subject_dir, config)
    assignment = solve_configured_global_assignment(sessions, config, reference=reference)
    predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    return predicted, _variant_name(
        config.cost,
        edge_threshold_policy=config.edge_threshold_policy,
        higher_order_consistency_config=_higher_order_consistency_config(config),
    )


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


# pylint: disable=too-many-arguments
def oracle_ground_truth_solver_assignment(
    reference: Track2pReference,
    sessions: Sequence[Track2pSession],
    *,
    curated_only: bool = False,
    max_gap: int = 2,
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
    match_cost: float = 0.0,
    nonmatch_cost: float = 1.0e6,
) -> GlobalAssignmentRun:
    """Run the normal global solver on oracle manual-GT pairwise costs.

    Unlike :func:`oracle_ground_truth_link_tracks`, this diagnostic does not
    stitch ground-truth links directly into track rows. It builds pairwise cost
    matrices in the same loaded-ROI coordinate system used by the PyRecEst path
    cover solver, assigns ``match_cost`` to manual-GT links, assigns
    ``nonmatch_cost`` to every other link, and then runs the normal solver. A
    failure here therefore points to solver priors, max-gap policy, thresholding,
    ROI-index conversion, or score/evaluation plumbing rather than pairwise
    registration/ranking.
    """

    sessions = list(sessions)
    if reference.n_sessions != len(sessions):
        raise ValueError(
            f"Reference has {reference.n_sessions} sessions; got {len(sessions)}"
        )
    if reference.n_sessions <= 1:
        raise ValueError(
            "oracle_ground_truth_solver_assignment requires at least two sessions; "
            "use oracle_ground_truth_link_tracks for single-session references"
        )
    _validate_oracle_cost_values(
        match_cost=match_cost, nonmatch_cost=nonmatch_cost
    )

    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    session_edges = session_edge_pairs(reference.n_sessions, max_gap=max_gap)
    pairwise_costs = _oracle_pairwise_costs_from_reference(
        reference,
        sessions,
        max_gap=max_gap,
        curated_only=curated_only,
        match_cost=match_cost,
        nonmatch_cost=nonmatch_cost,
    )
    return solve_global_assignment_from_pairwise_costs(
        pairwise_costs,
        session_sizes=session_sizes,
        session_edges=session_edges,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
    )


def _oracle_pairwise_costs_from_reference(
    reference: Track2pReference,
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int,
    curated_only: bool,
    match_cost: float,
    nonmatch_cost: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Return GT-vs-non-GT pairwise costs in loaded-ROI coordinates."""

    sessions = list(sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError("sessions must have the same length as the reference")

    _validate_oracle_cost_values(
        match_cost=match_cost, nonmatch_cost=nonmatch_cost
    )
    loaded_position_by_suite2p_index = [
        _suite2p_index_to_loaded_position(session) for session in sessions
    ]
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)

    pairwise_costs: dict[tuple[int, int], np.ndarray] = {}
    for source_session, target_session in session_edge_pairs(
        reference.n_sessions, max_gap=max_gap
    ):
        cost_matrix = np.full(
            (session_sizes[source_session], session_sizes[target_session]),
            float(nonmatch_cost),
            dtype=float,
        )
        for source_roi, target_roi in reference.pairwise_matches(
            source_session, target_session, curated_only=curated_only
        ):
            source_position = _loaded_position_for_reference_roi(
                loaded_position_by_suite2p_index[source_session],
                source_roi,
                session_index=source_session,
            )
            target_position = _loaded_position_for_reference_roi(
                loaded_position_by_suite2p_index[target_session],
                target_roi,
                session_index=target_session,
            )
            cost_matrix[source_position, target_position] = float(match_cost)
        pairwise_costs[(source_session, target_session)] = cost_matrix
    return pairwise_costs


def _suite2p_index_to_loaded_position(session: Track2pSession) -> dict[int, int]:
    plane = session.plane_data
    if plane.roi_indices is None:
        return {index: index for index in range(int(plane.n_rois))}

    roi_indices = np.asarray(plane.roi_indices, dtype=int).reshape(-1)
    if roi_indices.shape[0] != int(plane.n_rois):
        raise ValueError("plane_data.roi_indices must contain one index per loaded ROI")

    positions: dict[int, int] = {}
    for loaded_position, suite2p_index in enumerate(roi_indices):
        suite2p_index = int(suite2p_index)
        if suite2p_index in positions:
            raise ValueError(f"Duplicate Suite2p ROI {suite2p_index}")
        positions[suite2p_index] = int(loaded_position)
    return positions


def _loaded_position_for_reference_roi(
    loaded_position_by_suite2p_index: Mapping[int, int],
    suite2p_roi_index: int,
    *,
    session_index: int,
) -> int:
    try:
        return loaded_position_by_suite2p_index[int(suite2p_roi_index)]
    except KeyError as exc:
        raise ValueError(
            "Reference ROI index is absent from the loaded ROI set for session "
            f"{session_index}: {int(suite2p_roi_index)}. Try --include-non-cells "
            "or adjust --cell-probability-threshold/overlap filtering."
        ) from exc


def _validate_oracle_cost_values(*, match_cost: float, nonmatch_cost: float) -> None:
    if not np.isfinite(match_cost):
        raise ValueError("match_cost must be finite")
    if not np.isfinite(nonmatch_cost):
        raise ValueError("nonmatch_cost must be finite")
    if float(nonmatch_cost) <= float(match_cost):
        raise ValueError("nonmatch_cost must be larger than match_cost")


def solve_configured_global_assignment(
    sessions: Sequence[Track2pSession],
    config: Track2pBenchmarkConfig,
    *,
    cost: AssociationCost | None = None,
    calibrated_model: Any | None = None,
    reference: Track2pReference | None = None,
) -> GlobalAssignmentRun:
    """Run global assignment using the benchmark configuration knobs."""

    selected_cost = config.cost if cost is None else cost
    if config.edge_threshold_policy == "manual-oracle":
        if reference is None:
            raise ValueError("manual-oracle edge thresholds require a reference")
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "--edge-threshold-policy manual-oracle requires independent "
                "manual ground truth, not Track2p or aligned-row references"
            )
        pairwise_costs = build_registered_pairwise_costs(
            sessions,
            max_gap=config.max_gap,
            cost=selected_cost,
            calibrated_model=calibrated_model,
            transform_type=config.transform_type,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        )
        session_edges = session_edge_pairs(len(sessions), max_gap=config.max_gap)
        true_match_masks = _reference_match_masks_for_loaded_sessions(
            reference,
            sessions,
            session_edges,
            curated_only=config.curated_only,
        )
        edge_cost_thresholds = compute_manual_oracle_edge_cost_thresholds(
            pairwise_costs,
            true_match_masks,
        )
        return solve_global_assignment_from_pairwise_costs(
            pairwise_costs,
            session_sizes=tuple(int(session.plane_data.n_rois) for session in sessions),
            session_edges=session_edges,
            start_cost=config.start_cost,
            end_cost=config.end_cost,
            gap_penalty=config.gap_penalty,
            cost_threshold=config.cost_threshold,
            edge_cost_thresholds=edge_cost_thresholds,
        )

    return solve_global_assignment_for_sessions(
        sessions,
        max_gap=config.max_gap,
        cost=selected_cost,
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
        higher_order_consistency_config=_higher_order_consistency_config(config),
        edge_threshold_policy=config.edge_threshold_policy,
        edge_threshold_otsu_bins=config.edge_threshold_otsu_bins,
        edge_threshold_otsu_max_cost=config.edge_threshold_otsu_max_cost,
    )


def solve_configured_track2p_style_propagation(
    sessions: Sequence[Track2pSession],
    config: Track2pBenchmarkConfig,
    *,
    cost: AssociationCost | None = None,
    calibrated_model: Any | None = None,
    seed_detection_indices: Sequence[int] | None = None,
) -> GlobalAssignmentRun:
    """Run seed-restricted Track2p-style propagation with benchmark knobs."""

    return solve_track2p_style_propagation_for_sessions(
        sessions,
        cost=config.cost if cost is None else cost,
        calibrated_model=calibrated_model,
        transform_type=config.transform_type,
        cost_threshold=config.cost_threshold,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        seed_session=config.seed_session,
        seed_detection_indices=seed_detection_indices,
    )


def _higher_order_consistency_config(
    config: Track2pBenchmarkConfig,
) -> HigherOrderConsistencyConfig | None:
    """Return the enabled triplet-consistency config for a benchmark run."""

    higher_order_config = HigherOrderConsistencyConfig(
        triplet_weight=config.higher_order_triplet_weight,
        support_top_k=config.higher_order_support_top_k,
        support_cost_cap=config.higher_order_support_cost_cap,
        max_penalty=config.higher_order_max_penalty,
        large_cost=config.higher_order_large_cost,
    )
    return higher_order_config if higher_order_config.enabled else None


def _variant_name(
    cost: AssociationCost,
    *,
    edge_threshold_policy: EdgeThresholdPolicy = "none",
    higher_order_consistency_config: HigherOrderConsistencyConfig | None = None,
) -> str:
    if cost == "registered-iou":
        variant = "Same costs + global assignment"
    elif cost == "calibrated":
        variant = "Calibrated costs + global assignment"
    else:
        variant = "BayesCaTrack costs + global assignment"
    if edge_threshold_policy != "none":
        variant += f" + {edge_threshold_policy} edge thresholds"
    if (
        higher_order_consistency_config is not None
        and higher_order_consistency_config.enabled
    ):
        variant += " + triplet consistency"
    return variant


def _track2p_style_variant_name(cost: AssociationCost) -> str:
    if cost == "registered-iou":
        return "Same costs + Track2p-style propagation"
    if cost == "calibrated":
        return "Calibrated costs + Track2p-style propagation"
    return "BayesCaTrack costs + Track2p-style propagation"


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


def _load_reference_validation_sessions(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> list[Track2pSession]:
    if config.method != "track2p-clone":
        return _load_subject_sessions(subject_dir, config)

    from bayescatrack.experiments.internal_track2p_clone import (
        load_internal_track2p_clone_sessions,
    )

    return load_internal_track2p_clone_sessions(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        iscell_threshold=config.cell_probability_threshold,
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


def _detection_indices_for_suite2p_rois(
    session: Track2pSession,
    suite2p_roi_indices: set[int],
) -> tuple[int, ...]:
    if not suite2p_roi_indices:
        return tuple()

    plane = session.plane_data
    if plane.roi_indices is None:
        suite2p_to_detection = {
            int(index): int(index) for index in range(int(plane.n_rois))
        }
    else:
        suite2p_to_detection = {
            int(suite2p_index): int(detection_index)
            for detection_index, suite2p_index in enumerate(
                np.asarray(plane.roi_indices, dtype=int).reshape(-1)
            )
        }

    missing = sorted(
        int(index) for index in suite2p_roi_indices if index not in suite2p_to_detection
    )
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Reference seed ROIs are absent from loaded session "
            f"{session.session_name!r}: {preview}{suffix}"
        )
    return tuple(
        suite2p_to_detection[int(index)] for index in sorted(suite2p_roi_indices)
    )


def _reference_match_masks_for_loaded_sessions(
    reference: Track2pReference,
    sessions: Sequence[Track2pSession],
    session_edges: Sequence[tuple[int, int]],
    *,
    curated_only: bool,
) -> dict[tuple[int, int], np.ndarray]:
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            "Reference and loaded sessions must have the same number of sessions"
        )
    suite2p_to_loaded = [_loaded_suite2p_to_loaded_index(session) for session in sessions]
    masks: dict[tuple[int, int], np.ndarray] = {}
    for source_session, target_session in session_edges:
        mask = np.zeros(
            (
                sessions[source_session].plane_data.n_rois,
                sessions[target_session].plane_data.n_rois,
            ),
            dtype=bool,
        )
        for source_roi, target_roi in reference.pairwise_matches(
            source_session, target_session, curated_only=curated_only
        ):
            source_loaded = suite2p_to_loaded[source_session].get(int(source_roi))
            target_loaded = suite2p_to_loaded[target_session].get(int(target_roi))
            if source_loaded is not None and target_loaded is not None:
                mask[source_loaded, target_loaded] = True
        masks[(source_session, target_session)] = mask
    return masks


def _loaded_suite2p_to_loaded_index(session: Track2pSession) -> dict[int, int]:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return {index: index for index in range(session.plane_data.n_rois)}
    return {
        int(suite2p_index): loaded_index
        for loaded_index, suite2p_index in enumerate(
            np.asarray(roi_indices, dtype=int).reshape(-1)
        )
    }


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
        track2p_iou_dist_threshold=args.track2p_iou_dist_threshold,
        track2p_threshold_method=args.track2p_threshold_method,
        track2p_threshold_remove_zeros=args.track2p_threshold_remove_zeros,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        edge_threshold_policy=args.edge_threshold_policy,
        edge_threshold_otsu_bins=args.edge_threshold_otsu_bins,
        edge_threshold_otsu_max_cost=args.edge_threshold_otsu_max_cost,
        oracle_match_cost=args.oracle_match_cost,
        oracle_nonmatch_cost=args.oracle_nonmatch_cost,
        higher_order_triplet_weight=args.higher_order_triplet_weight,
        higher_order_support_top_k=args.higher_order_support_top_k,
        higher_order_support_cost_cap=args.higher_order_support_cost_cap,
        higher_order_max_penalty=args.higher_order_max_penalty,
        higher_order_large_cost=args.higher_order_large_cost,
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
        "calibration_model",
        "monotone_preference_pairs",
        "monotone_iterations",
        "monotone_training_loss",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_table_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
