"""Track2p-as-teacher disagreement diagnostics for Track2p benchmarks.

The command compares three longitudinal edge sets on the same manual-GT
subjects: manual ground truth, Track2p output, and BayesCaTrack global
assignment.  It writes CSVs that make Track2p useful as a teacher/debug oracle
without using Track2p as the paper-facing reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    _load_pyrecest_multisession_solver,
    build_registered_pairwise_costs,
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.reference import Track2pReference, load_track2p_reference

# pylint: disable=protected-access,too-many-locals,too-many-arguments,too-many-branches
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _resolve_track2p_reference_path,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)

CSVValue = str | int | float | bool | None
CSVRow = dict[str, CSVValue]
EdgeKey = tuple[int, int, int, int]
EdgeScope = Literal["solver", "consecutive", "all-pairs"]

DETAIL_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_a_name",
    "session_b_name",
    "gap",
    "roi_a",
    "roi_b",
    "category",
    "manual_gt_label",
    "track2p_label",
    "bayescatrack_label",
    "edge_source",
    "bayes_candidate_present",
    "bayes_eligible_after_threshold",
    "bayes_cost",
    "bayes_adjusted_cost",
    "bayes_row_rank",
    "bayes_col_rank",
    "bayes_mutual_top1",
    "bayes_best_target_suite2p",
    "bayes_best_adjusted_cost",
    "bayes_margin_to_best",
    "bayes_missing_reason",
    "cost_scale",
    "cost_threshold",
    "gap_penalty",
]

SUMMARY_FIELDNAMES = [
    "scope",
    "subject",
    "session_a",
    "session_b",
    "session_a_name",
    "session_b_name",
    "gap",
    "category",
    "count",
    "manual_gt_label",
    "track2p_label",
    "bayescatrack_label",
    "candidate_present_rate",
    "eligible_after_threshold_rate",
    "median_bayes_row_rank",
    "median_bayes_adjusted_cost",
]

METRIC_FIELDNAMES = [
    "subject",
    "actor",
    "variant",
    "reference_source",
    "n_sessions",
    "cost_scale",
    "cost_threshold",
    "start_cost",
    "end_cost",
    "gap_penalty",
    "pairwise_f1",
    "pairwise_precision",
    "pairwise_recall",
    "complete_track_f1",
    "complete_tracks",
    "mean_track_length",
    "seed_session",
    "reference_seed_rois",
    "evaluated_prediction_tracks",
    "dropped_prediction_tracks",
]


@dataclass(frozen=True)
class Track2pTeacherDebugConfig:
    """Configuration for Track2p teacher/debug-oracle exports."""

    benchmark: Track2pBenchmarkConfig
    track2p_reference: Path | None = None
    output_dir: Path = Path("track2p_teacher_debug")
    details_output: Path | None = None
    summary_output: Path | None = None
    metrics_output: Path | None = None
    cost_scale: float = 1.0
    edge_scope: EdgeScope = "solver"


@dataclass(frozen=True)
class Track2pTeacherDebugReport:
    """In-memory result of a teacher/debug export."""

    detail_rows: tuple[CSVRow, ...]
    summary_rows: tuple[CSVRow, ...]
    metric_rows: tuple[CSVRow, ...]


@dataclass(frozen=True)
class _BayesEdgeCostLookup:
    """Suite2p-indexed lookup wrapper around Bayes pairwise cost matrices."""

    pairwise_costs: Mapping[tuple[int, int], np.ndarray]
    suite2p_to_loaded: tuple[dict[int, int], ...]
    loaded_to_suite2p: tuple[np.ndarray, ...]
    gap_penalty: float
    cost_threshold: float | None

    @classmethod
    def from_sessions(
        cls,
        sessions: Sequence[Track2pSession],
        pairwise_costs: Mapping[tuple[int, int], np.ndarray],
        *,
        gap_penalty: float,
        cost_threshold: float | None,
    ) -> "_BayesEdgeCostLookup":
        loaded_to_suite2p = tuple(_suite2p_roi_indices_for_session(session) for session in sessions)
        suite2p_to_loaded = tuple(
            {int(suite2p_index): int(loaded_index) for loaded_index, suite2p_index in enumerate(indices)}
            for indices in loaded_to_suite2p
        )
        return cls(
            pairwise_costs=pairwise_costs,
            suite2p_to_loaded=suite2p_to_loaded,
            loaded_to_suite2p=loaded_to_suite2p,
            gap_penalty=float(gap_penalty),
            cost_threshold=cost_threshold,
        )

    def lookup(self, session_a: int, session_b: int, roi_a: int, roi_b: int) -> CSVRow:
        """Return Bayes cost/rank diagnostics for one Suite2p-indexed pair."""

        edge = (int(session_a), int(session_b))
        if edge not in self.pairwise_costs:
            return _missing_cost_row("edge_not_in_bayes_cost_graph")

        loaded_a = self.suite2p_to_loaded[session_a].get(int(roi_a))
        loaded_b = self.suite2p_to_loaded[session_b].get(int(roi_b))
        if loaded_a is None:
            return _missing_cost_row("roi_a_not_loaded")
        if loaded_b is None:
            return _missing_cost_row("roi_b_not_loaded")

        cost_matrix = np.asarray(self.pairwise_costs[edge], dtype=float)
        if loaded_a >= cost_matrix.shape[0] or loaded_b >= cost_matrix.shape[1]:
            return _missing_cost_row("loaded_roi_out_of_bounds")

        gap = session_b - session_a
        adjustment = self.gap_penalty * max(0, gap - 1)
        adjusted = cost_matrix + adjustment
        cost = float(cost_matrix[loaded_a, loaded_b])
        adjusted_cost = float(adjusted[loaded_a, loaded_b])
        candidate_present = bool(np.isfinite(adjusted_cost))
        eligible = bool(candidate_present and (self.cost_threshold is None or adjusted_cost <= self.cost_threshold))

        row = adjusted[loaded_a, :]
        col = adjusted[:, loaded_b]
        row_rank = _rank_1_based(row, loaded_b)
        col_rank = _rank_1_based(col, loaded_a)
        finite_row = np.isfinite(row)
        if np.any(finite_row):
            best_loaded_target = int(np.nanargmin(np.where(finite_row, row, np.inf)))
            best_cost = float(row[best_loaded_target])
            best_suite2p = int(self.loaded_to_suite2p[session_b][best_loaded_target])
            margin = float(adjusted_cost - best_cost) if candidate_present else None
        else:
            best_suite2p = None
            best_cost = None
            margin = None

        return {
            "bayes_candidate_present": candidate_present,
            "bayes_eligible_after_threshold": eligible,
            "bayes_cost": cost if np.isfinite(cost) else None,
            "bayes_adjusted_cost": adjusted_cost if candidate_present else None,
            "bayes_row_rank": row_rank,
            "bayes_col_rank": col_rank,
            "bayes_mutual_top1": bool(row_rank == 1 and col_rank == 1),
            "bayes_best_target_suite2p": best_suite2p,
            "bayes_best_adjusted_cost": best_cost,
            "bayes_margin_to_best": margin,
            "bayes_missing_reason": "" if candidate_present else "nonfinite_cost",
        }


def run_track2p_teacher_debug(config: Track2pTeacherDebugConfig) -> Track2pTeacherDebugReport:
    """Run teacher/debug diagnostics and write the configured CSV outputs."""

    benchmark = config.benchmark
    if benchmark.method != "global-assignment":
        raise ValueError("Track2p teacher/debug diagnostics require method='global-assignment'")
    if benchmark.split != "subject":
        raise ValueError("Track2p teacher/debug diagnostics currently support split='subject' only")
    if benchmark.cost == "calibrated":
        raise ValueError("cost='calibrated' needs LOSO training and is not supported by this diagnostic")
    if config.cost_scale <= 0.0 or not np.isfinite(config.cost_scale):
        raise ValueError("cost_scale must be a positive finite number")

    subject_dirs = discover_subject_dirs(benchmark.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {benchmark.data}")

    detail_rows: list[CSVRow] = []
    metric_rows: list[CSVRow] = []
    progress = ProgressReporter(len(subject_dirs), enabled=benchmark.progress, label="teacher-debug")
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        manual_reference = _load_reference_for_subject(subject_dir, data_root=benchmark.data, config=benchmark)
        _validate_reference_for_benchmark(manual_reference, subject_dir=subject_dir, config=benchmark)
        if manual_reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError("Teacher/debug diagnostics require independent manual ground truth")

        sessions = _load_subject_sessions(subject_dir, benchmark)
        _validate_reference_roi_indices(manual_reference, sessions)
        track2p_reference = _load_track2p_teacher_reference(subject_dir, data_root=benchmark.data, config=config)
        _validate_teacher_reference(track2p_reference, manual_reference, subject=subject_dir.name)

        base_costs = build_registered_pairwise_costs(
            sessions,
            max_gap=benchmark.max_gap,
            cost=benchmark.cost,
            transform_type=benchmark.transform_type,
            order=benchmark.order,
            weighted_centroids=benchmark.weighted_centroids,
            velocity_variance=benchmark.velocity_variance,
            regularization=benchmark.regularization,
            pairwise_cost_kwargs=benchmark.pairwise_cost_kwargs,
        )
        scaled_costs = _scale_pairwise_costs(base_costs, config.cost_scale)
        solver_result = _load_pyrecest_multisession_solver()(
            scaled_costs,
            session_sizes=tuple(int(session.plane_data.n_rois) for session in sessions),
            start_cost=benchmark.start_cost,
            end_cost=benchmark.end_cost,
            gap_penalty=benchmark.gap_penalty,
            cost_threshold=benchmark.cost_threshold,
        )
        bayes_matrix = tracks_to_suite2p_index_matrix(solver_result.tracks, sessions)
        track2p_matrix = normalize_track_matrix(track2p_reference.suite2p_indices)
        manual_matrix = _reference_matrix(manual_reference, curated_only=benchmark.curated_only)

        metric_rows.extend(
            _subject_metric_rows(
                subject=subject_dir.name,
                manual_reference=manual_reference,
                track2p_matrix=track2p_matrix,
                bayes_matrix=bayes_matrix,
                benchmark=benchmark,
                cost_scale=config.cost_scale,
            )
        )

        lookup = _BayesEdgeCostLookup.from_sessions(
            sessions,
            scaled_costs,
            gap_penalty=benchmark.gap_penalty,
            cost_threshold=benchmark.cost_threshold,
        )
        detail_rows.extend(
            _subject_detail_rows(
                subject=subject_dir.name,
                session_names=manual_reference.session_names,
                manual_matrix=manual_matrix,
                track2p_matrix=track2p_matrix,
                bayes_matrix=bayes_matrix,
                benchmark=benchmark,
                config=config,
                lookup=lookup,
            )
        )

    summary_rows = _summary_rows(detail_rows)
    report = Track2pTeacherDebugReport(tuple(detail_rows), tuple(summary_rows), tuple(metric_rows))
    write_teacher_debug_report(report, config)
    return report


def write_teacher_debug_report(report: Track2pTeacherDebugReport, config: Track2pTeacherDebugConfig) -> dict[str, Path]:
    """Write detail, summary, and actor metric CSV files."""

    output_dir = Path(config.output_dir)
    details_path = config.details_output or output_dir / "teacher_disagreement_edges.csv"
    summary_path = config.summary_output or output_dir / "teacher_disagreement_summary.csv"
    metrics_path = config.metrics_output or output_dir / "teacher_actor_metrics.csv"
    _write_csv(details_path, report.detail_rows, DETAIL_FIELDNAMES)
    _write_csv(summary_path, report.summary_rows, SUMMARY_FIELDNAMES)
    _write_csv(metrics_path, report.metric_rows, METRIC_FIELDNAMES)
    return {"details": details_path, "summary": summary_path, "metrics": metrics_path}


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for teacher/debug diagnostics."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-teacher-debug",
        description="Export Bayes/Track2p/manual-GT disagreement diagnostics.",
    )
    parser.add_argument("--data", required=True, type=Path, help="Track2p dataset root or one subject directory")
    parser.add_argument("--plane", dest="plane_name", default="plane0", help="Plane name such as plane0")
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--reference", type=Path, default=None, help="Manual-GT CSV/root for the benchmark reference")
    parser.add_argument("--reference-kind", default="manual-gt", choices=("auto", "manual-gt"))
    parser.add_argument("--track2p-reference", type=Path, default=None, help="Optional Track2p output root/folder used as teacher")
    parser.add_argument("--curated-only", action="store_true", help="Evaluate only curated manual-GT reference tracks")
    parser.add_argument("--seed-session", type=int, default=0, help="Seed session used by benchmark scoring")
    parser.add_argument("--restrict-to-reference-seed-rois", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cost", default="registered-iou", choices=("registered-iou", "roi-aware"))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
        help="Registration transform used for Bayes costs",
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument("--cost-scale", type=float, default=1.0, help="Multiplier applied to Bayes pairwise costs before solving")
    parser.add_argument("--edge-scope", choices=("solver", "consecutive", "all-pairs"), default="solver")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument("--exclude-overlapping-pixels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("track2p_teacher_debug"))
    parser.add_argument("--details-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    report = run_track2p_teacher_debug(config)
    paths = write_teacher_debug_report(report, config)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


def _config_from_args(args: argparse.Namespace) -> Track2pTeacherDebugConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    benchmark = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="subject",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=cast(Any, args.reference_kind),
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=cast(AssociationCost, args.cost),
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
    return Track2pTeacherDebugConfig(
        benchmark=benchmark,
        track2p_reference=args.track2p_reference,
        output_dir=args.output_dir,
        details_output=args.details_output,
        summary_output=args.summary_output,
        metrics_output=args.metrics_output,
        cost_scale=args.cost_scale,
        edge_scope=cast(EdgeScope, args.edge_scope),
    )


def _load_track2p_teacher_reference(subject_dir: Path, *, data_root: Path, config: Track2pTeacherDebugConfig) -> Track2pReference:
    reference_root = config.track2p_reference
    if reference_root is None:
        reference_root = subject_dir / "track2p"
    reference_path = _resolve_track2p_reference_path(subject_dir, data_root=data_root, reference_root=reference_root)
    if reference_path is None:
        raise FileNotFoundError(f"Could not resolve Track2p teacher output for subject {subject_dir.name!r} under {reference_root}")
    return load_track2p_reference(reference_path, plane_name=config.benchmark.plane_name)


def _validate_teacher_reference(teacher: Track2pReference, manual: Track2pReference, *, subject: str) -> None:
    if teacher.n_sessions != manual.n_sessions:
        raise ValueError(f"Subject {subject!r}: Track2p teacher has {teacher.n_sessions} sessions, manual GT has {manual.n_sessions}")
    if teacher.session_names != manual.session_names:
        raise ValueError(f"Subject {subject!r}: Track2p teacher session order does not match manual GT")


def _subject_metric_rows(
    *,
    subject: str,
    manual_reference: Track2pReference,
    track2p_matrix: np.ndarray,
    bayes_matrix: np.ndarray,
    benchmark: Track2pBenchmarkConfig,
    cost_scale: float,
) -> list[CSVRow]:
    rows: list[CSVRow] = []
    for actor, matrix, variant in (
        ("track2p", track2p_matrix, "Track2p teacher output"),
        ("bayescatrack", bayes_matrix, "BayesCaTrack global assignment"),
    ):
        scores = _score_prediction_against_reference(matrix, manual_reference, config=benchmark)
        row: CSVRow = {
            "subject": subject,
            "actor": actor,
            "variant": variant,
            "reference_source": manual_reference.source,
            "n_sessions": manual_reference.n_sessions,
            "cost_scale": float(cost_scale),
            "cost_threshold": _threshold_label(benchmark.cost_threshold),
            "start_cost": float(benchmark.start_cost),
            "end_cost": float(benchmark.end_cost),
            "gap_penalty": float(benchmark.gap_penalty),
        }
        for key in METRIC_FIELDNAMES:
            if key in scores:
                row[key] = cast(CSVValue, scores[key])
        rows.append(row)
    return rows


def _subject_detail_rows(
    *,
    subject: str,
    session_names: Sequence[str],
    manual_matrix: np.ndarray,
    track2p_matrix: np.ndarray,
    bayes_matrix: np.ndarray,
    benchmark: Track2pBenchmarkConfig,
    config: Track2pTeacherDebugConfig,
    lookup: _BayesEdgeCostLookup,
) -> list[CSVRow]:
    edges = _edge_pairs(manual_matrix.shape[1], max_gap=benchmark.max_gap, scope=config.edge_scope)
    manual_edges = _track_matrix_edge_set(manual_matrix, edges=edges)
    track2p_edges = _track_matrix_edge_set(track2p_matrix, edges=edges)
    bayes_edges = _track_matrix_edge_set(bayes_matrix, edges=edges)
    all_edges = sorted(manual_edges | track2p_edges | bayes_edges)

    rows: list[CSVRow] = []
    for session_a, session_b, roi_a, roi_b in all_edges:
        manual = (session_a, session_b, roi_a, roi_b) in manual_edges
        teacher = (session_a, session_b, roi_a, roi_b) in track2p_edges
        bayes = (session_a, session_b, roi_a, roi_b) in bayes_edges
        row: CSVRow = {
            "subject": subject,
            "session_a": int(session_a),
            "session_b": int(session_b),
            "session_a_name": session_names[session_a],
            "session_b_name": session_names[session_b],
            "gap": int(session_b - session_a),
            "roi_a": int(roi_a),
            "roi_b": int(roi_b),
            "category": classify_teacher_edge(manual, teacher, bayes),
            "manual_gt_label": manual,
            "track2p_label": teacher,
            "bayescatrack_label": bayes,
            "edge_source": _edge_source_label(manual, teacher, bayes),
            "cost_scale": float(config.cost_scale),
            "cost_threshold": _threshold_label(benchmark.cost_threshold),
            "gap_penalty": float(benchmark.gap_penalty),
        }
        row.update(lookup.lookup(session_a, session_b, roi_a, roi_b))
        rows.append(row)
    return rows


def classify_teacher_edge(manual_gt: bool, track2p: bool, bayescatrack: bool) -> str:
    """Return the 2x2x2 disagreement bucket for one edge."""

    if manual_gt and track2p and bayescatrack:
        return "all_agree_manual_positive"
    if manual_gt and track2p and not bayescatrack:
        return "bayes_missed_teacher_edge"
    if manual_gt and not track2p and bayescatrack:
        return "bayes_found_track2p_missed_edge"
    if manual_gt and not track2p and not bayescatrack:
        return "both_missed_manual_edge"
    if not manual_gt and track2p and not bayescatrack:
        return "track2p_false_positive_bayes_rejected"
    if not manual_gt and not track2p and bayescatrack:
        return "bayes_hard_false_positive"
    if not manual_gt and track2p and bayescatrack:
        return "teacher_and_bayes_false_positive"
    return "unobserved_true_negative"


def _edge_pairs(n_sessions: int, *, max_gap: int, scope: EdgeScope) -> tuple[tuple[int, int], ...]:
    if scope == "solver":
        return session_edge_pairs(n_sessions, max_gap=max_gap)
    if scope == "consecutive":
        return tuple((idx, idx + 1) for idx in range(max(0, n_sessions - 1)))
    if scope == "all-pairs":
        return tuple((i, j) for i in range(max(0, n_sessions - 1)) for j in range(i + 1, n_sessions))
    raise ValueError(f"Unsupported edge scope: {scope!r}")


def _track_matrix_edge_set(track_matrix: np.ndarray, *, edges: Sequence[tuple[int, int]]) -> set[EdgeKey]:
    matrix = normalize_track_matrix(track_matrix)
    edge_set: set[EdgeKey] = set()
    for row in matrix:
        for session_a, session_b in edges:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if _valid_roi(roi_a) and _valid_roi(roi_b):
                edge_set.add((int(session_a), int(session_b), int(roi_a), int(roi_b)))
    return edge_set


def _summary_rows(detail_rows: Sequence[CSVRow]) -> tuple[CSVRow, ...]:
    groups: dict[tuple[Any, ...], list[CSVRow]] = defaultdict(list)
    for row in detail_rows:
        groups[("dataset", "", "", "", "", "", "", row["category"])].append(row)
        groups[("subject", row["subject"], "", "", "", "", "", row["category"])].append(row)
        groups[
            (
                "session_edge",
                row["subject"],
                row["session_a"],
                row["session_b"],
                row["session_a_name"],
                row["session_b_name"],
                row["gap"],
                row["category"],
            )
        ].append(row)

    summary: list[CSVRow] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        scope, subject, session_a, session_b, session_a_name, session_b_name, gap, category = key
        summary.append(
            {
                "scope": scope,
                "subject": subject,
                "session_a": session_a,
                "session_b": session_b,
                "session_a_name": session_a_name,
                "session_b_name": session_b_name,
                "gap": gap,
                "category": category,
                "count": len(rows),
                "manual_gt_label": _uniform_or_blank(row["manual_gt_label"] for row in rows),
                "track2p_label": _uniform_or_blank(row["track2p_label"] for row in rows),
                "bayescatrack_label": _uniform_or_blank(row["bayescatrack_label"] for row in rows),
                "candidate_present_rate": _mean_bool(row.get("bayes_candidate_present") for row in rows),
                "eligible_after_threshold_rate": _mean_bool(row.get("bayes_eligible_after_threshold") for row in rows),
                "median_bayes_row_rank": _median_numeric(row.get("bayes_row_rank") for row in rows),
                "median_bayes_adjusted_cost": _median_numeric(row.get("bayes_adjusted_cost") for row in rows),
            }
        )
    return tuple(summary)


def _scale_pairwise_costs(pairwise_costs: Mapping[tuple[int, int], np.ndarray], scale: float) -> dict[tuple[int, int], np.ndarray]:
    return {edge: np.asarray(costs, dtype=float) * float(scale) for edge, costs in pairwise_costs.items()}


def _suite2p_roi_indices_for_session(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int).reshape(-1)
    return np.arange(plane.n_rois, dtype=int)


def _missing_cost_row(reason: str) -> CSVRow:
    return {
        "bayes_candidate_present": False,
        "bayes_eligible_after_threshold": False,
        "bayes_cost": None,
        "bayes_adjusted_cost": None,
        "bayes_row_rank": None,
        "bayes_col_rank": None,
        "bayes_mutual_top1": False,
        "bayes_best_target_suite2p": None,
        "bayes_best_adjusted_cost": None,
        "bayes_margin_to_best": None,
        "bayes_missing_reason": reason,
    }


def _rank_1_based(values: np.ndarray, index: int) -> int | None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if index < 0 or index >= values.shape[0] or not np.isfinite(values[index]):
        return None
    return int(1 + np.sum(values < values[index]))


def _edge_source_label(manual: bool, track2p: bool, bayes: bool) -> str:
    sources = []
    if manual:
        sources.append("manual_gt")
    if track2p:
        sources.append("track2p")
    if bayes:
        sources.append("bayescatrack")
    return "+".join(sources)


def _valid_roi(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(cast(Any, value)) >= 0
    except (TypeError, ValueError):
        return False


def _threshold_label(threshold: float | None) -> str | float:
    return "none" if threshold is None else float(threshold)


def _mean_bool(values: Iterable[CSVValue]) -> float | None:
    bool_values = [bool(value) for value in values if value is not None]
    if not bool_values:
        return None
    return float(np.mean(bool_values))


def _median_numeric(values: Iterable[CSVValue]) -> float | None:
    numeric_values = [float(value) for value in values if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))]
    if not numeric_values:
        return None
    return float(np.median(np.asarray(numeric_values, dtype=float)))


def _uniform_or_blank(values: Iterable[CSVValue]) -> CSVValue:
    unique = {value for value in values}
    if len(unique) == 1:
        return next(iter(unique))
    return ""


def _write_csv(path: Path, rows: Sequence[CSVRow], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
