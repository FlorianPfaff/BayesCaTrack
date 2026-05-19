"""Activity tie-breaker sweeps for Track2p global-assignment benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    build_registered_pairwise_costs,
    solve_global_assignment_from_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    _variant_name,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_cost_sweep import (
    _format_value,
    _pairwise_cost_statistics,
    _parse_nonnegative_values,
    _threshold_label,
)

# pylint: disable=protected-access,too-many-locals

_ACTIVITY_COST_COMPONENTS = (
    "activity_tiebreaker_cost",
    "fluorescence_similarity_cost",
    "spike_similarity_cost",
    "trace_std_absdiff",
    "trace_skew_absdiff",
    "event_rate_absdiff",
    "neuropil_ratio_absdiff",
)
_TRACE_SOURCES = ("auto", "traces", "spike_traces", "neuropil_traces")


@dataclass(frozen=True)
class ActivityTieBreakerSweepConfig:
    """Configuration for a Track2p activity tie-breaker sweep."""

    benchmark: Track2pBenchmarkConfig
    activity_tie_breaker_weights: tuple[float, ...]
    activity_tie_breaker_component: str = "activity_tiebreaker_cost"
    activity_trace_source: str = "auto"
    activity_event_threshold: float = 0.0


@dataclass(frozen=True)
class ActivityTieBreakerSweepRun:
    """One activity tie-breaker setting."""

    weight: float
    sweep_index: int
    sweep_count: int


def run_track2p_activity_tie_breaker_sweep(
    config: ActivityTieBreakerSweepConfig,
) -> list[SubjectBenchmarkResult]:
    """Run all configured activity tie-breaker benchmark rows."""

    return list(iter_track2p_activity_tie_breaker_sweep(config))


def iter_track2p_activity_tie_breaker_sweep(
    config: ActivityTieBreakerSweepConfig,
) -> Iterator[SubjectBenchmarkResult]:
    """Yield Track2p global-assignment rows over weak activity tie-breaker weights."""

    benchmark = config.benchmark
    if benchmark.method != "global-assignment":
        raise ValueError("Activity tie-breaker sweeps require method='global-assignment'")
    if benchmark.split != "subject":
        raise ValueError("Activity tie-breaker sweeps currently support split='subject' only")
    if benchmark.cost == "calibrated":
        raise ValueError(
            "cost='calibrated' requires LOSO training and is not supported by this sweep"
        )

    runs = _sweep_runs(config.activity_tie_breaker_weights)
    subject_dirs = discover_subject_dirs(benchmark.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {benchmark.data}"
        )

    progress = ProgressReporter(
        len(subject_dirs) * len(runs),
        enabled=benchmark.progress,
        label="activity-sweep",
    )
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=benchmark.data, config=benchmark
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=benchmark
        )
        sessions = _load_subject_sessions(subject_dir, benchmark)
        if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
            _validate_reference_roi_indices(reference, sessions)
        session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)

        for run in runs:
            progress.step(
                f"running {subject_dir.name} activity_weight={run.weight:g}"
            )
            pairwise_costs = _build_activity_pairwise_costs(
                sessions,
                benchmark,
                activity_tie_breaker_weight=run.weight,
                activity_tie_breaker_component=config.activity_tie_breaker_component,
                activity_trace_source=config.activity_trace_source,
                activity_event_threshold=config.activity_event_threshold,
            )
            assignment = solve_global_assignment_from_pairwise_costs(
                pairwise_costs,
                session_sizes=session_sizes,
                start_cost=benchmark.start_cost,
                end_cost=benchmark.end_cost,
                gap_penalty=benchmark.gap_penalty,
                cost_threshold=benchmark.cost_threshold,
            )
            predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
            scores: dict[str, float | int | str] = dict(
                _score_prediction_against_reference(predicted, reference, config=benchmark)
            )
            scores = {
                **scores,
                "sweep_index": int(run.sweep_index),
                "sweep_count": int(run.sweep_count),
                "activity_tie_breaker_weight": float(run.weight),
                "activity_tie_breaker_component": config.activity_tie_breaker_component,
                "activity_trace_source": config.activity_trace_source,
                "activity_event_threshold": float(config.activity_event_threshold),
                "cost_threshold": _threshold_label(benchmark.cost_threshold),
                "start_cost": float(benchmark.start_cost),
                "end_cost": float(benchmark.end_cost),
                "gap_penalty": float(benchmark.gap_penalty),
                **_pairwise_cost_statistics(pairwise_costs, benchmark.cost_threshold),
            }
            yield SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=_activity_variant_name(benchmark.cost, run.weight),
                method=benchmark.method,
                scores=scores,
                n_sessions=reference.n_sessions,
                reference_source=reference.source,
            )


def _build_activity_pairwise_costs(
    sessions: Sequence[Any],
    benchmark: Track2pBenchmarkConfig,
    *,
    activity_tie_breaker_weight: float,
    activity_tie_breaker_component: str,
    activity_trace_source: str,
    activity_event_threshold: float,
) -> dict[tuple[int, int], np.ndarray]:
    return build_registered_pairwise_costs(
        sessions,
        max_gap=benchmark.max_gap,
        cost=benchmark.cost,
        transform_type=benchmark.transform_type,
        order=benchmark.order,
        weighted_centroids=benchmark.weighted_centroids,
        velocity_variance=benchmark.velocity_variance,
        regularization=benchmark.regularization,
        pairwise_cost_kwargs=benchmark.pairwise_cost_kwargs,
        activity_tie_breaker_weight=activity_tie_breaker_weight,
        activity_tie_breaker_component=activity_tie_breaker_component,
        activity_trace_source=activity_trace_source,
        activity_event_threshold=activity_event_threshold,
    )


def _sweep_runs(weights: Sequence[float]) -> tuple[ActivityTieBreakerSweepRun, ...]:
    normalised = _normalise_activity_weights(weights)
    return tuple(
        ActivityTieBreakerSweepRun(
            weight=weight,
            sweep_index=index + 1,
            sweep_count=len(normalised),
        )
        for index, weight in enumerate(normalised)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-activity-tie-breaker-sweep",
        description="Sweep weak activity tie-breaker weights for Track2p global-assignment benchmarks.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
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
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=(
            "registered-iou",
            "registered-soft-iou",
            "registered-shifted-iou",
            "roi-aware",
            "roi-aware-shifted",
        ),
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-affine", "fov-translation", "none"),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument(
        "--no-cost-threshold",
        action="store_true",
        help="Disable the solver edge-cost threshold",
    )
    parser.add_argument(
        "--activity-tie-breaker-weights",
        "--activity-weight-sweep",
        dest="activity_tie_breaker_weights",
        default="0,0.01,0.03,0.1,0.3",
        help="Comma-separated non-negative activity tie-breaker weights.",
    )
    parser.add_argument(
        "--activity-tie-breaker-component",
        default="activity_tiebreaker_cost",
        choices=_ACTIVITY_COST_COMPONENTS,
        help="Pairwise activity component added as a weak cost plane.",
    )
    parser.add_argument(
        "--activity-trace-source",
        default="auto",
        choices=_TRACE_SOURCES,
        help="Trace source used by the activity component extractor.",
    )
    parser.add_argument(
        "--activity-event-threshold",
        type=float,
        default=0.0,
        help="Spike/event threshold for event-rate activity features.",
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument(
        "--write-incrementally",
        action="store_true",
        help="Write CSV rows to --output as each activity weight completes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    if args.write_incrementally:
        if args.output is None:
            parser.error("--write-incrementally requires --output")
        if args.format != "csv":
            parser.error("--write-incrementally currently supports --format csv only")
        write_activity_sweep_results_incrementally(
            iter_track2p_activity_tie_breaker_sweep(config), args.output
        )
        return 0

    rows = [
        result.to_dict()
        for result in run_track2p_activity_tie_breaker_sweep(config)
    ]
    if args.output is not None:
        write_activity_sweep_results(rows, args.output, args.format)
    else:
        _write_activity_sweep_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> ActivityTieBreakerSweepConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    benchmark = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
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
    if args.activity_event_threshold < 0.0 or not np.isfinite(args.activity_event_threshold):
        raise ValueError("--activity-event-threshold must be non-negative and finite")
    return ActivityTieBreakerSweepConfig(
        benchmark=benchmark,
        activity_tie_breaker_weights=_parse_nonnegative_values(
            args.activity_tie_breaker_weights,
            name="--activity-tie-breaker-weights",
        ),
        activity_tie_breaker_component=args.activity_tie_breaker_component,
        activity_trace_source=args.activity_trace_source,
        activity_event_threshold=float(args.activity_event_threshold),
    )


def write_activity_sweep_results(
    rows: Sequence[dict[str, float | int | str]], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_activity_sweep_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_activity_sweep_table(rows) + "\n", encoding="utf-8")


def write_activity_sweep_results_incrementally(
    results: Iterable[SubjectBenchmarkResult], output_path: Path
) -> int:
    """Write CSV rows as each activity setting completes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        fieldnames: list[str] = []
        for result in results:
            row = result.to_dict()
            if writer is None:
                fieldnames = _activity_sweep_fieldnames([row])
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            unexpected = sorted(set(row) - set(fieldnames))
            if unexpected:
                raise ValueError(
                    "Incremental CSV rows introduced unexpected fields: "
                    + ", ".join(unexpected)
                )
            writer.writerow(row)
            handle.flush()
            rows_written += 1
    return rows_written


def _write_activity_sweep_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: str
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_activity_sweep_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_activity_sweep_table(rows))


def format_activity_sweep_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    columns = [
        "subject",
        "activity_tie_breaker_weight",
        "activity_tie_breaker_component",
        "activity_trace_source",
        "cost_threshold",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "cost_median",
        "cost_p90",
        "cost_threshold_admitted_fraction",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def _activity_sweep_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "sweep_index",
        "sweep_count",
        "activity_tie_breaker_weight",
        "activity_tie_breaker_component",
        "activity_trace_source",
        "activity_event_threshold",
        "cost_threshold",
        "start_cost",
        "end_cost",
        "gap_penalty",
        "cost_edges",
        "cost_values",
        "cost_finite_values",
        "cost_nonfinite_values",
        "cost_min",
        "cost_p10",
        "cost_median",
        "cost_p90",
        "cost_max",
        "cost_threshold_admitted_count",
        "cost_threshold_admitted_fraction",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _activity_variant_name(cost: AssociationCost, weight: float) -> str:
    base = _variant_name(cost)
    if weight <= 0.0:
        return base
    return f"{base} + activity tie-breaker {weight:g}"


def _normalise_activity_weights(values: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if not weights:
        raise ValueError("At least one activity tie-breaker weight is required")
    if any((not np.isfinite(weight)) or weight < 0.0 for weight in weights):
        raise ValueError("Activity tie-breaker weights must be non-negative finite numbers")
    return weights


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
