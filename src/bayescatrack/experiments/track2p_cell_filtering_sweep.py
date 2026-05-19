"""Suite2p cell-filtering sensitivity sweeps for Track2p benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _predict_subject_tracks,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)

# pylint: disable=protected-access,too-many-locals

_FILTERED_MODE = "filtered"
_INCLUDE_NON_CELLS_MODE = "include-non-cells"
_DEFAULT_CELL_PROBABILITY_THRESHOLDS = (0.0, 0.25, 0.5, 0.75)
_DEFAULT_CELL_PROBABILITY_WEIGHTS = (0.0,)


@dataclass(frozen=True)
class CellFilteringSweepConfig:
    """Configuration for a Suite2p cell-filtering sensitivity sweep."""

    benchmark: Track2pBenchmarkConfig
    cell_probability_thresholds: tuple[float, ...] = _DEFAULT_CELL_PROBABILITY_THRESHOLDS
    filter_modes: tuple[str, ...] = (_FILTERED_MODE, _INCLUDE_NON_CELLS_MODE)
    cell_probability_weights: tuple[float, ...] = _DEFAULT_CELL_PROBABILITY_WEIGHTS
    fail_fast: bool = False


@dataclass(frozen=True)
class CellFilteringSweepRun:
    """One Suite2p cell-filtering setting."""

    include_non_cells: bool
    cell_probability_threshold: float
    cell_probability_weight: float
    sweep_index: int
    sweep_count: int


def run_track2p_cell_filtering_sweep(
    config: CellFilteringSweepConfig,
) -> list[SubjectBenchmarkResult]:
    """Run all configured Suite2p cell-filtering benchmark rows."""

    return list(iter_track2p_cell_filtering_sweep(config))


def iter_track2p_cell_filtering_sweep(
    config: CellFilteringSweepConfig,
) -> Iterator[SubjectBenchmarkResult]:
    """Yield benchmark rows over Suite2p cell-probability filtering settings.

    Unlike :func:`run_track2p_benchmark`, this sweep records filtering failures
    as rows instead of aborting the whole run. That makes it suitable for finding
    whether manual-GT ROIs disappear before association because of Suite2p
    ``iscell.npy`` filtering.
    """

    benchmark = config.benchmark
    runs = _sweep_runs(config)
    subject_dirs = discover_subject_dirs(benchmark.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {benchmark.data}"
        )

    progress = ProgressReporter(
        len(subject_dirs) * len(runs),
        enabled=benchmark.progress,
        label="cell-filter-sweep",
    )
    for subject_dir in subject_dirs:
        for run in runs:
            progress.step(
                f"running {subject_dir.name} {_filter_policy(run)}"
            )
            yield _run_subject_cell_filtering_setting(
                subject_dir,
                benchmark=benchmark,
                run=run,
                data_root=benchmark.data,
                fail_fast=config.fail_fast,
            )


def _run_subject_cell_filtering_setting(
    subject_dir: Path,
    *,
    benchmark: Track2pBenchmarkConfig,
    run: CellFilteringSweepRun,
    data_root: Path,
    fail_fast: bool,
) -> SubjectBenchmarkResult:
    run_config = _benchmark_for_run(benchmark, run)
    reference_source = "unavailable"
    n_sessions = 0
    try:
        reference = _load_reference_for_subject(
            subject_dir, data_root=data_root, config=run_config
        )
        reference_source = reference.source
        n_sessions = reference.n_sessions
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=run_config
        )
        sessions = _load_subject_sessions(subject_dir, run_config)
        if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
            _validate_reference_roi_indices(reference, sessions)
        predicted, variant = _predict_subject_tracks(
            subject_dir, run_config, reference=reference
        )
        scores: dict[str, float | int | str] = dict(
            _score_prediction_against_reference(predicted, reference, config=run_config)
        )
        scores.update(_run_metadata(run, status="ok"))
        return SubjectBenchmarkResult(
            subject=subject_dir.name,
            variant=f"{variant} [{_filter_policy(run)}]",
            method=run_config.method,
            scores=scores,
            n_sessions=reference.n_sessions,
            reference_source=reference.source,
        )
    except (IndexError, ValueError) as exc:
        if fail_fast:
            raise
        scores = _run_metadata(run, status="error")
        scores.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return SubjectBenchmarkResult(
            subject=subject_dir.name,
            variant=f"Filtering failed [{_filter_policy(run)}]",
            method=run_config.method,
            scores=scores,
            n_sessions=n_sessions,
            reference_source=reference_source,
        )


def _benchmark_for_run(
    benchmark: Track2pBenchmarkConfig,
    run: CellFilteringSweepRun,
) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = dict(benchmark.pairwise_cost_kwargs or {})
    if run.cell_probability_weight > 0.0 or "cell_probability_weight" in pairwise_cost_kwargs:
        pairwise_cost_kwargs["cell_probability_weight"] = float(
            run.cell_probability_weight
        )
    return replace(
        benchmark,
        include_non_cells=run.include_non_cells,
        cell_probability_threshold=run.cell_probability_threshold,
        pairwise_cost_kwargs=pairwise_cost_kwargs or None,
    )


def _run_metadata(
    run: CellFilteringSweepRun, *, status: str
) -> dict[str, float | int | str]:
    return {
        "sweep_index": int(run.sweep_index),
        "sweep_count": int(run.sweep_count),
        "status": status,
        "suite2p_filter_policy": _filter_policy(run),
        "include_non_cells": _format_bool(run.include_non_cells),
        "cell_probability_threshold": float(run.cell_probability_threshold),
        "cell_probability_weight": float(run.cell_probability_weight),
    }


def _filter_policy(run: CellFilteringSweepRun) -> str:
    if run.include_non_cells:
        return "include-non-cells"
    return f"iscell>={run.cell_probability_threshold:g}"


def _sweep_runs(config: CellFilteringSweepConfig) -> tuple[CellFilteringSweepRun, ...]:
    thresholds = _normalise_probability_values(config.cell_probability_thresholds)
    weights = _normalise_nonnegative_values(
        config.cell_probability_weights, name="cell probability weights"
    )
    modes = _normalise_filter_modes(config.filter_modes)
    runs: list[CellFilteringSweepRun] = []
    for mode in modes:
        include_non_cells = mode == _INCLUDE_NON_CELLS_MODE
        for threshold in thresholds:
            for weight in weights:
                runs.append(
                    CellFilteringSweepRun(
                        include_non_cells=include_non_cells,
                        cell_probability_threshold=threshold,
                        cell_probability_weight=weight,
                        sweep_index=0,
                        sweep_count=0,
                    )
                )
    if not runs:
        raise ValueError("At least one cell-filtering sweep setting is required")
    sweep_count = len(runs)
    return tuple(
        replace(run, sweep_index=index + 1, sweep_count=sweep_count)
        for index, run in enumerate(runs)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-cell-filtering-sweep",
        description="Sweep Suite2p iscell filtering and cell-probability thresholds for Track2p benchmarks.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--method",
        default="global-assignment",
        choices=("track2p-baseline", "global-assignment", "oracle-gt-links"),
        help="Benchmark variant to run for every filtering setting",
    )
    parser.add_argument(
        "--split",
        default="subject",
        choices=("subject", "leave-one-subject-out"),
        help="Evaluation split policy",
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
            "calibrated",
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
        "--cell-probability-thresholds",
        default="0,0.25,0.5,0.75",
        help="Comma-separated Suite2p iscell probability thresholds to test when filtering cells.",
    )
    parser.add_argument(
        "--filter-modes",
        default="filtered,include-non-cells",
        help="Comma-separated modes: filtered, include-non-cells.",
    )
    parser.add_argument(
        "--cell-probability-weights",
        default="0",
        help="Comma-separated non-negative cell_probability_weight values merged into pairwise cost kwargs.",
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
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
        "--fail-fast",
        action="store_true",
        help="Raise the first filtering/benchmark error instead of recording an error row.",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument(
        "--write-incrementally",
        action="store_true",
        help="Write CSV rows to --output as each filtering setting completes.",
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
        write_cell_filtering_sweep_results_incrementally(
            iter_track2p_cell_filtering_sweep(config), args.output
        )
        return 0

    rows = [result.to_dict() for result in run_track2p_cell_filtering_sweep(config)]
    if args.output is not None:
        write_cell_filtering_sweep_results(rows, args.output, args.format)
    else:
        _write_cell_filtering_sweep_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> CellFilteringSweepConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    benchmark = Track2pBenchmarkConfig(
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
        include_non_cells=False,
        cell_probability_threshold=0.5,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )
    return CellFilteringSweepConfig(
        benchmark=benchmark,
        cell_probability_thresholds=_parse_probability_values(
            args.cell_probability_thresholds,
            name="--cell-probability-thresholds",
        ),
        filter_modes=_parse_filter_modes(args.filter_modes),
        cell_probability_weights=_parse_nonnegative_values(
            args.cell_probability_weights,
            name="--cell-probability-weights",
        ),
        fail_fast=args.fail_fast,
    )


def write_cell_filtering_sweep_results(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=_cell_filtering_sweep_fieldnames(rows)
            )
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_cell_filtering_sweep_table(rows) + "\n", encoding="utf-8")


def write_cell_filtering_sweep_results_incrementally(
    results: Iterable[SubjectBenchmarkResult], output_path: Path
) -> int:
    """Write CSV rows as each filtering setting completes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    fieldnames = _cell_filtering_sweep_fieldnames([])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
            handle.flush()
            rows_written += 1
    return rows_written


def _write_cell_filtering_sweep_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_cell_filtering_sweep_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_cell_filtering_sweep_table(rows))


def format_cell_filtering_sweep_table(
    rows: Sequence[dict[str, float | int | str]],
) -> str:
    columns = [
        "subject",
        "status",
        "suite2p_filter_policy",
        "cell_probability_threshold",
        "cell_probability_weight",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "evaluated_prediction_tracks",
        "dropped_prediction_tracks",
        "error_message",
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


def _cell_filtering_sweep_fieldnames(
    rows: Sequence[dict[str, float | int | str]],
) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "sweep_index",
        "sweep_count",
        "status",
        "suite2p_filter_policy",
        "include_non_cells",
        "cell_probability_threshold",
        "cell_probability_weight",
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
        "error_type",
        "error_message",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    if not rows:
        return preferred
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _parse_filter_modes(value: str) -> tuple[str, ...]:
    aliases = {
        "filtered": _FILTERED_MODE,
        "cells-only": _FILTERED_MODE,
        "cell": _FILTERED_MODE,
        "cells": _FILTERED_MODE,
        "include-non-cells": _INCLUDE_NON_CELLS_MODE,
        "include_non_cells": _INCLUDE_NON_CELLS_MODE,
        "all": _INCLUDE_NON_CELLS_MODE,
        "all-rois": _INCLUDE_NON_CELLS_MODE,
    }
    modes: list[str] = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part not in aliases:
            raise ValueError(
                f"Unknown filter mode {raw_part!r}; expected filtered or include-non-cells"
            )
        mode = aliases[part]
        if mode not in modes:
            modes.append(mode)
    return _normalise_filter_modes(tuple(modes))


def _parse_probability_values(value: str, *, name: str) -> tuple[float, ...]:
    return _normalise_probability_values(_parse_float_list(value, name=name))


def _parse_nonnegative_values(value: str, *, name: str) -> tuple[float, ...]:
    return _normalise_nonnegative_values(_parse_float_list(value, name=name), name=name)


def _parse_float_list(value: str, *, name: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            parsed.append(float(part))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of numbers") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(parsed)


def _normalise_filter_modes(values: Sequence[str]) -> tuple[str, ...]:
    modes = tuple(values)
    if not modes:
        raise ValueError("At least one filter mode is required")
    invalid = sorted(set(modes) - {_FILTERED_MODE, _INCLUDE_NON_CELLS_MODE})
    if invalid:
        raise ValueError("Unknown filter modes: " + ", ".join(invalid))
    return tuple(dict.fromkeys(modes))


def _normalise_probability_values(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one cell-probability threshold is required")
    if any((not np.isfinite(value)) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Cell-probability thresholds must be finite values in [0, 1]")
    return tuple(dict.fromkeys(thresholds))


def _normalise_nonnegative_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if not weights:
        raise ValueError(f"At least one {name} value is required")
    if any((not np.isfinite(value)) or value < 0.0 for value in weights):
        raise ValueError(f"{name} must be non-negative finite values")
    return tuple(dict.fromkeys(weights))


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        return f"{numeric:.4g}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
