"""Empirical registration-model selection for Track2p-style benchmarks."""

# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-statements
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.experiments.registration_qa_report import (
    RegistrationQAConfig,
    run_registration_qa_report,
    summarize_registration_qa_links,
)
from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind,
    Track2pBenchmarkConfig,
)
from bayescatrack.experiments.track2p_cost_sweep import (
    CostSweepConfig,
    run_track2p_cost_sweep,
)
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES

RegistrationSelectionCost = Literal["registered-iou", "roi-aware"]
OutputFormat = Literal["table", "json", "csv"]
SelectionMetric = Literal[
    "auto",
    "pairwise_f1",
    "complete_track_f1",
    "gt_recall_at_1",
    "gt_recall_at_5",
    "median_registered_iou",
    "median_cost_margin",
]

DEFAULT_TRANSFORM_TYPES = (
    "fov-affine",
    "bspline",
    "tps",
    "local-affine-grid",
    "optical-flow",
)


@dataclass(frozen=True)
class RegistrationModelSelectionConfig:
    """Configuration for empirical registration-model selection."""

    data: Path
    reference: Path | None = None
    reference_kind: ReferenceKind = "manual-gt"
    allow_track2p_as_reference_for_smoke_test: bool = False
    curated_only: bool = False
    plane_name: str = "plane0"
    input_format: str = "auto"
    transform_types: tuple[str, ...] = DEFAULT_TRANSFORM_TYPES
    include_oracle: bool = False
    cost: RegistrationSelectionCost = "registered-iou"
    max_gap: int = 2
    cost_scale: float = 1.25
    start_cost: float = 1.0
    end_cost: float = 1.0
    gap_penalty: float = 0.6
    cost_threshold: float | None = 2.0
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
    run_benchmark: bool = True
    selection_metric: SelectionMetric = "auto"
    continue_on_error: bool = True
    progress: bool = True


@dataclass(frozen=True)
class RegistrationModelSelectionResult:
    """Complete output of a registration-model selection run."""

    summary_rows: tuple[dict[str, Any], ...]
    qa_summary_rows: tuple[dict[str, Any], ...]
    qa_link_rows: tuple[dict[str, Any], ...]
    benchmark_rows: tuple[dict[str, Any], ...]
    failure_rows: tuple[dict[str, Any], ...]
    selected_registration_model: str | None
    selection_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_registration_model": self.selected_registration_model,
            "selection_metric": self.selection_metric,
            "summary": list(self.summary_rows),
            "registration_qa_summary": list(self.qa_summary_rows),
            "registration_qa_links": list(self.qa_link_rows),
            "benchmark": list(self.benchmark_rows),
            "failures": list(self.failure_rows),
        }


def run_registration_model_selection(
    config: RegistrationModelSelectionConfig,
) -> RegistrationModelSelectionResult:
    """Run QA and benchmark evidence for each registration model and select one."""

    transform_types = _registration_model_sequence(config)
    qa_link_rows: list[dict[str, Any]] = []
    qa_summary_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for transform_type in transform_types:
        if config.progress:
            print(
                f"registration-model-selection: {transform_type}",
                file=sys.stderr,
                flush=True,
            )
        qa_ok = _append_registration_qa_rows(
            transform_type,
            config,
            qa_link_rows=qa_link_rows,
            qa_summary_rows=qa_summary_rows,
            failure_rows=failure_rows,
        )
        if not qa_ok and not config.continue_on_error:
            raise RuntimeError(
                f"Registration QA failed for transform_type={transform_type!r}; "
                "rerun with --continue-on-error to keep partial results."
            )
        if config.run_benchmark and transform_type != "gt-affine-oracle":
            benchmark_ok = _append_benchmark_rows(
                transform_type,
                config,
                benchmark_rows=benchmark_rows,
                failure_rows=failure_rows,
            )
            if not benchmark_ok and not config.continue_on_error:
                raise RuntimeError(
                    f"Benchmark failed for transform_type={transform_type!r}; "
                    "rerun with --continue-on-error to keep partial results."
                )
        elif config.run_benchmark and transform_type == "gt-affine-oracle":
            failure_rows.append(
                {
                    "registration_model": transform_type,
                    "stage": "benchmark",
                    "status": "skipped",
                    "error": "gt-affine-oracle uses manual-GT landmarks and is QA-only",
                }
            )

    selection_metric = _resolve_selection_metric(
        config.selection_metric,
        benchmark_rows,
    )
    summary_rows = _summarize_model_selection(
        transform_types,
        qa_summary_rows=qa_summary_rows,
        benchmark_rows=benchmark_rows,
        failure_rows=failure_rows,
        selection_metric=selection_metric,
        config=config,
    )
    selected_model = next(
        (
            str(row["registration_model"])
            for row in summary_rows
            if bool(row.get("selected", False))
        ),
        None,
    )
    return RegistrationModelSelectionResult(
        summary_rows=tuple(summary_rows),
        qa_summary_rows=tuple(qa_summary_rows),
        qa_link_rows=tuple(qa_link_rows),
        benchmark_rows=tuple(benchmark_rows),
        failure_rows=tuple(failure_rows),
        selected_registration_model=selected_model,
        selection_metric=selection_metric,
    )


def write_registration_model_selection_results(
    result: RegistrationModelSelectionResult,
    output_dir: Path,
) -> None:
    """Write all registration-model selection artifacts into one directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "registration_model_selection_summary.csv", result.summary_rows)
    _write_csv(output_dir / "registration_model_qa_summary.csv", result.qa_summary_rows)
    _write_csv(output_dir / "registration_model_qa_links.csv", result.qa_link_rows)
    _write_csv(output_dir / "registration_model_benchmark.csv", result.benchmark_rows)
    _write_csv(output_dir / "registration_model_failures.csv", result.failure_rows)
    (output_dir / "registration_model_selection.json").write_text(
        json.dumps(_json_ready(result.to_dict()), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "registration_model_selection.md").write_text(
        format_model_selection_table(result.summary_rows) + "\n",
        encoding="utf-8",
    )


def format_model_selection_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format model-selection rows as a compact Markdown table."""

    columns = [
        "selected",
        "registration_model",
        "status",
        "selection_metric",
        "selection_value",
        "pairwise_f1",
        "complete_track_f1",
        "gt_recall_at_1",
        "gt_recall_at_5",
        "median_registered_iou",
        "median_registered_centroid_distance",
        "median_gt_rank",
        "median_cost_margin",
        "qa_gt_links",
        "benchmark_subjects",
        "error_count",
    ]
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append(
            "| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(body)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark registration-model-selection",
        description=(
            "Empirically select a registration model by running manual-GT registration QA "
            "and, optionally, full Track2p global-assignment benchmarks for each candidate."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
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
    parser.add_argument(
        "--transform-types",
        default=",".join(DEFAULT_TRANSFORM_TYPES),
        help=(
            "Comma-separated registration models to compare. Supported values include "
            f"{', '.join(REGISTRATION_TRANSFORM_TYPES)}."
        ),
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Also run the manual-GT affine oracle QA row; this is not benchmarked.",
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware"),
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--cost-scale",
        type=float,
        default=1.25,
        help="Scale applied to pairwise costs before solver thresholding in benchmark runs.",
    )
    parser.add_argument("--start-cost", type=float, default=1.0)
    parser.add_argument("--end-cost", type=float, default=1.0)
    parser.add_argument("--gap-penalty", type=float, default=0.6)
    parser.add_argument("--cost-threshold", type=float, default=2.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
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
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run full global-assignment benchmark rows in addition to registration QA.",
    )
    parser.add_argument(
        "--selection-metric",
        default="auto",
        choices=(
            "auto",
            "pairwise_f1",
            "complete_track_f1",
            "gt_recall_at_1",
            "gt_recall_at_5",
            "median_registered_iou",
            "median_cost_margin",
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--format", default="table", choices=("table", "json", "csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_registration_model_selection(_config_from_args(args))
    if args.output_dir is not None:
        write_registration_model_selection_results(result, args.output_dir)
    elif args.format == "json":
        print(json.dumps(_json_ready(result.to_dict()), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(result.summary_rows))
        writer.writeheader()
        writer.writerows(result.summary_rows)
    else:
        print(format_model_selection_table(result.summary_rows))
    return 0


def _append_registration_qa_rows(
    transform_type: str,
    config: RegistrationModelSelectionConfig,
    *,
    qa_link_rows: list[dict[str, Any]],
    qa_summary_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> bool:
    try:
        link_rows = [
            _tag_model_row(row, transform_type, config)
            for row in run_registration_qa_report(
                _registration_qa_config(config, transform_type)
            )
        ]
    except Exception as exc:  # pragma: no cover - exercised on optional backend failures
        failure_rows.append(_failure_row(transform_type, "registration_qa", exc))
        return False
    summary_rows = [
        _tag_model_row(row, transform_type, config)
        for row in summarize_registration_qa_links(link_rows)
    ]
    qa_link_rows.extend(link_rows)
    qa_summary_rows.extend(summary_rows)
    return True


def _append_benchmark_rows(
    transform_type: str,
    config: RegistrationModelSelectionConfig,
    *,
    benchmark_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> bool:
    try:
        rows = [
            _tag_model_row(result.to_dict(), transform_type, config)
            for result in run_track2p_cost_sweep(
                CostSweepConfig(
                    benchmark=_benchmark_config(config, transform_type),
                    cost_scales=(config.cost_scale,),
                    cost_thresholds=(config.cost_threshold,),
                    start_costs=(config.start_cost,),
                    end_costs=(config.end_cost,),
                    gap_penalties=(config.gap_penalty,),
                )
            )
        ]
    except Exception as exc:  # pragma: no cover - exercised on optional backend failures
        failure_rows.append(_failure_row(transform_type, "benchmark", exc))
        return False
    benchmark_rows.extend(rows)
    return True


def _registration_qa_config(
    config: RegistrationModelSelectionConfig,
    transform_type: str,
) -> RegistrationQAConfig:
    return RegistrationQAConfig(
        data=config.data,
        reference=config.reference,
        reference_kind=config.reference_kind,
        allow_track2p_as_reference_for_smoke_test=config.allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        plane_name=config.plane_name,
        input_format=config.input_format,
        max_gap=config.max_gap,
        transform_type=transform_type,
        cost=config.cost,
        cost_threshold=_qa_raw_cost_threshold(config),
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        progress=config.progress,
    )


def _benchmark_config(
    config: RegistrationModelSelectionConfig,
    transform_type: str,
) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=config.data,
        method="global-assignment",
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=config.reference,
        reference_kind=config.reference_kind,
        allow_track2p_as_reference_for_smoke_test=config.allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        cost=config.cost,
        max_gap=config.max_gap,
        transform_type=transform_type,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        progress=config.progress,
    )


def _summarize_model_selection(
    transform_types: Sequence[str],
    *,
    qa_summary_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    selection_metric: str,
    config: RegistrationModelSelectionConfig,
) -> list[dict[str, Any]]:
    qa_by_model = _group_by_model(qa_summary_rows)
    benchmark_by_model = _group_by_model(benchmark_rows)
    failures_by_model = _group_by_model(failure_rows)
    summary_rows: list[dict[str, Any]] = []
    for transform_type in transform_types:
        qa_rows = qa_by_model.get(transform_type, [])
        bench_rows = benchmark_by_model.get(transform_type, [])
        failures = failures_by_model.get(transform_type, [])
        row = {
            "selected": False,
            "registration_model": transform_type,
            "status": _model_status(qa_rows, bench_rows, failures, config),
            "selection_metric": selection_metric,
            "selection_value": np.nan,
            "cost": config.cost,
            "cost_scale": float(config.cost_scale),
            "solver_cost_threshold": _threshold_label(config.cost_threshold),
            "qa_raw_cost_threshold": _threshold_label(_qa_raw_cost_threshold(config)),
            "start_cost": float(config.start_cost),
            "end_cost": float(config.end_cost),
            "gap_penalty": float(config.gap_penalty),
            "qa_session_edges": int(len(qa_rows)),
            "qa_gt_links": int(_sum_numeric(qa_rows, "n_gt_links")),
            "benchmark_subjects": int(len(bench_rows)),
            "error_count": int(len(failures)),
            "errors": " | ".join(str(row.get("error", "")) for row in failures),
            "gt_recall_at_1": _weighted_mean(qa_rows, "gt_recall_at_1", "n_gt_links"),
            "gt_recall_at_5": _weighted_mean(qa_rows, "gt_recall_at_5", "n_gt_links"),
            "gt_recall_at_10": _weighted_mean(qa_rows, "gt_recall_at_10", "n_gt_links"),
            "gt_admissible_rate": _weighted_mean(qa_rows, "gt_admissible_rate", "n_gt_links"),
            "median_registered_iou": _weighted_mean(qa_rows, "median_registered_iou", "n_gt_links"),
            "median_registered_centroid_distance": _weighted_mean(qa_rows, "median_registered_centroid_distance", "n_gt_links"),
            "median_gt_rank": _weighted_mean(qa_rows, "median_gt_rank", "n_gt_links"),
            "median_cost_margin": _weighted_mean(qa_rows, "median_cost_margin", "n_gt_links"),
            "empty_registered_fraction": _weighted_mean(qa_rows, "empty_registered_fraction", "n_gt_links"),
            "pairwise_f1": _mean_numeric(bench_rows, "pairwise_f1"),
            "pairwise_precision": _mean_numeric(bench_rows, "pairwise_precision"),
            "pairwise_recall": _mean_numeric(bench_rows, "pairwise_recall"),
            "complete_track_f1": _mean_numeric(bench_rows, "complete_track_f1"),
            "complete_track_precision": _mean_numeric(bench_rows, "complete_track_precision"),
            "complete_track_recall": _mean_numeric(bench_rows, "complete_track_recall"),
        }
        row["selection_value"] = row.get(selection_metric, np.nan)
        summary_rows.append(row)

    best_index = _best_row_index(summary_rows, selection_metric)
    if best_index is not None:
        summary_rows[best_index]["selected"] = True
    return summary_rows


def _config_from_args(args: argparse.Namespace) -> RegistrationModelSelectionConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        pairwise_cost_kwargs = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(pairwise_cost_kwargs, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    return RegistrationModelSelectionConfig(
        data=args.data,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        plane_name=args.plane_name,
        input_format=args.input_format,
        transform_types=_parse_transform_types(args.transform_types),
        include_oracle=args.include_oracle,
        cost=args.cost,
        max_gap=args.max_gap,
        cost_scale=args.cost_scale,
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
        run_benchmark=args.run_benchmark,
        selection_metric=args.selection_metric,
        continue_on_error=args.continue_on_error,
        progress=args.progress,
    )


def _registration_model_sequence(
    config: RegistrationModelSelectionConfig,
) -> tuple[str, ...]:
    models = list(config.transform_types)
    if config.include_oracle and "gt-affine-oracle" not in models:
        models.append("gt-affine-oracle")
    return tuple(models)


def _parse_transform_types(value: str) -> tuple[str, ...]:
    items = tuple(
        item.strip()
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    )
    if not items:
        raise ValueError("At least one --transform-types entry is required")
    valid = set(REGISTRATION_TRANSFORM_TYPES) | {"gt-affine-oracle"}
    invalid = sorted(set(items) - valid)
    if invalid:
        valid_text = ", ".join(sorted(valid))
        raise ValueError(
            f"Unsupported registration model(s): {', '.join(invalid)}. Valid values: {valid_text}"
        )
    return tuple(dict.fromkeys(items))


def _qa_raw_cost_threshold(config: RegistrationModelSelectionConfig) -> float | None:
    if config.cost_threshold is None:
        return None
    if not np.isfinite(config.cost_scale) or config.cost_scale <= 0.0:
        raise ValueError("--cost-scale must be a positive finite number")
    return float(config.cost_threshold) / float(config.cost_scale)


def _tag_model_row(
    row: Mapping[str, Any],
    transform_type: str,
    config: RegistrationModelSelectionConfig,
) -> dict[str, Any]:
    return {
        "registration_model": transform_type,
        "cost_scale": float(config.cost_scale),
        "solver_cost_threshold": _threshold_label(config.cost_threshold),
        "qa_raw_cost_threshold": _threshold_label(_qa_raw_cost_threshold(config)),
        **dict(row),
    }


def _failure_row(transform_type: str, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "registration_model": transform_type,
        "stage": stage,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _resolve_selection_metric(
    metric: SelectionMetric,
    benchmark_rows: Sequence[Mapping[str, Any]],
) -> str:
    if metric != "auto":
        return str(metric)
    return "pairwise_f1" if benchmark_rows else "gt_recall_at_1"


def _best_row_index(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> int | None:
    values = [_as_float(row.get(metric, np.nan)) for row in rows]
    valid = [
        (index, value)
        for index, value in enumerate(values)
        if np.isfinite(value) and str(rows[index].get("status", "")) != "failed"
    ]
    if not valid:
        return None
    if metric in {"median_gt_rank", "median_registered_centroid_distance", "empty_registered_fraction"}:
        return min(valid, key=lambda item: item[1])[0]
    return max(valid, key=lambda item: item[1])[0]


def _model_status(
    qa_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    config: RegistrationModelSelectionConfig,
) -> str:
    failed_stages = {str(row.get("stage", "")) for row in failure_rows if row.get("status") == "failed"}
    if "registration_qa" in failed_stages and ("benchmark" in failed_stages or not benchmark_rows):
        return "failed"
    if "registration_qa" in failed_stages:
        return "qa_failed"
    if config.run_benchmark and "benchmark" in failed_stages:
        return "benchmark_failed"
    if config.run_benchmark and not benchmark_rows and not any(
        row.get("status") == "skipped" for row in failure_rows
    ):
        return "benchmark_missing"
    if not qa_rows:
        return "qa_missing"
    return "ok"


def _group_by_model(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("registration_model", ""))].append(row)
    return grouped


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    weight_key: str,
) -> float:
    values: list[float] = []
    weights: list[float] = []
    for row in rows:
        value = _as_float(row.get(key, np.nan))
        weight = _as_float(row.get(weight_key, 1.0))
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0:
            values.append(value)
            weights.append(weight)
    if not values:
        return np.nan
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _mean_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_as_float(row.get(key, np.nan)) for row in rows]
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if finite.size else np.nan


def _sum_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_as_float(row.get(key, np.nan)) for row in rows]
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.sum(finite)) if finite.size else 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(_json_ready(dict(row)) for row in rows)


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return value


def _threshold_label(value: float | None) -> str | float:
    return "none" if value is None else float(value)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
