"""Input-mask and Suite2p ROI-filtering sweeps for Track2p benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)

# pylint: disable=too-many-instance-attributes


@dataclass(frozen=True)
class MaskInputSweepConfig:
    """Configuration for a Track2p input-mask/filtering sweep."""

    benchmark: Track2pBenchmarkConfig
    include_non_cells: tuple[bool, ...] = (False, True)
    cell_probability_thresholds: tuple[float, ...] = (0.5,)
    weighted_masks: tuple[bool, ...] = (False, True)
    weighted_centroids: tuple[bool, ...] | None = None
    exclude_overlapping_pixels: tuple[bool, ...] = (True, False)


@dataclass(frozen=True)
class MaskInputSetting:
    """One input-mask/filtering setting within a sweep."""

    include_non_cells: bool
    cell_probability_threshold: float | None
    weighted_masks: bool
    weighted_centroids: bool
    exclude_overlapping_pixels: bool
    sweep_index: int
    sweep_count: int

    @property
    def label(self) -> str:
        roi_filter = (
            "all-rois"
            if self.include_non_cells
            else f"iscell>={float(self.cell_probability_threshold):g}"
        )
        mask_mode = "lam-masks" if self.weighted_masks else "binary-masks"
        centroid_mode = "weighted-centroids" if self.weighted_centroids else "binary-centroids"
        overlap_mode = "drop-overlap" if self.exclude_overlapping_pixels else "keep-overlap"
        return "/".join((roi_filter, mask_mode, centroid_mode, overlap_mode))

    def to_score_fields(self) -> dict[str, float | int | str]:
        return {
            "sweep_index": int(self.sweep_index),
            "sweep_count": int(self.sweep_count),
            "input_variant": self.label,
            "include_non_cells": _bool_label(self.include_non_cells),
            "cell_probability_threshold": (
                "all" if self.cell_probability_threshold is None else float(self.cell_probability_threshold)
            ),
            "weighted_masks": _bool_label(self.weighted_masks),
            "weighted_centroids": _bool_label(self.weighted_centroids),
            "exclude_overlapping_pixels": _bool_label(self.exclude_overlapping_pixels),
        }


def run_track2p_mask_input_sweep(config: MaskInputSweepConfig) -> list[SubjectBenchmarkResult]:
    """Run Track2p benchmarks over input-mask and ROI-filtering variants."""

    return list(iter_track2p_mask_input_sweep(config))


def iter_track2p_mask_input_sweep(config: MaskInputSweepConfig) -> Iterator[SubjectBenchmarkResult]:
    """Yield one benchmark row per subject and input-mask/filtering setting."""

    for setting in _mask_input_settings(config):
        benchmark_config = _benchmark_for_setting(config.benchmark, setting)
        for result in run_track2p_benchmark(benchmark_config):
            yield SubjectBenchmarkResult(
                subject=result.subject,
                variant=result.variant,
                method=result.method,
                scores={**dict(result.scores), **setting.to_score_fields()},
                n_sessions=result.n_sessions,
                reference_source=result.reference_source,
            )


def _benchmark_for_setting(
    benchmark: Track2pBenchmarkConfig, setting: MaskInputSetting
) -> Track2pBenchmarkConfig:
    threshold = (
        benchmark.cell_probability_threshold
        if setting.cell_probability_threshold is None
        else setting.cell_probability_threshold
    )
    return replace(
        benchmark,
        include_non_cells=setting.include_non_cells,
        cell_probability_threshold=threshold,
        weighted_masks=setting.weighted_masks,
        weighted_centroids=setting.weighted_centroids,
        exclude_overlapping_pixels=setting.exclude_overlapping_pixels,
    )


def _mask_input_settings(config: MaskInputSweepConfig) -> tuple[MaskInputSetting, ...]:
    include_options = _normalise_bool_options(
        config.include_non_cells, name="include_non_cells"
    )
    thresholds = _normalise_cell_probability_thresholds(config.cell_probability_thresholds)
    weighted_mask_options = _normalise_bool_options(
        config.weighted_masks, name="weighted_masks"
    )
    weighted_centroid_options = (
        None
        if config.weighted_centroids is None
        else _normalise_bool_options(config.weighted_centroids, name="weighted_centroids")
    )
    overlap_options = _normalise_bool_options(
        config.exclude_overlapping_pixels, name="exclude_overlapping_pixels"
    )

    partial: list[tuple[bool, float | None, bool, bool, bool]] = []
    for include_non_cells in include_options:
        threshold_options: tuple[float | None, ...] = (None,) if include_non_cells else thresholds
        for threshold in threshold_options:
            for weighted_masks in weighted_mask_options:
                centroid_options = (
                    (weighted_masks,)
                    if weighted_centroid_options is None
                    else weighted_centroid_options
                )
                for weighted_centroids in centroid_options:
                    for exclude_overlapping_pixels in overlap_options:
                        partial.append(
                            (
                                include_non_cells,
                                threshold,
                                weighted_masks,
                                weighted_centroids,
                                exclude_overlapping_pixels,
                            )
                        )

    sweep_count = len(partial)
    return tuple(
        MaskInputSetting(
            include_non_cells=include_non_cells,
            cell_probability_threshold=threshold,
            weighted_masks=weighted_masks,
            weighted_centroids=weighted_centroids,
            exclude_overlapping_pixels=exclude_overlapping_pixels,
            sweep_index=index,
            sweep_count=sweep_count,
        )
        for index, (
            include_non_cells,
            threshold,
            weighted_masks,
            weighted_centroids,
            exclude_overlapping_pixels,
        ) in enumerate(partial, start=1)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-mask-input-sweep",
        description="Sweep Suite2p ROI filtering, weighted masks, weighted centroids, and overlap-pixel handling.",
    )
    parser.add_argument("--data", required=True, type=Path, help="Track2p dataset root or one subject directory")
    parser.add_argument(
        "--method",
        default="global-assignment",
        choices=("track2p-baseline", "global-assignment", "oracle-gt-links"),
        help="Benchmark method to evaluate for each input setting",
    )
    parser.add_argument(
        "--split",
        default="subject",
        choices=("subject", "leave-one-subject-out"),
        help="Evaluation split policy",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--restrict-to-reference-seed-rois", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-non-cell-options",
        default="false,true",
        help="Comma-separated booleans controlling Suite2p iscell filtering, e.g. false,true",
    )
    parser.add_argument(
        "--cell-probability-thresholds",
        default="0.5",
        help="Comma-separated Suite2p iscell probability thresholds used when non-cells are excluded",
    )
    parser.add_argument(
        "--weighted-mask-options",
        default="false,true",
        help="Comma-separated booleans controlling Suite2p lam-weighted mask reconstruction",
    )
    parser.add_argument(
        "--weighted-centroid-options",
        default="auto",
        help="auto to follow weighted-mask mode, or comma-separated booleans",
    )
    parser.add_argument(
        "--exclude-overlap-options",
        default="true,false",
        help="Comma-separated booleans controlling Suite2p overlap-pixel removal",
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    rows = [result.to_dict() for result in run_track2p_mask_input_sweep(config)]
    if args.output is not None:
        write_mask_sweep_results(rows, args.output, args.format)
    else:
        _write_mask_sweep_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> MaskInputSweepConfig:
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
        order=args.order,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )
    return MaskInputSweepConfig(
        benchmark=benchmark,
        include_non_cells=_parse_bool_options(args.include_non_cell_options, name="--include-non-cell-options"),
        cell_probability_thresholds=_parse_threshold_options(args.cell_probability_thresholds),
        weighted_masks=_parse_bool_options(args.weighted_mask_options, name="--weighted-mask-options"),
        weighted_centroids=_parse_weighted_centroid_options(args.weighted_centroid_options),
        exclude_overlapping_pixels=_parse_bool_options(args.exclude_overlap_options, name="--exclude-overlap-options"),
    )


def _parse_weighted_centroid_options(raw: str) -> tuple[bool, ...] | None:
    if raw.strip().casefold() == "auto":
        return None
    return _parse_bool_options(raw, name="--weighted-centroid-options")


def _parse_threshold_options(raw: str) -> tuple[float, ...]:
    tokens = _split_tokens(raw, name="--cell-probability-thresholds")
    values: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"--cell-probability-thresholds contains a non-numeric value: {token!r}"
            ) from exc
        values.append(value)
    return _normalise_cell_probability_thresholds(tuple(values))


def _parse_bool_options(raw: str, *, name: str) -> tuple[bool, ...]:
    return _normalise_bool_options(
        tuple(_parse_bool_token(token, name=name) for token in _split_tokens(raw, name=name)),
        name=name,
    )


def _split_tokens(raw: str, *, name: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a comma-separated list with no empty entries")
    return tokens


def _parse_bool_token(token: str, *, name: str) -> bool:
    value = token.casefold()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} contains a non-boolean value: {token!r}")


def _normalise_bool_options(values: Sequence[bool], *, name: str) -> tuple[bool, ...]:
    normalised = tuple(dict.fromkeys(bool(value) for value in values))
    if not normalised:
        raise ValueError(f"At least one {name} option is required")
    return normalised


def _normalise_cell_probability_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(dict.fromkeys(float(value) for value in values))
    if not thresholds:
        raise ValueError("At least one cell-probability threshold is required")
    if any((not np.isfinite(value)) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Cell-probability thresholds must be finite values in [0, 1]")
    return thresholds


def write_mask_sweep_results(
    rows: Sequence[dict[str, float | int | str]], output_path: Path, output_format: OutputFormat
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_mask_sweep_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_mask_sweep_table(rows) + "\n", encoding="utf-8")


def _write_mask_sweep_stdout(rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_mask_sweep_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_mask_sweep_table(rows))


def format_mask_sweep_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    columns = [
        "subject",
        "input_variant",
        "include_non_cells",
        "cell_probability_threshold",
        "weighted_masks",
        "weighted_centroids",
        "exclude_overlapping_pixels",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        body.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(body)


def _mask_sweep_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "sweep_index",
        "sweep_count",
        "input_variant",
        "include_non_cells",
        "cell_probability_threshold",
        "weighted_masks",
        "weighted_centroids",
        "exclude_overlapping_pixels",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
