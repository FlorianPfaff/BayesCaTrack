"""Shifted-overlap ablations for Track2p raw Suite2p benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bayescatrack.association.shifted_overlap import install_shifted_overlap_cost_patch
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    Track2pBenchmarkConfig,
    _config_from_args as _benchmark_config_from_args,
    build_arg_parser as _build_track2p_arg_parser,
    format_benchmark_table,
    run_track2p_benchmark,
    write_results,
)
from bayescatrack.experiments.track2p_edge_ranking import (
    DEFAULT_EDGE_RANKING_FEATURES,
    DEFAULT_SIMILARITY_FEATURES,
    run_track2p_edge_ranking,
)
from bayescatrack.fov_affine_registration import (
    register_measurement_plane_by_fov_affine,
)

DEFAULT_SHIFTED_IOU_RADII = (0, 2, 4, 6, 8)
DEFAULT_SHIFTED_MASK_COSINE_WEIGHTS = (0.5, 1.0)
SHIFTED_EDGE_RANKING_FEATURES = (
    "shifted_iou",
    "shifted_iou_cost",
    "shifted_iou_shift_norm",
    "shifted_mask_cosine_similarity",
    "shifted_mask_cosine_cost",
    "iou_for_cost",
    "mask_cosine_for_cost",
)
SHIFTED_SIMILARITY_FEATURES = (
    "shifted_iou",
    "shifted_mask_cosine_similarity",
    "iou_for_cost",
    "mask_cosine_for_cost",
)


@dataclass(frozen=True)
class ShiftedOverlapVariant:
    """One shifted-overlap cost variant to run through the benchmark."""

    name: str
    shifted_iou_radius: int
    use_shifted_iou_for_iou_cost: bool
    shifted_iou_weight: float
    shifted_mask_cosine_weight: float
    pairwise_cost_kwargs: Mapping[str, Any]

    def row_metadata(self) -> dict[str, float | int | str]:
        """Return CSV-friendly metadata for benchmark result rows."""

        return {
            "shifted_overlap_variant": self.name,
            "shifted_iou_radius": int(self.shifted_iou_radius),
            "use_shifted_iou_for_iou_cost": int(self.use_shifted_iou_for_iou_cost),
            "shifted_iou_weight": float(self.shifted_iou_weight),
            "shifted_mask_cosine_weight": float(self.shifted_mask_cosine_weight),
        }


def build_shifted_overlap_variants(
    radii: Sequence[int] = DEFAULT_SHIFTED_IOU_RADII,
    *,
    shifted_cosine_weights: Sequence[float] = DEFAULT_SHIFTED_MASK_COSINE_WEIGHTS,
    include_exact_baseline: bool = True,
    include_additive_shifted_iou: bool = False,
) -> tuple[ShiftedOverlapVariant, ...]:
    """Return exact, shifted-IoU, and shifted-cosine ablation variants."""

    normalized_radii = _normalize_nonnegative_ints(radii, name="radii")
    normalized_cosine_weights = _normalize_nonnegative_floats(
        shifted_cosine_weights, name="shifted_cosine_weights"
    )
    variants: list[ShiftedOverlapVariant] = []
    if include_exact_baseline:
        variants.append(
            ShiftedOverlapVariant(
                name="exact-iou",
                shifted_iou_radius=0,
                use_shifted_iou_for_iou_cost=False,
                shifted_iou_weight=0.0,
                shifted_mask_cosine_weight=0.0,
                pairwise_cost_kwargs={},
            )
        )

    for radius in normalized_radii:
        if radius == 0:
            continue
        shifted_iou_kwargs: dict[str, Any] = {
            "shifted_iou_radius": int(radius),
            "use_shifted_iou_for_iou_cost": True,
        }
        variants.append(
            ShiftedOverlapVariant(
                name=f"shifted-iou-r{radius}",
                shifted_iou_radius=int(radius),
                use_shifted_iou_for_iou_cost=True,
                shifted_iou_weight=0.0,
                shifted_mask_cosine_weight=0.0,
                pairwise_cost_kwargs=shifted_iou_kwargs,
            )
        )
        for cosine_weight in normalized_cosine_weights:
            if cosine_weight <= 0.0:
                continue
            variants.append(
                ShiftedOverlapVariant(
                    name=(
                        f"shifted-iou-r{radius}-shifted-cosine-"
                        f"w{_float_slug(cosine_weight)}"
                    ),
                    shifted_iou_radius=int(radius),
                    use_shifted_iou_for_iou_cost=True,
                    shifted_iou_weight=0.0,
                    shifted_mask_cosine_weight=float(cosine_weight),
                    pairwise_cost_kwargs={
                        **shifted_iou_kwargs,
                        "shifted_mask_cosine_weight": float(cosine_weight),
                    },
                )
            )
        if include_additive_shifted_iou:
            variants.append(
                ShiftedOverlapVariant(
                    name=f"additive-shifted-iou-r{radius}-w1",
                    shifted_iou_radius=int(radius),
                    use_shifted_iou_for_iou_cost=False,
                    shifted_iou_weight=1.0,
                    shifted_mask_cosine_weight=0.0,
                    pairwise_cost_kwargs={
                        "shifted_iou_radius": int(radius),
                        "shifted_iou_weight": 1.0,
                    },
                )
            )
    return tuple(variants)


def run_shifted_overlap_ablation(
    config: Track2pBenchmarkConfig,
    variants: Sequence[ShiftedOverlapVariant],
    *,
    edge_ranking_output_dir: Path | None = None,
    edge_ranking_features: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Run benchmark rows for every shifted-overlap variant."""

    if config.method != "global-assignment":
        raise ValueError("Shifted-overlap ablations require method='global-assignment'")
    if config.cost == "calibrated":
        raise ValueError(
            "Shifted-overlap ablations currently support raw costs only; "
            "use cost='registered-iou' or cost='roi-aware'."
        )
    variants = tuple(variants)
    if not variants:
        raise ValueError("At least one shifted-overlap variant is required")

    rows: list[dict[str, float | int | str]] = []
    with _runtime_patches():
        for variant in variants:
            variant_config = _config_for_variant(config, variant)
            results = run_track2p_benchmark(variant_config)
            for result in results:
                row = result.to_dict()
                row.update(variant.row_metadata())
                row["transform_type"] = variant_config.transform_type
                rows.append(row)
            if edge_ranking_output_dir is not None:
                _run_edge_ranking_for_variant(
                    variant_config,
                    variant,
                    output_dir=edge_ranking_output_dir,
                    feature_names=edge_ranking_features,
                )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for shifted-overlap ablations."""

    parser = _build_track2p_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-shifted-overlap-ablation"
    parser.description = (
        "Run shifted-IoU radius and shifted-mask-cosine ablations on the "
        "Track2p raw Suite2p benchmark."
    )
    _update_parser_action(
        parser,
        "method",
        required=False,
        default="global-assignment",
        choices=("global-assignment",),
        help="Benchmark method; shifted-overlap ablations require global assignment.",
    )
    _update_parser_action(
        parser,
        "cost",
        choices=("registered-iou", "roi-aware"),
        help="Raw pairwise cost family to augment with shifted-overlap terms.",
    )
    _update_parser_action(
        parser,
        "transform_type",
        default="fov-affine",
        choices=("affine", "rigid", "fov-affine", "fov-translation", "none"),
        help="Registration transform; fov-affine is the default for this ablation.",
    )
    parser.add_argument(
        "--radii",
        default=",".join(str(radius) for radius in DEFAULT_SHIFTED_IOU_RADII),
        help="Comma-separated shifted-IoU radii in pixels; default: 0,2,4,6,8.",
    )
    parser.add_argument(
        "--shifted-cosine-weights",
        default=",".join(
            str(weight) for weight in DEFAULT_SHIFTED_MASK_COSINE_WEIGHTS
        ),
        help=(
            "Comma-separated additive shifted-mask-cosine weights to test for "
            "each positive radius; use an empty string to disable cosine variants."
        ),
    )
    parser.add_argument(
        "--no-exact-baseline",
        action="store_true",
        help="Do not include the exact registered-IoU baseline row.",
    )
    parser.add_argument(
        "--include-additive-shifted-iou",
        action="store_true",
        help="Also test additive shifted-IoU as a tie-breaker instead of replacement.",
    )
    parser.add_argument(
        "--edge-ranking-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-variant edge-ranking CSVs; when set, "
            "the same shifted-overlap variants are ranked against manual GT."
        ),
    )
    parser.add_argument(
        "--edge-ranking-feature",
        dest="edge_ranking_features",
        action="append",
        default=None,
        help=(
            "Feature/component to include in optional edge-ranking outputs; "
            "repeat to override the default feature set."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _benchmark_config_from_args(args)
    try:
        variants = build_shifted_overlap_variants(
            _parse_int_list(args.radii, name="radii"),
            shifted_cosine_weights=_parse_float_list(
                args.shifted_cosine_weights,
                name="shifted_cosine_weights",
                allow_empty=True,
            ),
            include_exact_baseline=not args.no_exact_baseline,
            include_additive_shifted_iou=args.include_additive_shifted_iou,
        )
        rows = run_shifted_overlap_ablation(
            config,
            variants,
            edge_ranking_output_dir=args.edge_ranking_output_dir,
            edge_ranking_features=args.edge_ranking_features,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _config_for_variant(
    config: Track2pBenchmarkConfig, variant: ShiftedOverlapVariant
) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = dict(config.pairwise_cost_kwargs or {})
    pairwise_cost_kwargs.update(dict(variant.pairwise_cost_kwargs))
    return replace(config, pairwise_cost_kwargs=pairwise_cost_kwargs)


@contextmanager
def _runtime_patches() -> Any:
    import bayescatrack.association.calibrated_costs as calibrated_costs
    import bayescatrack.association.pyrecest_global_assignment as assignment

    original_assignment_register = assignment.register_plane_pair
    original_calibrated_register = calibrated_costs.register_plane_pair
    original_pairwise_cost = install_shifted_overlap_cost_patch()
    assignment.register_plane_pair = _register_plane_pair_with_fov_affine
    calibrated_costs.register_plane_pair = _register_plane_pair_with_fov_affine
    try:
        yield
    finally:
        assignment.register_plane_pair = original_assignment_register
        calibrated_costs.register_plane_pair = original_calibrated_register
        CalciumPlaneData.build_pairwise_cost_matrix = original_pairwise_cost  # type: ignore[method-assign]


def _register_plane_pair_with_fov_affine(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str = "affine",
) -> CalciumPlaneData:
    if transform_type == "fov-affine":
        return register_measurement_plane_by_fov_affine(
            reference_plane,
            moving_plane,
        ).registered_measurement_plane
    from bayescatrack.track2p_registration import register_plane_pair

    return register_plane_pair(
        reference_plane,
        moving_plane,
        transform_type=transform_type,
    )


def _run_edge_ranking_for_variant(
    config: Track2pBenchmarkConfig,
    variant: ShiftedOverlapVariant,
    *,
    output_dir: Path,
    feature_names: Sequence[str] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant.name}_edge_ranking.csv"
    summary_output_path = output_dir / f"{variant.name}_edge_ranking_summary.csv"
    run_track2p_edge_ranking(
        config,
        output_path,
        summary_output_path=summary_output_path,
        feature_names=_edge_ranking_features_for_variant(variant, feature_names),
        similarity_features=_similarity_features_for_variant(variant),
    )


def _edge_ranking_features_for_variant(
    variant: ShiftedOverlapVariant,
    override: Sequence[str] | None,
) -> tuple[str, ...]:
    if override is not None:
        return tuple(dict.fromkeys(str(feature) for feature in override))
    features = list(DEFAULT_EDGE_RANKING_FEATURES)
    if variant.shifted_iou_radius > 0:
        features.extend(SHIFTED_EDGE_RANKING_FEATURES)
    return tuple(dict.fromkeys(features))


def _similarity_features_for_variant(
    variant: ShiftedOverlapVariant,
) -> tuple[str, ...]:
    features = list(DEFAULT_SIMILARITY_FEATURES)
    if variant.shifted_iou_radius > 0:
        features.extend(SHIFTED_SIMILARITY_FEATURES)
    return tuple(dict.fromkeys(features))


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    if value.strip() == "":
        raise ValueError(f"--{name.replace('_', '-')} must not be empty")
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name.replace('_', '-')} must contain integers") from exc
    return _normalize_nonnegative_ints(parsed, name=name)


def _parse_float_list(
    value: str, *, name: str, allow_empty: bool = False
) -> tuple[float, ...]:
    if value.strip() == "":
        if allow_empty:
            return ()
        raise ValueError(f"--{name.replace('_', '-')} must not be empty")
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name.replace('_', '-')} must contain numbers") from exc
    return _normalize_nonnegative_floats(parsed, name=name)


def _normalize_nonnegative_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        integer_value = int(value)
        if integer_value < 0:
            raise ValueError(f"{name} must contain only non-negative integers")
        if integer_value not in normalized:
            normalized.append(integer_value)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(normalized)


def _normalize_nonnegative_floats(
    values: Sequence[float], *, name: str
) -> tuple[float, ...]:
    normalized: list[float] = []
    for value in values:
        float_value = float(value)
        if float_value < 0.0:
            raise ValueError(f"{name} must contain only non-negative values")
        if float_value not in normalized:
            normalized.append(float_value)
    return tuple(normalized)


def _float_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _update_parser_action(
    parser: argparse.ArgumentParser, dest: str, **updates: Any
) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            for key, value in updates.items():
                setattr(action, key, value)
            return
    raise RuntimeError(f"Could not find parser action {dest!r}")


def _write_stdout(
    rows: Sequence[Mapping[str, float | int | str]],
    output_format: OutputFormat,
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_benchmark_table(list(rows)))


def _fieldnames(rows: Sequence[Mapping[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "shifted_overlap_variant",
        "method",
        "n_sessions",
        "reference_source",
        "transform_type",
        "shifted_iou_radius",
        "use_shifted_iou_for_iou_cost",
        "shifted_iou_weight",
        "shifted_mask_cosine_weight",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    row_keys = {key for row in rows for key in row}
    return [key for key in preferred if key in row_keys] + sorted(
        row_keys - set(preferred)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
