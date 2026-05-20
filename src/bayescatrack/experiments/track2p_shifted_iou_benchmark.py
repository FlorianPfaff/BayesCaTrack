"""Track2p benchmark wrapper with local shifted-IoU registered costs."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Any, cast

from bayescatrack.association.pyrecest_global_assignment import AssociationCost
from bayescatrack.association.shifted_overlap import install_shifted_overlap_cost_patch
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _config_from_args,
    _format_table_value,
    build_arg_parser,
    run_track2p_benchmark,
)

_SHIFTED_SWEEP_COSTS = (
    "registered-iou",
    "registered-shifted-iou",
    "roi-aware",
    "roi-aware-shifted",
)
_TRUE_LABELS = frozenset(("1", "true", "t", "yes", "y", "on"))
_FALSE_LABELS = frozenset(("0", "false", "f", "no", "n", "off"))


@dataclass(frozen=True)
class ShiftedIouSetting:
    """One shifted-overlap parameter setting in a benchmark sweep."""

    cost: AssociationCost
    radius: int
    additive_weight: float
    mask_cosine_weight: float
    shift_penalty_weight: float
    shift_penalty_scale: float | None
    transform_type: str
    weighted_masks: bool
    sweep_index: int = 1
    sweep_count: int = 1


def _add_shifted_iou_options(parser: Any) -> None:
    parser.add_argument(
        "--shifted-iou-radius",
        type=int,
        default=4,
        help=(
            "Local integer-shift radius in pixels. The IoU cost uses the best "
            "overlap after shifting each measurement ROI within [-radius, radius] "
            "in x/y. Exact IoU remains available in pairwise components."
        ),
    )
    parser.add_argument(
        "--shifted-iou-radii",
        "--shifted-iou-radius-sweep",
        dest="shifted_iou_radii",
        default=None,
        help=(
            "Optional comma-separated shifted-IoU radii to sweep. When omitted, "
            "the single --shifted-iou-radius value is used."
        ),
    )
    parser.add_argument(
        "--shifted-iou-additive-weight",
        type=float,
        default=0.0,
        help=(
            "Optional additive shifted-IoU cost weight. By default shifted IoU "
            "replaces the registered-IoU term instead of adding another term."
        ),
    )
    parser.add_argument(
        "--shifted-iou-additive-weights",
        "--shifted-iou-additive-weight-sweep",
        dest="shifted_iou_additive_weights",
        default=None,
        help=(
            "Optional comma-separated additive shifted-IoU weights to sweep. "
            "When omitted, the single --shifted-iou-additive-weight value is used."
        ),
    )
    parser.add_argument(
        "--shifted-mask-cosine-weight",
        type=float,
        default=0.0,
        help="Optional additive best-shift mask-cosine cost weight.",
    )
    parser.add_argument(
        "--shifted-mask-cosine-weights",
        "--shifted-mask-cosine-weight-sweep",
        dest="shifted_mask_cosine_weights",
        default=None,
        help=(
            "Optional comma-separated best-shift mask-cosine weights to sweep. "
            "When omitted, the single --shifted-mask-cosine-weight value is used."
        ),
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Optional cost weight for the residual local shift selected by shifted IoU. "
            "This regularizes against large local shifts that recover overlap."
        ),
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-scale",
        type=float,
        default=None,
        help=(
            "Positive scale for the shifted-IoU residual-shift penalty. Defaults "
            "to the shifted-IoU radius."
        ),
    )
    parser.add_argument(
        "--costs",
        default=None,
        help=(
            "Comma-separated association costs for a sweep. Supported values are "
            f"{', '.join(_SHIFTED_SWEEP_COSTS)}. Defaults to the scalar --cost value."
        ),
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-weights",
        "--shift-penalty-weights",
        dest="shifted_iou_shift_penalty_weights",
        default=None,
        help=(
            "Comma-separated residual-shift penalty weights for a sweep, e.g. "
            "0,0.1,0.25,0.5. Defaults to the scalar "
            "--shifted-iou-shift-penalty-weight value."
        ),
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-scales",
        "--shifted-iou-shift-penalty-scale-sweep",
        dest="shifted_iou_shift_penalty_scales",
        default=None,
        help=(
            "Comma-separated residual-shift penalty scales for a sweep. Use "
            "'none' for the default scale equal to the radius. Defaults to the "
            "scalar --shifted-iou-shift-penalty-scale value."
        ),
    )
    parser.add_argument(
        "--transform-types",
        default=None,
        help=(
            "Comma-separated registration transform types for a sweep, e.g. "
            "affine,rigid,fov-translation. Defaults to the scalar --transform-type value."
        ),
    )
    parser.add_argument(
        "--weighted-mask-states",
        default=None,
        help=(
            "Comma-separated booleans controlling weighted-mask reconstruction in a sweep, "
            "e.g. false,true. Defaults to the scalar --weighted-masks flag."
        ),
    )


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_shifted_iou_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return

    print(format_shifted_iou_table(rows))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-shifted-iou"
    _add_shifted_iou_options(parser)
    args = parser.parse_args(argv)
    _validate_shifted_iou_options(args)

    config = _config_from_args(args)
    original_pairwise_cost = install_shifted_overlap_cost_patch()
    try:
        if _uses_sweep_args(args):
            rows = _run_shifted_iou_sweep(args, config)
        else:
            rows = _run_shifted_iou_setting(
                args,
                config,
                ShiftedIouSetting(
                    cost=cast(AssociationCost, config.cost),
                    radius=int(args.shifted_iou_radius),
                    additive_weight=float(args.shifted_iou_additive_weight),
                    mask_cosine_weight=float(args.shifted_mask_cosine_weight),
                    shift_penalty_weight=float(args.shifted_iou_shift_penalty_weight),
                    shift_penalty_scale=args.shifted_iou_shift_penalty_scale,
                    transform_type=config.transform_type,
                    weighted_masks=config.weighted_masks,
                ),
            )
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_pairwise_cost  # type: ignore[method-assign]

    if args.output is not None:
        write_shifted_iou_results(rows, Path(args.output), args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _validate_shifted_iou_options(args: Any) -> None:
    if args.shifted_iou_radius < 0:
        raise ValueError("--shifted-iou-radius must be non-negative")
    if args.shifted_iou_additive_weight < 0.0 or not isfinite(args.shifted_iou_additive_weight):
        raise ValueError("--shifted-iou-additive-weight must be non-negative and finite")
    if args.shifted_mask_cosine_weight < 0.0 or not isfinite(args.shifted_mask_cosine_weight):
        raise ValueError("--shifted-mask-cosine-weight must be non-negative and finite")
    if args.shifted_iou_shift_penalty_weight < 0.0 or not isfinite(args.shifted_iou_shift_penalty_weight):
        raise ValueError("--shifted-iou-shift-penalty-weight must be non-negative and finite")
    if (
        args.shifted_iou_shift_penalty_scale is not None
        and (args.shifted_iou_shift_penalty_scale <= 0.0 or not isfinite(args.shifted_iou_shift_penalty_scale))
    ):
        raise ValueError("--shifted-iou-shift-penalty-scale must be strictly positive and finite")


def _uses_sweep_args(args: Any) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "costs",
            "shifted_iou_radii",
            "shifted_iou_additive_weights",
            "shifted_mask_cosine_weights",
            "shifted_iou_shift_penalty_weights",
            "shifted_iou_shift_penalty_scales",
            "transform_types",
            "weighted_mask_states",
        )
    )


def _run_shifted_iou_sweep(args: Any, base_config: Track2pBenchmarkConfig) -> list[dict[str, Any]]:
    settings = _sweep_settings(args, base_config)
    rows: list[dict[str, Any]] = []
    for setting in settings:
        rows.extend(_run_shifted_iou_setting(args, base_config, setting))
    return rows


def _run_shifted_iou_setting(
    args: Any,
    base_config: Track2pBenchmarkConfig,
    setting: ShiftedIouSetting,
) -> list[dict[str, Any]]:
    pairwise_cost_kwargs = _shifted_pairwise_cost_kwargs(
        base_config.pairwise_cost_kwargs,
        radius=setting.radius,
        additive_weight=setting.additive_weight,
        mask_cosine_weight=setting.mask_cosine_weight,
        shift_penalty_weight=setting.shift_penalty_weight,
        shift_penalty_scale=setting.shift_penalty_scale,
    )
    config = replace(
        base_config,
        cost=setting.cost,
        transform_type=setting.transform_type,
        weighted_masks=setting.weighted_masks,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    rows = [result.to_dict() for result in run_track2p_benchmark(config)]
    _add_shifted_iou_metadata(
        rows,
        setting=setting,
    )
    return rows


def _shifted_pairwise_cost_kwargs(
    base_kwargs: dict[str, Any] | None,
    *,
    radius: int,
    additive_weight: float,
    mask_cosine_weight: float,
    shift_penalty_weight: float,
    shift_penalty_scale: float | None,
) -> dict[str, Any]:
    pairwise_cost_kwargs = dict(base_kwargs or {})
    pairwise_cost_kwargs.update(
        {
            "shifted_iou_radius": int(radius),
            "use_shifted_iou_for_iou_cost": int(radius) > 0,
            "shifted_iou_weight": float(additive_weight),
            "shifted_mask_cosine_weight": float(mask_cosine_weight),
            "shifted_iou_shift_penalty_weight": float(shift_penalty_weight),
        }
    )
    if shift_penalty_scale is None:
        pairwise_cost_kwargs.pop("shifted_iou_shift_penalty_scale", None)
    else:
        pairwise_cost_kwargs["shifted_iou_shift_penalty_scale"] = float(shift_penalty_scale)
    return pairwise_cost_kwargs


def _add_shifted_iou_metadata(
    rows: list[dict[str, Any]],
    *,
    setting: ShiftedIouSetting,
) -> None:
    for row in rows:
        row["association_cost"] = setting.cost
        row["transform_type"] = setting.transform_type
        row["weighted_masks"] = bool(setting.weighted_masks)
        row["shifted_iou_radius"] = int(setting.radius)
        row["shifted_iou_additive_weight"] = float(setting.additive_weight)
        row["shifted_mask_cosine_weight"] = float(setting.mask_cosine_weight)
        row["shifted_iou_shift_penalty_weight"] = float(setting.shift_penalty_weight)
        row["shifted_iou_shift_penalty_scale"] = (
            "" if setting.shift_penalty_scale is None else float(setting.shift_penalty_scale)
        )
        if setting.sweep_count > 1:
            row["sweep_index"] = int(setting.sweep_index)
            row["sweep_count"] = int(setting.sweep_count)


def _sweep_settings(args: Any, base_config: Track2pBenchmarkConfig) -> tuple[ShiftedIouSetting, ...]:
    costs = _parse_costs(args.costs, default=(base_config.cost,))
    radii = _parse_nonnegative_int_values(
        args.shifted_iou_radii,
        default=(int(args.shifted_iou_radius),),
        name="--shifted-iou-radii",
    )
    additive_weights = _parse_nonnegative_float_values(
        args.shifted_iou_additive_weights,
        default=(float(args.shifted_iou_additive_weight),),
        name="--shifted-iou-additive-weights",
    )
    mask_cosine_weights = _parse_nonnegative_float_values(
        args.shifted_mask_cosine_weights,
        default=(float(args.shifted_mask_cosine_weight),),
        name="--shifted-mask-cosine-weights",
    )
    shift_penalty_weights = _parse_nonnegative_float_values(
        args.shifted_iou_shift_penalty_weights,
        default=(float(args.shifted_iou_shift_penalty_weight),),
        name="--shifted-iou-shift-penalty-weights",
    )
    shift_penalty_scales = _parse_positive_float_or_none_values(
        args.shifted_iou_shift_penalty_scales,
        default=(args.shifted_iou_shift_penalty_scale,),
        name="--shifted-iou-shift-penalty-scales",
    )
    transform_types = _parse_string_values(
        args.transform_types,
        default=(base_config.transform_type,),
        name="--transform-types",
    )
    weighted_mask_states = _parse_bool_values(
        args.weighted_mask_states,
        default=(bool(base_config.weighted_masks),),
        name="--weighted-mask-states",
    )

    raw_settings = list(
        product(
            costs,
            radii,
            additive_weights,
            mask_cosine_weights,
            shift_penalty_weights,
            shift_penalty_scales,
            transform_types,
            weighted_mask_states,
        )
    )
    sweep_count = len(raw_settings)
    return tuple(
        ShiftedIouSetting(
            cost=cost,
            radius=radius,
            additive_weight=additive_weight,
            mask_cosine_weight=mask_cosine_weight,
            shift_penalty_weight=shift_penalty_weight,
            shift_penalty_scale=shift_penalty_scale,
            transform_type=transform_type,
            weighted_masks=weighted_masks,
            sweep_index=index + 1,
            sweep_count=sweep_count,
        )
        for index, (
            cost,
            radius,
            additive_weight,
            mask_cosine_weight,
            shift_penalty_weight,
            shift_penalty_scale,
            transform_type,
            weighted_masks,
        ) in enumerate(raw_settings)
    )


def _parse_costs(raw: str | None, *, default: Sequence[str]) -> tuple[AssociationCost, ...]:
    values = _parse_string_values(raw, default=default, name="--costs")
    invalid = sorted(set(values) - set(_SHIFTED_SWEEP_COSTS))
    if invalid:
        raise ValueError(
            "--costs contains unsupported values: "
            + ", ".join(invalid)
            + "; supported values are "
            + ", ".join(_SHIFTED_SWEEP_COSTS)
        )
    return tuple(cast(AssociationCost, value) for value in values)


def _parse_nonnegative_int_values(raw: str | None, *, default: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in _parse_string_values(raw, default=tuple(str(value) for value in default), name=name))
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated integers") from exc
    if any(value < 0 for value in values):
        raise ValueError(f"{name} values must be non-negative")
    return values


def _parse_nonnegative_float_values(raw: str | None, *, default: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in _parse_string_values(raw, default=tuple(str(value) for value in default), name=name))
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers") from exc
    if any(value < 0.0 or not isfinite(value) for value in values):
        raise ValueError(f"{name} values must be non-negative finite numbers")
    return values


def _parse_positive_float_or_none_values(
    raw: str | None,
    *,
    default: Sequence[float | None],
    name: str,
) -> tuple[float | None, ...]:
    if raw is None:
        return tuple(default)
    values: list[float | None] = []
    for value in _parse_string_values(raw, default=(), name=name):
        if value.casefold() in {"none", "null", "default"}:
            values.append(None)
            continue
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated numbers or none") from exc
        if parsed <= 0.0 or not isfinite(parsed):
            raise ValueError(f"{name} values must be strictly positive finite numbers or none")
        values.append(parsed)
    return tuple(values)


def _parse_bool_values(raw: str | None, *, default: Sequence[bool], name: str) -> tuple[bool, ...]:
    if raw is None:
        return tuple(bool(value) for value in default)
    values: list[bool] = []
    for value in _parse_string_values(raw, default=(), name=name):
        normalized = value.casefold()
        if normalized in _TRUE_LABELS:
            values.append(True)
        elif normalized in _FALSE_LABELS:
            values.append(False)
        else:
            raise ValueError(f"{name} values must be booleans, got {value!r}")
    return tuple(values)


def write_shifted_iou_results(
    rows: Sequence[dict[str, Any]], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_shifted_iou_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_shifted_iou_table(rows) + "\n", encoding="utf-8")


def format_shifted_iou_table(rows: Sequence[dict[str, Any]]) -> str:
    columns = [
        "subject",
        "association_cost",
        "transform_type",
        "weighted_masks",
        "shifted_iou_radius",
        "shifted_iou_shift_penalty_weight",
        "shifted_iou_shift_penalty_scale",
        "shifted_iou_additive_weight",
        "shifted_mask_cosine_weight",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_table_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def _shifted_iou_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "association_cost",
        "transform_type",
        "weighted_masks",
        "n_sessions",
        "reference_source",
        "sweep_index",
        "sweep_count",
        "shifted_iou_radius",
        "shifted_iou_additive_weight",
        "shifted_mask_cosine_weight",
        "shifted_iou_shift_penalty_weight",
        "shifted_iou_shift_penalty_scale",
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


def _parse_string_values(raw: str | None, *, default: Sequence[str], name: str) -> tuple[str, ...]:
    if raw is None:
        return tuple(str(value) for value in default)
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
