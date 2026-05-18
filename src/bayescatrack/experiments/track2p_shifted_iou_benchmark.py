"""Track2p benchmark wrapper with local shifted-IoU registered costs."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from bayescatrack.association.shifted_overlap import install_shifted_overlap_cost_patch
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_benchmark import (
    _config_from_args,
    _csv_fieldnames,
    build_arg_parser,
    run_track2p_benchmark,
    write_results,
)


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
        "--shifted-iou-additive-weight",
        type=float,
        default=0.0,
        help=(
            "Optional additive shifted-IoU cost weight. By default shifted IoU "
            "replaces the registered-IoU term instead of adding another term."
        ),
    )
    parser.add_argument(
        "--shifted-mask-cosine-weight",
        type=float,
        default=0.0,
        help="Optional additive best-shift mask-cosine cost weight.",
    )


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    from bayescatrack.experiments.track2p_benchmark import format_benchmark_table

    print(format_benchmark_table(rows))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-shifted-iou"
    _add_shifted_iou_options(parser)
    args = parser.parse_args(argv)
    if args.shifted_iou_radius < 0:
        raise ValueError("--shifted-iou-radius must be non-negative")
    if args.shifted_iou_additive_weight < 0.0:
        raise ValueError("--shifted-iou-additive-weight must be non-negative")
    if args.shifted_mask_cosine_weight < 0.0:
        raise ValueError("--shifted-mask-cosine-weight must be non-negative")

    config = _config_from_args(args)
    pairwise_cost_kwargs = dict(config.pairwise_cost_kwargs or {})
    pairwise_cost_kwargs.update(
        {
            "shifted_iou_radius": int(args.shifted_iou_radius),
            "use_shifted_iou_for_iou_cost": int(args.shifted_iou_radius) > 0,
            "shifted_iou_weight": float(args.shifted_iou_additive_weight),
            "shifted_mask_cosine_weight": float(args.shifted_mask_cosine_weight),
        }
    )
    config = replace(config, pairwise_cost_kwargs=pairwise_cost_kwargs)

    original_pairwise_cost = install_shifted_overlap_cost_patch()
    try:
        results = run_track2p_benchmark(config)
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_pairwise_cost  # type: ignore[method-assign]

    rows = [result.to_dict() for result in results]
    for row in rows:
        row["shifted_iou_radius"] = int(args.shifted_iou_radius)
        row["shifted_iou_additive_weight"] = float(args.shifted_iou_additive_weight)
        row["shifted_mask_cosine_weight"] = float(args.shifted_mask_cosine_weight)
    if args.output is not None:
        write_results(rows, Path(args.output), args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
