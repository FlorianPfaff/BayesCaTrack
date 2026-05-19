"""Track2p benchmark wrapper with self-bootstrapped registration refinement."""

from __future__ import annotations

import argparse
from typing import Any

from bayescatrack.association.bootstrap_registration import (
    BootstrapRegistrationConfig,
    solve_bootstrapped_global_assignment_for_sessions,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments import track2p_benchmark as _benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the normal Track2p parser plus bootstrap-registration options."""

    parser = _benchmark.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-bootstrap"
    parser.description = (
        "Run Track2p global-assignment benchmarks with assignment-guided "
        "residual registration refinement."
    )
    parser.add_argument(
        "--bootstrap-registration-iterations",
        type=int,
        default=2,
        help="Number of assign/refit/rerun bootstrap-registration iterations",
    )
    parser.add_argument(
        "--bootstrap-registration-transform",
        choices=("translation", "affine"),
        default="affine",
        help="Residual transform fitted from high-confidence assignment anchors",
    )
    parser.add_argument(
        "--bootstrap-registration-min-matches",
        type=int,
        default=6,
        help="Minimum high-confidence anchors required to refine one session edge",
    )
    parser.add_argument(
        "--bootstrap-registration-min-cost-margin",
        type=float,
        default=0.25,
        help="Minimum row/column cost margin for anchors used to refit registration",
    )
    parser.add_argument(
        "--bootstrap-registration-max-anchor-cost",
        type=float,
        default=None,
        help="Optional maximum pairwise cost for registration-refit anchors",
    )
    parser.add_argument(
        "--bootstrap-registration-max-rmse",
        type=float,
        default=8.0,
        help="Reject an edge refit whose anchor RMSE exceeds this many pixels",
    )
    parser.add_argument(
        "--bootstrap-registration-no-refine-skip-edges",
        action="store_true",
        help="Refine only consecutive-session edges, even when --max-gap admits skips",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.method != "global-assignment":
        parser.error(
            "track2p-bootstrap currently supports --method global-assignment only"
        )
    if args.split != "subject":
        parser.error("track2p-bootstrap currently supports --split subject only")

    config = _benchmark._config_from_args(args)  # pylint: disable=protected-access
    bootstrap_config = BootstrapRegistrationConfig(
        iterations=args.bootstrap_registration_iterations,
        transform=args.bootstrap_registration_transform,
        min_matches=args.bootstrap_registration_min_matches,
        min_cost_margin=args.bootstrap_registration_min_cost_margin,
        max_anchor_cost=args.bootstrap_registration_max_anchor_cost,
        max_rmse=args.bootstrap_registration_max_rmse,
        refine_skip_edges=not args.bootstrap_registration_no_refine_skip_edges,
    )

    original_solve = _benchmark.solve_configured_global_assignment
    original_variant_name = _benchmark._variant_name  # pylint: disable=protected-access

    def _solve_with_bootstrap(
        sessions: list[Track2pSession] | tuple[Track2pSession, ...],
        benchmark_config: _benchmark.Track2pBenchmarkConfig,
        *,
        cost: AssociationCost | None = None,
        calibrated_model: Any | None = None,
    ) -> GlobalAssignmentRun:
        return solve_bootstrapped_global_assignment_for_sessions(
            sessions,
            bootstrap_config=bootstrap_config,
            max_gap=benchmark_config.max_gap,
            cost=benchmark_config.cost if cost is None else cost,
            calibrated_model=calibrated_model,
            transform_type=benchmark_config.transform_type,
            start_cost=benchmark_config.start_cost,
            end_cost=benchmark_config.end_cost,
            gap_penalty=benchmark_config.gap_penalty,
            cost_threshold=benchmark_config.cost_threshold,
            order=benchmark_config.order,
            weighted_centroids=benchmark_config.weighted_centroids,
            velocity_variance=benchmark_config.velocity_variance,
            regularization=benchmark_config.regularization,
            pairwise_cost_kwargs=benchmark_config.pairwise_cost_kwargs,
        )

    def _bootstrapped_variant_name(cost: AssociationCost) -> str:
        base_name = original_variant_name(cost)
        return f"Bootstrap registration + {base_name}"

    _benchmark.solve_configured_global_assignment = _solve_with_bootstrap
    # pylint: disable-next=protected-access
    _benchmark._variant_name = _bootstrapped_variant_name
    try:
        results = _benchmark.run_track2p_benchmark(config)
    finally:
        _benchmark.solve_configured_global_assignment = original_solve
        # pylint: disable-next=protected-access
        _benchmark._variant_name = original_variant_name

    rows = [result.to_dict() for result in results]
    if args.output is not None:
        _benchmark.write_results(rows, args.output, args.format)
    else:
        _benchmark._write_stdout(rows, args.format)  # pylint: disable=protected-access
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
