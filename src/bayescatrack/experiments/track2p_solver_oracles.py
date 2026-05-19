"""Run Track2p solver-oracle benchmark ablations as first-class artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bayescatrack.association.pyrecest_global_assignment import (
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.solver_oracles import (
    oracle_edge_costs,
    oracle_rank_k_costs,
    oracle_registration_costs,
    solve_from_pairwise_costs,
)
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.reference import Track2pReference

SolverOracle = Literal["edge-costs", "rank-k", "oracle-registration"]

DEFAULT_SOLVER_ORACLES: tuple[SolverOracle, ...] = (
    "edge-costs",
    "rank-k",
    "oracle-registration",
)
DEFAULT_RANK_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Track2pSolverOracleConfig:
    """Configuration for solver-oracle Track2p benchmark diagnostics."""

    benchmark: Track2pBenchmarkConfig
    oracles: tuple[SolverOracle, ...] = DEFAULT_SOLVER_ORACLES
    rank_ks: tuple[int, ...] = DEFAULT_RANK_KS
    large_cost: float = 1.0e6
    min_fit_links: int = 3
    require_full_rank: bool = True
    ridge: float = 0.0


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def run_track2p_solver_oracles(
    config: Track2pSolverOracleConfig,
) -> list[SubjectBenchmarkResult]:
    """Run solver-oracle ablations over one Track2p subject or dataset root.

    These diagnostics intentionally use manual ground-truth information to build
    pairwise costs before the global solver. They are therefore not reportable as
    independent tracking results. They are meant to localize failures:

    * ``edge-costs`` checks whether the global solver, track-row conversion, and
      scoring can recover tracks when all true edges are made free and all other
      edges are prohibitively expensive.
    * ``rank-k`` keeps a true edge only if it is already within the top-k row
      candidates under the selected base cost, exposing how pairwise rank quality
      limits downstream track F1.
    * ``oracle-registration`` fits manual-GT affine registration landmarks per
      session edge, then uses the normal pairwise cost to estimate the upper
      bound from better registration alone.
    """

    _validate_config(config)
    subject_dirs = tuple(discover_subject_dirs(config.benchmark.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.benchmark.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.benchmark.progress, label="solver-oracles"
    )
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.benchmark.data, config=config.benchmark
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config.benchmark
        )
        sessions = _load_subject_sessions(subject_dir, config.benchmark)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference, curated_only=config.benchmark.curated_only
        )

        if "edge-costs" in config.oracles:
            costs = oracle_edge_costs(
                sessions,
                reference_matrix,
                max_gap=config.benchmark.max_gap,
                large_cost=config.large_cost,
            )
            results.append(
                _score_solver_oracle(
                    subject=subject_dir.name,
                    variant="Oracle true-edge costs + global assignment",
                    oracle="edge-costs",
                    pairwise_costs=costs,
                    sessions=sessions,
                    reference=reference,
                    config=config,
                    extra_scores={"rank_k": ""},
                )
            )

        if "rank-k" in config.oracles:
            for rank_k in config.rank_ks:
                costs = oracle_rank_k_costs(
                    sessions,
                    reference_matrix,
                    rank_k=rank_k,
                    max_gap=config.benchmark.max_gap,
                    cost=config.benchmark.cost,
                    transform_type=config.benchmark.transform_type,
                    order=config.benchmark.order,
                    weighted_centroids=config.benchmark.weighted_centroids,
                    velocity_variance=config.benchmark.velocity_variance,
                    regularization=config.benchmark.regularization,
                    pairwise_cost_kwargs=config.benchmark.pairwise_cost_kwargs,
                    large_cost=config.large_cost,
                )
                results.append(
                    _score_solver_oracle(
                        subject=subject_dir.name,
                        variant=(
                            f"Oracle rank<={rank_k} true-edge costs "
                            "+ global assignment"
                        ),
                        oracle="rank-k",
                        pairwise_costs=costs,
                        sessions=sessions,
                        reference=reference,
                        config=config,
                        extra_scores={"rank_k": int(rank_k)},
                    )
                )

        if "oracle-registration" in config.oracles:
            costs = oracle_registration_costs(
                sessions,
                reference_matrix,
                max_gap=config.benchmark.max_gap,
                cost=config.benchmark.cost,
                order=config.benchmark.order,
                weighted_centroids=config.benchmark.weighted_centroids,
                velocity_variance=config.benchmark.velocity_variance,
                regularization=config.benchmark.regularization,
                pairwise_cost_kwargs=config.benchmark.pairwise_cost_kwargs,
                large_cost=config.large_cost,
                min_fit_links=config.min_fit_links,
                require_full_rank=config.require_full_rank,
                ridge=config.ridge,
            )
            results.append(
                _score_solver_oracle(
                    subject=subject_dir.name,
                    variant=(
                        "Oracle affine registration + normal costs "
                        "+ global assignment"
                    ),
                    oracle="oracle-registration",
                    pairwise_costs=costs,
                    sessions=sessions,
                    reference=reference,
                    config=config,
                    extra_scores={"rank_k": ""},
                )
            )
    return results


# pylint: disable=too-many-statements
def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for solver-oracle benchmark artifacts."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-solver-oracles",
        description="Run manual-GT solver-oracle ablations for Track2p-style datasets.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional ground_truth.csv file, ground-truth root, subject "
            "directory, or track2p folder"
        ),
    )
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware"),
        help="Base pairwise cost used by rank-k and oracle-registration diagnostics",
    )
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
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
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs",
    )
    parser.add_argument(
        "--oracle",
        dest="oracles",
        action="append",
        choices=DEFAULT_SOLVER_ORACLES,
        default=None,
        help="Oracle diagnostic to run; repeat to select multiple. Defaults to all.",
    )
    parser.add_argument(
        "--rank-k",
        dest="rank_ks",
        action="append",
        type=int,
        default=None,
        help="Rank-k threshold for the rank-k oracle; repeat for multiple k values.",
    )
    parser.add_argument("--large-cost", type=float, default=1.0e6)
    parser.add_argument("--min-fit-links", type=int, default=3)
    parser.add_argument("--allow-rank-deficient-fit", action="store_true")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--format", choices=("table", "json", "csv"), default="csv"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    results = run_track2p_solver_oracles(config)
    rows = [result.to_dict() for result in results]
    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, args.format)
    return 0


# pylint: disable=too-many-arguments
def _score_solver_oracle(
    *,
    subject: str,
    variant: str,
    oracle: SolverOracle,
    pairwise_costs: Mapping[tuple[int, int], Any],
    sessions: Sequence[Any],
    reference: Track2pReference,
    config: Track2pSolverOracleConfig,
    extra_scores: Mapping[str, int | str] | None = None,
) -> SubjectBenchmarkResult:
    assignment = solve_from_pairwise_costs(
        pairwise_costs,
        sessions,
        max_gap=config.benchmark.max_gap,
        start_cost=config.benchmark.start_cost,
        end_cost=config.benchmark.end_cost,
        gap_penalty=config.benchmark.gap_penalty,
        cost_threshold=config.benchmark.cost_threshold,
    )
    predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    scores: dict[str, float | int | str] = {
        **_score_prediction_against_reference(
            predicted, reference, config=config.benchmark
        ),
        "solver_oracle": oracle,
        "base_cost": config.benchmark.cost,
        "max_gap": int(config.benchmark.max_gap),
        "start_cost": float(config.benchmark.start_cost),
        "end_cost": float(config.benchmark.end_cost),
        "gap_penalty": float(config.benchmark.gap_penalty),
        "cost_threshold": (
            ""
            if config.benchmark.cost_threshold is None
            else float(config.benchmark.cost_threshold)
        ),
        "large_cost": float(config.large_cost),
    }
    if extra_scores:
        scores.update(dict(extra_scores))
    return SubjectBenchmarkResult(
        subject=subject,
        variant=variant,
        method="global-assignment",
        scores=scores,
        n_sessions=reference.n_sessions,
        reference_source=reference.source,
    )


# pylint: disable=too-many-locals
def _config_from_args(args: argparse.Namespace) -> Track2pSolverOracleConfig:
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
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
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
    return Track2pSolverOracleConfig(
        benchmark=benchmark,
        oracles=tuple(args.oracles or DEFAULT_SOLVER_ORACLES),
        rank_ks=tuple(args.rank_ks or DEFAULT_RANK_KS),
        large_cost=args.large_cost,
        min_fit_links=args.min_fit_links,
        require_full_rank=not args.allow_rank_deficient_fit,
        ridge=args.ridge,
    )


def _validate_config(config: Track2pSolverOracleConfig) -> None:
    if config.benchmark.cost not in {"registered-iou", "roi-aware"}:
        raise ValueError(
            "solver-oracle benchmarks currently support registered-iou or roi-aware "
            "base costs"
        )
    if not config.oracles:
        raise ValueError("At least one solver oracle must be requested")
    unknown_oracles = sorted(set(config.oracles) - set(DEFAULT_SOLVER_ORACLES))
    if unknown_oracles:
        raise ValueError(f"Unknown solver oracle(s): {', '.join(unknown_oracles)}")
    if config.benchmark.max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if config.large_cost <= 0.0:
        raise ValueError("large_cost must be positive")
    if "rank-k" in config.oracles:
        invalid_rank_ks = [rank_k for rank_k in config.rank_ks if rank_k < 1]
        if invalid_rank_ks:
            raise ValueError("rank_k values must be at least 1")
    if config.min_fit_links < 3:
        raise ValueError("min_fit_links must be at least 3")
    if config.ridge < 0.0:
        raise ValueError("ridge must be non-negative")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
