"""LOSO-safe tuning of global-assignment solver priors.

This module tunes only the PyRecEst path-cover solver knobs
``start_cost``, ``end_cost``, ``gap_penalty``, and ``cost_threshold`` on
training subjects.  It deliberately does not learn or change the pairwise
association cost, so the birth/death/skip/admission priors remain separated
from edge ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    _load_pyrecest_multisession_solver,
    build_registered_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ReferenceKind,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    _variant_name,
    discover_subject_dirs,
    format_benchmark_table,
    solve_configured_global_assignment,
)
from bayescatrack.reference import Track2pReference

SolverPriorObjective = Literal["pairwise_f1", "complete_track_f1", "mean_f1"]


class SolverPriorSubject(Protocol):
    """Loaded subject object accepted by the solver-prior tuner."""

    @property
    def subject_name(self) -> str:
        """Subject identifier used in result metadata."""

    sessions: tuple[Track2pSession, ...]
    reference: Track2pReference


@dataclass(frozen=True)
class SubjectSolverPriorData:
    """Loaded sessions and reference identities for one subject."""

    subject_dir: Path
    sessions: tuple[Track2pSession, ...]
    reference: Track2pReference

    @property
    def subject_name(self) -> str:
        return self.subject_dir.name


@dataclass(frozen=True)
class SolverPriorCandidate:
    """One start/end/gap/threshold setting for the global solver."""

    start_cost: float
    end_cost: float
    gap_penalty: float
    cost_threshold: float | None

    def config_for(self, config: Track2pBenchmarkConfig) -> Track2pBenchmarkConfig:
        """Return a benchmark config with this candidate applied."""

        return replace(
            config,
            start_cost=float(self.start_cost),
            end_cost=float(self.end_cost),
            gap_penalty=float(self.gap_penalty),
            cost_threshold=None if self.cost_threshold is None else float(self.cost_threshold),
        )

    def score_fields(self) -> dict[str, float | str]:
        return {
            "learned_start_cost": float(self.start_cost),
            "learned_end_cost": float(self.end_cost),
            "learned_gap_penalty": float(self.gap_penalty),
            "learned_cost_threshold": threshold_label(self.cost_threshold),
        }


@dataclass(frozen=True)
class SolverPriorSearchConfig:
    """Finite solver-prior search space."""

    start_costs: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    end_costs: tuple[float, ...] = ()
    gap_penalties: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.2)
    cost_thresholds: tuple[float | None, ...] = (1.5, 2.0, 2.5, None)
    objective: SolverPriorObjective = "complete_track_f1"

    def candidates(self) -> tuple[SolverPriorCandidate, ...]:
        starts = _positive_values(self.start_costs, name="start_costs")
        ends = _positive_values(self.end_costs or self.start_costs, name="end_costs")
        gaps = _nonnegative_values(self.gap_penalties, name="gap_penalties")
        thresholds = _thresholds(self.cost_thresholds)
        return tuple(
            SolverPriorCandidate(start, end, gap, threshold)
            for start in starts
            for end in ends
            for gap in gaps
            for threshold in thresholds
        )


@dataclass(frozen=True)
class SolverPriorTuningResult:
    """Best training-fold candidate and aggregate training scores."""

    best_candidate: SolverPriorCandidate
    best_scores: Mapping[str, float | int]
    objective: SolverPriorObjective
    evaluated_candidates: int
    training_subjects: tuple[str, ...]

    def config_with_best_priors(self, config: Track2pBenchmarkConfig) -> Track2pBenchmarkConfig:
        return self.best_candidate.config_for(config)

    def score_fields(self) -> dict[str, float | int | str]:
        fields: dict[str, float | int | str] = {
            "solver_prior_learned": 1,
            "solver_prior_objective": self.objective,
            "solver_prior_candidates": int(self.evaluated_candidates),
            "solver_prior_training_subjects": ",".join(self.training_subjects),
            **self.best_candidate.score_fields(),
        }
        for key, value in self.best_scores.items():
            fields[f"solver_prior_training_{key}"] = value
        return fields


@dataclass(frozen=True)
class LosoSolverPriorFold:
    """One held-out subject result."""

    held_out_subject: str
    training_subjects: tuple[str, ...]
    benchmark: SubjectBenchmarkResult
    tuning: SolverPriorTuningResult

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            **self.benchmark.to_dict(),
            "held_out_subject": self.held_out_subject,
            "training_subjects": ",".join(self.training_subjects),
        }


@dataclass(frozen=True)
class LosoSolverPriorResult:
    """All folds from a LOSO solver-prior run."""

    folds: tuple[LosoSolverPriorFold, ...]

    def to_rows(self) -> list[dict[str, float | int | str]]:
        return [fold.to_dict() for fold in self.folds]

    def to_benchmark_results(self) -> list[SubjectBenchmarkResult]:
        return [fold.benchmark for fold in self.folds]


@dataclass(frozen=True)
class _PreparedSubject:
    subject_name: str
    sessions: tuple[Track2pSession, ...]
    reference: Track2pReference
    pairwise_costs: Mapping[tuple[int, int], np.ndarray]
    session_sizes: tuple[int, ...]


def tune_solver_priors(
    training_subjects: Sequence[SolverPriorSubject],
    *,
    config: Track2pBenchmarkConfig,
    search: SolverPriorSearchConfig | None = None,
    cost: AssociationCost | None = None,
    calibrated_model: Any | None = None,
) -> SolverPriorTuningResult:
    """Tune solver priors on training subjects only."""

    subjects = tuple(training_subjects)
    if not subjects:
        raise ValueError("At least one training subject is required")
    search = search or SolverPriorSearchConfig()
    candidates = search.candidates()
    if not candidates:
        raise ValueError("At least one solver-prior candidate is required")

    prepared = tuple(
        _prepare_subject(subject, config=config, cost=cost or config.cost, calibrated_model=calibrated_model)
        for subject in subjects
    )
    solver = _load_pyrecest_multisession_solver()
    best_candidate = candidates[0]
    best_scores = _score_candidate(best_candidate, prepared, config=config, solver=solver)
    for candidate in candidates[1:]:
        scores = _score_candidate(candidate, prepared, config=config, solver=solver)
        if _objective_tuple(scores, search.objective) > _objective_tuple(best_scores, search.objective):
            best_candidate = candidate
            best_scores = scores
    return SolverPriorTuningResult(
        best_candidate=best_candidate,
        best_scores=best_scores,
        objective=search.objective,
        evaluated_candidates=len(candidates),
        training_subjects=tuple(subject.subject_name for subject in subjects),
    )


def run_track2p_loso_solver_priors(
    config: Track2pBenchmarkConfig,
    *,
    search: SolverPriorSearchConfig | None = None,
) -> LosoSolverPriorResult:
    """Run non-calibrated global assignment with LOSO-tuned solver priors."""

    if config.method != "global-assignment":
        raise ValueError("LOSO solver-prior tuning requires method='global-assignment'")
    if config.split != "leave-one-subject-out":
        raise ValueError("LOSO solver-prior tuning requires split='leave-one-subject-out'")
    if config.cost == "calibrated":
        raise ValueError("Use the calibrated LOSO path for cost='calibrated'")

    subjects = tuple(_load_subject_data(subject_dir, config=config) for subject_dir in discover_subject_dirs(config.data))
    if len(subjects) < 2:
        raise ValueError("LOSO solver-prior tuning requires at least two subject directories")

    folds: list[LosoSolverPriorFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject
            for index, subject in enumerate(subjects)
            if index != held_out_index
        )
        tuning = tune_solver_priors(
            cast(Sequence[SolverPriorSubject], training_subjects),
            config=config,
            search=search,
            cost=config.cost,
        )
        solve_config = tuning.config_with_best_priors(config)
        assignment = solve_configured_global_assignment(held_out.sessions, solve_config, cost=config.cost)
        predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, held_out.sessions)
        scores: dict[str, float | int | str] = {
            **_score_prediction_against_reference(predicted, held_out.reference, config=solve_config),
            **tuning.score_fields(),
        }
        folds.append(
            LosoSolverPriorFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(subject.subject_name for subject in training_subjects),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant=f"{_variant_name(config.cost)} + LOSO learned solver priors",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                tuning=tuning,
            )
        )
    return LosoSolverPriorResult(folds=tuple(folds))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.solver_prior_tuning",
        description="Tune global-assignment solver priors with leave-one-subject-out training folds.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--reference-kind", default="manual-gt", choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"))
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--cost", default="registered-iou", choices=("registered-iou", "roi-aware"))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--transform-type", default="affine", choices=("affine", "rigid", "fov-translation", "none"))
    parser.add_argument("--start-costs", default="0.5,1,1.5,2")
    parser.add_argument("--end-costs", default="")
    parser.add_argument("--gap-penalties", default="0,0.3,0.6,0.9,1.2")
    parser.add_argument("--cost-thresholds", default="1.5,2,2.5,none")
    parser.add_argument("--objective", default="complete_track_f1", choices=("pairwise_f1", "complete_track_f1", "mean_f1"))
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument("--exclude-overlapping-pixels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pairwise_cost_kwargs = json.loads(args.pairwise_cost_kwargs_json) if args.pairwise_cost_kwargs_json else None
    if pairwise_cost_kwargs is not None and not isinstance(pairwise_cost_kwargs, dict):
        raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        reference=args.reference,
        reference_kind=cast(ReferenceKind, args.reference_kind),
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        cost=cast(AssociationCost, args.cost),
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        weighted_centroids=args.weighted_centroids,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    search = SolverPriorSearchConfig(
        start_costs=parse_positive_list(args.start_costs, name="--start-costs"),
        end_costs=parse_positive_list(args.end_costs, name="--end-costs") if args.end_costs else (),
        gap_penalties=parse_nonnegative_list(args.gap_penalties, name="--gap-penalties"),
        cost_thresholds=parse_threshold_list(args.cost_thresholds),
        objective=cast(SolverPriorObjective, args.objective),
    )
    rows = run_track2p_loso_solver_priors(config, search=search).to_rows()
    _write_rows(rows, args.output, args.format)
    return 0


def _prepare_subject(
    subject: SolverPriorSubject,
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None,
) -> _PreparedSubject:
    pairwise_costs = build_registered_pairwise_costs(
        subject.sessions,
        max_gap=config.max_gap,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
    )
    return _PreparedSubject(
        subject_name=subject.subject_name,
        sessions=tuple(subject.sessions),
        reference=subject.reference,
        pairwise_costs=pairwise_costs,
        session_sizes=tuple(int(session.plane_data.n_rois) for session in subject.sessions),
    )


def _score_candidate(
    candidate: SolverPriorCandidate,
    subjects: Sequence[_PreparedSubject],
    *,
    config: Track2pBenchmarkConfig,
    solver: Any,
) -> dict[str, float | int]:
    fold_config = candidate.config_for(config)
    subject_scores = []
    for subject in subjects:
        result = solver(
            subject.pairwise_costs,
            session_sizes=subject.session_sizes,
            start_cost=candidate.start_cost,
            end_cost=candidate.end_cost,
            gap_penalty=candidate.gap_penalty,
            cost_threshold=candidate.cost_threshold,
        )
        predicted = tracks_to_suite2p_index_matrix(result.tracks, subject.sessions)
        subject_scores.append(_score_prediction_against_reference(predicted, subject.reference, config=fold_config))
    return _aggregate_scores(subject_scores)


def _aggregate_scores(subject_scores: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    aggregate: dict[str, float | int] = {"subjects": len(subject_scores)}
    for prefix in ("pairwise", "complete_track"):
        tp = sum(int(scores.get(f"{prefix}_true_positives", 0)) for scores in subject_scores)
        fp = sum(int(scores.get(f"{prefix}_false_positives", 0)) for scores in subject_scores)
        fn = sum(int(scores.get(f"{prefix}_false_negatives", 0)) for scores in subject_scores)
        aggregate[f"{prefix}_true_positives"] = tp
        aggregate[f"{prefix}_false_positives"] = fp
        aggregate[f"{prefix}_false_negatives"] = fn
        aggregate[f"{prefix}_f1"] = _f1(tp, fp, fn)
    aggregate["mean_f1"] = 0.5 * (float(aggregate["pairwise_f1"]) + float(aggregate["complete_track_f1"]))
    return aggregate


def _objective_tuple(scores: Mapping[str, float | int], objective: SolverPriorObjective) -> tuple[float, float, float]:
    return (float(scores[objective]), float(scores["complete_track_f1"]), float(scores["pairwise_f1"]))


def _load_subject_data(subject_dir: Path, *, config: Track2pBenchmarkConfig) -> SubjectSolverPriorData:
    sessions = tuple(_load_subject_sessions(subject_dir, config))
    reference = _load_reference_for_subject(subject_dir, data_root=config.data, config=config)
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
        _validate_reference_roi_indices(reference, sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError(f"Subject {subject_dir.name!r} has {len(sessions)} loaded sessions but {reference.n_sessions} reference sessions")
    return SubjectSolverPriorData(subject_dir=subject_dir, sessions=sessions, reference=reference)


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else float(2 * tp / denom)


def parse_positive_list(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(_parse_float(token, name=name) for token in _split(raw, name=name))
    return _positive_values(values, name=name)


def parse_nonnegative_list(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(_parse_float(token, name=name) for token in _split(raw, name=name))
    return _nonnegative_values(values, name=name)


def parse_threshold_list(raw: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for token in _split(raw, name="--cost-thresholds"):
        values.append(None if token.lower() in {"none", "null", "off", "disabled"} else _parse_float(token, name="--cost-thresholds"))
    return _thresholds(tuple(values))


def _split(raw: str, *, name: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a comma-separated list without empty entries")
    return tokens


def _parse_float(token: str, *, name: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value: {token!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} values must be finite")
    return value


def _positive_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value <= 0.0 or not np.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be positive finite numbers")
    return result


def _nonnegative_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value < 0.0 or not np.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be non-negative finite numbers")
    return result


def _thresholds(values: Sequence[float | None]) -> tuple[float | None, ...]:
    result = tuple(None if value is None else float(value) for value in values)
    if not result or any(value is not None and not np.isfinite(value) for value in result):
        raise ValueError("cost thresholds must be finite numbers or none")
    return result


def threshold_label(threshold: float | None) -> float | str:
    return "none" if threshold is None else float(threshold)


def _write_rows(rows: Sequence[dict[str, float | int | str]], output: Path | None, output_format: str) -> None:
    if output_format == "json":
        text = json.dumps(list(rows), indent=2) + "\n"
    elif output_format == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
    else:
        text = format_benchmark_table(rows) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "held_out_subject",
        "training_subjects",
        "pairwise_f1",
        "complete_track_f1",
        "learned_start_cost",
        "learned_end_cost",
        "learned_gap_penalty",
        "learned_cost_threshold",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
