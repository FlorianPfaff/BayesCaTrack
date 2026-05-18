"""LOSO solver-prior tuning for calibrated Track2p global assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    build_registered_pairwise_costs,
    session_edge_pairs,
    solve_global_assignment_from_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _score_prediction_against_reference,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SubjectCalibrationData,
    _collect_training_examples,
    _load_subject_calibration_data,
    _loso_logistic_model_kwargs,
    _reference_training_options,
    _stringify_class_weight,
    _training_sample_weight,
    _validate_sample_weight_strategy,
)

SampleWeightStrategy = Literal["none", "balanced"]
SolverPriorObjective = Literal["pairwise_f1", "complete_track_f1"]

DEFAULT_SOLVER_PRIOR_START_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_END_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_GAP_PENALTIES = (0.6, 1.0, 0.0)
DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS = (2.0, 4.0, 6.0, None)


@dataclass(frozen=True)
class SolverPriorParameters:
    """Global assignment prior constants selected inside a training fold."""

    start_cost: float
    end_cost: float
    gap_penalty: float
    cost_threshold: float | None


@dataclass(frozen=True)
class SolverPriorTuningOptions:
    """Candidate grid and objective for fold-internal solver-prior tuning."""

    objective: SolverPriorObjective = "pairwise_f1"
    start_costs: tuple[float, ...] | None = None
    end_costs: tuple[float, ...] | None = None
    gap_penalties: tuple[float, ...] | None = None
    cost_thresholds: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class SolverPriorTuningResult:
    """Best solver-prior setting and mean training-subject scores."""

    parameters: SolverPriorParameters
    objective_name: SolverPriorObjective
    objective_value: float
    candidate_count: int
    mean_scores: Mapping[str, float]

    def to_score_dict(self) -> dict[str, float | int | str]:
        return {
            "solver_prior_objective": self.objective_name,
            "solver_prior_objective_score": float(self.objective_value),
            "solver_prior_candidate_count": int(self.candidate_count),
            "tuned_start_cost": float(self.parameters.start_cost),
            "tuned_end_cost": float(self.parameters.end_cost),
            "tuned_gap_penalty": float(self.parameters.gap_penalty),
            "tuned_cost_threshold": _threshold_label(self.parameters.cost_threshold),
        }


@dataclass(frozen=True)
class CachedSolverTuningSubject:
    """Training subject with expensive registered pairwise costs cached once."""

    subject: SubjectCalibrationData
    pairwise_costs: Mapping[tuple[int, int], np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[tuple[int, int], ...]


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_loso_solver_prior_tuning(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kwargs: Mapping[str, Any] | None = None,
    solver_prior_options: SolverPriorTuningOptions | None = None,
) -> LosoCalibrationResult:
    """Run calibrated LOSO while tuning solver priors on training subjects.

    This experiment keeps classifier fitting and held-out evaluation identical to
    the calibrated LOSO benchmark, but selects the global assignment start/end,
    gap, and threshold constants using only the training subjects in each fold.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO solver-prior tuning requires method='global-assignment' and cost='calibrated'"
        )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO solver-prior tuning requires at least two subject directories")

    feature_names = tuple(feature_names)
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    logistic_model_kwargs = _loso_logistic_model_kwargs(model_kwargs)
    solver_prior_options = solver_prior_options or SolverPriorTuningOptions()
    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 3),
        enabled=config.progress,
        label="LOSO-priors",
    )

    subjects = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(_load_subject_calibration_data(subject_dir, config=config))
    subject_data = tuple(subjects)
    folds: list[LosoCalibrationFold] = []

    for held_out_index, held_out in enumerate(subject_data):
        training_subjects = tuple(
            subject for index, subject in enumerate(subject_data) if index != held_out_index
        )
        training_features, training_labels = _collect_training_examples(
            training_subjects,
            config=config,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out.subject_name,
        )
        weights = _training_sample_weight(
            training_labels, sample_weight=sample_weight, strategy=sample_weight_strategy
        )
        progress.step(f"fitting model for {held_out.subject_name}")
        calibrated_model = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kwargs=logistic_model_kwargs,
        )
        progress.step(f"tuning solver priors for {held_out.subject_name}")
        tuning_result = tune_solver_priors_for_training_subjects(
            training_subjects,
            config=config,
            cost="calibrated",
            calibrated_model=calibrated_model,
            options=solver_prior_options,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = _solve_loso_assignment_with_priors(
            held_out,
            config=config,
            calibrated_model=calibrated_model,
            parameters=tuning_result.parameters,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, held_out.reference, config=config
        )
        calibration_scores = _score_holdout_calibration(
            calibrated_model,
            held_out,
            config=config,
            feature_names=feature_names,
        )
        positives = int(np.sum(training_labels))
        scores: dict[str, float | int | str] = {
            **base_scores,
            "training_examples": int(training_labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(training_labels.shape[0] - positives),
            "calibration_sample_weight_strategy": sample_weight_strategy,
            "calibration_class_weight": _stringify_class_weight(
                logistic_model_kwargs.get("class_weight")
            ),
            **tuning_result.to_score_dict(),
            **calibration_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(subject.subject_name for subject in training_subjects),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant="Calibrated costs + LOSO tuned-prior global assignment",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=int(training_labels.shape[0]),
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def tune_solver_priors_for_training_subjects(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None = None,
    options: SolverPriorTuningOptions | None = None,
) -> SolverPriorTuningResult:
    """Tune global-assignment priors on LOSO training subjects only."""

    if not training_subjects:
        raise ValueError("At least one training subject is required for solver-prior tuning")

    options = options or SolverPriorTuningOptions()
    candidates = _solver_prior_parameter_grid(config, options=options)
    cached_subjects = tuple(
        _cache_solver_tuning_subject(
            subject,
            config=config,
            cost=cost,
            calibrated_model=calibrated_model,
        )
        for subject in training_subjects
    )

    best: SolverPriorTuningResult | None = None
    for candidate in candidates:
        mean_scores = _score_solver_prior_candidate(candidate, cached_subjects, config=config)
        objective_value = float(mean_scores.get(options.objective, np.nan))
        if not np.isfinite(objective_value):
            continue
        if best is None or objective_value > best.objective_value:
            best = SolverPriorTuningResult(
                parameters=candidate,
                objective_name=options.objective,
                objective_value=objective_value,
                candidate_count=len(candidates),
                mean_scores=mean_scores,
            )
    if best is None:
        raise ValueError(
            f"No finite {options.objective!r} score was produced during solver-prior tuning"
        )
    return best


def _cache_solver_tuning_subject(
    subject: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None,
) -> CachedSolverTuningSubject:
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
    return CachedSolverTuningSubject(
        subject=subject,
        pairwise_costs=pairwise_costs,
        session_sizes=tuple(int(session.plane_data.n_rois) for session in subject.sessions),
        session_edges=session_edge_pairs(len(subject.sessions), max_gap=config.max_gap),
    )


def _score_solver_prior_candidate(
    candidate: SolverPriorParameters,
    cached_subjects: Sequence[CachedSolverTuningSubject],
    *,
    config: Track2pBenchmarkConfig,
) -> dict[str, float]:
    score_rows: list[Mapping[str, float | int | str]] = []
    for cached in cached_subjects:
        assignment = solve_global_assignment_from_pairwise_costs(
            cached.pairwise_costs,
            session_sizes=cached.session_sizes,
            session_edges=cached.session_edges,
            start_cost=candidate.start_cost,
            end_cost=candidate.end_cost,
            gap_penalty=candidate.gap_penalty,
            cost_threshold=candidate.cost_threshold,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, cached.subject.sessions
        )
        score_rows.append(
            _score_prediction_against_reference(
                predicted_matrix,
                cached.subject.reference,
                config=config,
            )
        )
    return _mean_numeric_scores(score_rows)


def _solve_loso_assignment_with_priors(
    held_out: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    calibrated_model: Any,
    parameters: SolverPriorParameters,
) -> Any:
    cached = _cache_solver_tuning_subject(
        held_out,
        config=config,
        cost="calibrated",
        calibrated_model=calibrated_model,
    )
    return solve_global_assignment_from_pairwise_costs(
        cached.pairwise_costs,
        session_sizes=cached.session_sizes,
        session_edges=cached.session_edges,
        start_cost=parameters.start_cost,
        end_cost=parameters.end_cost,
        gap_penalty=parameters.gap_penalty,
        cost_threshold=parameters.cost_threshold,
    )


def _score_holdout_calibration(
    calibrated_model: Any,
    held_out: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> dict[str, float | int]:
    features, labels = collect_reference_training_examples(
        held_out.sessions,
        held_out.reference,
        session_edges=session_edge_pairs(len(held_out.sessions), max_gap=config.max_gap),
        options=_reference_training_options(config, feature_names),
    )
    probabilities = np.asarray(
        calibrated_model.model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    return calibration_summary(probabilities, np.asarray(labels).reshape(-1))


def _mean_numeric_scores(rows: Sequence[Mapping[str, float | int | str]]) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    values_by_key.setdefault(key, []).append(numeric_value)
    return {key: float(np.mean(values)) for key, values in values_by_key.items() if values}


def _solver_prior_parameter_grid(
    config: Track2pBenchmarkConfig,
    *,
    options: SolverPriorTuningOptions,
) -> tuple[SolverPriorParameters, ...]:
    starts = _solver_float_grid(
        options.start_costs,
        defaults=DEFAULT_SOLVER_PRIOR_START_COSTS,
        current=config.start_cost,
        positive=True,
        name="solver prior start costs",
    )
    ends = _solver_float_grid(
        options.end_costs,
        defaults=DEFAULT_SOLVER_PRIOR_END_COSTS,
        current=config.end_cost,
        positive=True,
        name="solver prior end costs",
    )
    gaps = _solver_float_grid(
        options.gap_penalties,
        defaults=DEFAULT_SOLVER_PRIOR_GAP_PENALTIES,
        current=config.gap_penalty,
        positive=False,
        name="solver prior gap penalties",
    )
    thresholds = _solver_threshold_grid(
        options.cost_thresholds,
        defaults=DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS,
        current=config.cost_threshold,
    )
    return tuple(
        SolverPriorParameters(start, end, gap_penalty, threshold)
        for start in starts
        for end in ends
        for gap_penalty in gaps
        for threshold in thresholds
    )


def _solver_float_grid(
    configured: Sequence[float] | None,
    *,
    defaults: Sequence[float],
    current: float,
    positive: bool,
    name: str,
) -> tuple[float, ...]:
    values = tuple(configured) if configured is not None else (*defaults, float(current))
    values = _dedupe_float_values(values)
    if not values:
        raise ValueError(f"At least one {name} value is required")
    for value in values:
        invalid = value <= 0.0 if positive else value < 0.0
        if (not np.isfinite(value)) or invalid:
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} values must be {qualifier} finite numbers")
    return values


def _solver_threshold_grid(
    configured: Sequence[float | None] | None,
    *,
    defaults: Sequence[float | None],
    current: float | None,
) -> tuple[float | None, ...]:
    current_value = None if current is None else float(current)
    values = tuple(configured) if configured is not None else (*defaults, current_value)
    values = _dedupe_threshold_values(values)
    if not values:
        raise ValueError("At least one solver prior cost threshold is required")
    for value in values:
        if value is not None and ((not np.isfinite(value)) or value < 0.0):
            raise ValueError("solver prior cost thresholds must be non-negative finite numbers or none")
    return values


def _dedupe_float_values(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        numeric_value = float(value)
        if numeric_value not in result:
            result.append(numeric_value)
    return tuple(result)


def _dedupe_threshold_values(values: Sequence[float | None]) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for value in values:
        numeric_value = None if value is None else float(value)
        if numeric_value not in result:
            result.append(numeric_value)
    return tuple(result)


def _threshold_label(threshold: float | None) -> float | str:
    return "none" if threshold is None else float(threshold)
