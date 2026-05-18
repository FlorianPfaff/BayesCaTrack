"""Leave-one-subject-out calibrated Track2p benchmark folds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    collect_reference_training_examples,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    build_registered_pairwise_costs,
    session_edge_pairs,
    solve_global_assignment_from_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    balanced_binary_sample_weights,
    collect_candidate_limited_training_examples,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    solve_configured_global_assignment,
)
from bayescatrack.reference import Track2pReference

SampleWeightStrategy = Literal["none", "balanced"]
SolverPriorObjective = Literal["pairwise_f1", "complete_track_f1"]

DEFAULT_SOLVER_PRIOR_START_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_END_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_GAP_PENALTIES = (0.6, 1.0, 0.0)
DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS = (2.0, 4.0, 6.0, None)


@dataclass(frozen=True)
class SubjectCalibrationData:
    """Loaded sessions and reference identities for one subject."""

    subject_dir: Path
    sessions: tuple[Track2pSession, ...]
    reference: Track2pReference

    @property
    def subject_name(self) -> str:
        return self.subject_dir.name


@dataclass(frozen=True)
class SolverPriorParameters:
    """Global assignment prior constants selected inside a training fold."""

    start_cost: float
    end_cost: float
    gap_penalty: float
    cost_threshold: float | None


@dataclass(frozen=True)
class SolverPriorTuningOptions:
    """Candidate-grid and objective for training-fold solver-prior tuning."""

    objective: SolverPriorObjective = "pairwise_f1"
    start_costs: tuple[float, ...] | None = None
    end_costs: tuple[float, ...] | None = None
    gap_penalties: tuple[float, ...] | None = None
    cost_thresholds: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class SolverPriorTuningResult:
    """Best solver-prior setting and mean training-fold scores."""

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
    """Training subject with expensive pairwise costs cached once."""

    subject: SubjectCalibrationData
    pairwise_costs: Mapping[tuple[int, int], np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LosoCalibrationFold:
    """One leave-one-subject-out calibrated-association fold."""

    held_out_subject: str
    training_subjects: tuple[str, ...]
    benchmark: SubjectBenchmarkResult
    training_examples: int
    positive_examples: int

    @property
    def negative_examples(self) -> int:
        return int(self.training_examples - self.positive_examples)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            **self.benchmark.to_dict(),
            "held_out_subject": self.held_out_subject,
            "training_subjects": ",".join(self.training_subjects),
            "training_examples": int(self.training_examples),
            "positive_examples": int(self.positive_examples),
            "negative_examples": self.negative_examples,
        }


@dataclass(frozen=True)
class LosoCalibrationResult:
    """All folds from a leave-one-subject-out calibration run."""

    folds: tuple[LosoCalibrationFold, ...]
    feature_names: tuple[str, ...]
    max_gap: int

    def to_rows(self) -> list[dict[str, float | int | str]]:
        return [fold.to_dict() for fold in self.folds]

    def to_benchmark_results(self) -> list[SubjectBenchmarkResult]:
        return [fold.benchmark for fold in self.folds]


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_loso_calibration(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kwargs: Mapping[str, Any] | None = None,
    tune_solver_priors: bool = False,
    solver_prior_options: SolverPriorTuningOptions | None = None,
) -> LosoCalibrationResult:
    """Run calibrated global assignment with leave-one-subject-out model fitting.

    Hard-negative candidate limiting already changes the class prior seen by the
    logistic model. By default LOSO calibration therefore does not add
    inverse-frequency sample weights, and it disables PyRecEst's implicit
    ``class_weight='balanced'`` behavior unless the caller explicitly overrides
    ``model_kwargs['class_weight']``.

    When ``tune_solver_priors`` is true, start/end/gap/threshold constants are
    selected independently inside each LOSO fold using only the training
    subjects. The held-out subject is then solved once with the selected
    constants.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO calibration requires at least two subject directories")

    per_fold_steps = len(subject_dirs) + 2 + int(bool(tune_solver_priors))
    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * per_fold_steps,
        enabled=config.progress,
        label="LOSO",
    )
    subject_list: list[SubjectCalibrationData] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subject_list.append(_load_subject_calibration_data(subject_dir, config=config))
    subjects = tuple(subject_list)
    feature_names = tuple(feature_names)
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    logistic_model_kwargs = _loso_logistic_model_kwargs(model_kwargs)
    solver_prior_options = solver_prior_options or SolverPriorTuningOptions()
    folds: list[LosoCalibrationFold] = []

    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
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
        progress.step(f"scoring calibration for {held_out.subject_name}")
        calibration_scores = _score_holdout_calibration(
            calibrated_model,
            held_out,
            config=config,
            feature_names=feature_names,
        )
        tuned_priors: SolverPriorTuningResult | None = None
        solver_kwargs = {
            "start_cost": config.start_cost,
            "end_cost": config.end_cost,
            "gap_penalty": config.gap_penalty,
            "cost_threshold": config.cost_threshold,
        }
        if tune_solver_priors:
            progress.step(f"tuning solver priors for {held_out.subject_name}")
            tuned_priors = tune_solver_priors_for_training_subjects(
                training_subjects,
                config=config,
                cost="calibrated",
                calibrated_model=calibrated_model,
                options=solver_prior_options,
            )
            solver_kwargs = {
                "start_cost": tuned_priors.parameters.start_cost,
                "end_cost": tuned_priors.parameters.end_cost,
                "gap_penalty": tuned_priors.parameters.gap_penalty,
                "cost_threshold": tuned_priors.parameters.cost_threshold,
            }

        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=calibrated_model,
            **solver_kwargs,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, held_out.reference, config=config
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
            **({} if tuned_priors is None else tuned_priors.to_score_dict()),
            **calibration_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant=(
                        "Calibrated costs + LOSO tuned-prior global assignment"
                        if tune_solver_priors
                        else "Calibrated costs + LOSO global assignment"
                    ),
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


def _validate_sample_weight_strategy(strategy: str) -> SampleWeightStrategy:
    if strategy not in {"none", "balanced"}:
        raise ValueError("sample_weight_strategy must be either 'none' or 'balanced'")
    return cast(SampleWeightStrategy, strategy)


def _loso_logistic_model_kwargs(
    model_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return logistic kwargs that avoid implicit rebalancing by default."""

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("class_weight", None)
    return kwargs


def _training_sample_weight(
    labels: np.ndarray,
    *,
    sample_weight: Any | None,
    strategy: SampleWeightStrategy,
) -> Any | None:
    if sample_weight is not None:
        if strategy != "none":
            raise ValueError(
                "sample_weight_strategy must be 'none' when explicit sample_weight is supplied"
            )
        return sample_weight
    if strategy == "none":
        return None
    if strategy == "balanced":
        return balanced_binary_sample_weights(labels)
    raise ValueError(f"Unsupported sample_weight_strategy: {strategy!r}")


def _stringify_class_weight(class_weight: Any) -> str:
    return "None" if class_weight is None else str(class_weight)


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
        session_edges=session_edge_pairs(
            len(held_out.sessions), max_gap=config.max_gap
        ),
        options=_reference_training_options(config, feature_names),
    )
    probabilities = np.asarray(
        calibrated_model.model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    return calibration_summary(probabilities, np.asarray(labels).reshape(-1))


def _load_subject_calibration_data(
    subject_dir: Path, *, config: Track2pBenchmarkConfig
) -> SubjectCalibrationData:
    sessions = tuple(_load_subject_sessions(subject_dir, config))
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
        _validate_reference_roi_indices(reference, sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            f"Subject {subject_dir.name!r} has {len(sessions)} loaded sessions but "
            f"{reference.n_sessions} reference sessions"
        )
    return SubjectCalibrationData(
        subject_dir=subject_dir, sessions=sessions, reference=reference
    )


# pylint: disable=too-many-arguments
def _collect_training_examples(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    progress: ProgressReporter | None = None,
    held_out_subject: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    training_options = _reference_training_options(config, feature_names)
    hard_negative_options = CandidateHardNegativeOptions()
    for subject in training_subjects:
        if progress is not None:
            progress.step(
                f"collecting {subject.subject_name} training features for {held_out_subject}"
            )
        pairwise_blocks = collect_reference_pairwise_example_blocks(
            subject.sessions,
            subject.reference,
            session_edges=session_edge_pairs(
                len(subject.sessions), max_gap=config.max_gap
            ),
            options=training_options,
        )
        features, labels = collect_candidate_limited_training_examples(
            pairwise_blocks, options=hard_negative_options
        )
        feature_blocks.append(features)
        label_blocks.append(labels)
    if not feature_blocks:
        raise ValueError("At least one training subject is required")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def _reference_training_options(
    config: Track2pBenchmarkConfig, feature_names: Sequence[str]
) -> ReferenceTrainingOptions:
    return ReferenceTrainingOptions(
        curated_only=config.curated_only,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        feature_names=tuple(feature_names),
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
    )


def tune_solver_priors_for_training_subjects(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None = None,
    options: SolverPriorTuningOptions | None = None,
) -> SolverPriorTuningResult:
    """Tune global-assignment priors on LOSO training subjects only.

    Pairwise costs are built once per training subject. Each candidate then
    reuses those matrices and only re-runs the global path-cover solver.
    """

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
        mean_scores = _score_solver_prior_candidate(
            candidate,
            cached_subjects,
            config=config,
        )
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
        SolverPriorParameters(
            start_cost=start,
            end_cost=end,
            gap_penalty=gap_penalty,
            cost_threshold=threshold,
        )
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
    values = tuple(configured) if configured is not None else _with_current_float(defaults, current)
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
    values = tuple(configured) if configured is not None else _with_current_threshold(defaults, current)
    values = _dedupe_threshold_values(values)
    if not values:
        raise ValueError("At least one solver prior cost threshold is required")
    for value in values:
        if value is not None and ((not np.isfinite(value)) or value < 0.0):
            raise ValueError("solver prior cost thresholds must be non-negative finite numbers or none")
    return values


def _with_current_float(defaults: Sequence[float], current: float) -> tuple[float, ...]:
    return _dedupe_float_values((*defaults, float(current)))


def _with_current_threshold(
    defaults: Sequence[float | None], current: float | None
) -> tuple[float | None, ...]:
    current_value = None if current is None else float(current)
    return _dedupe_threshold_values((*defaults, current_value))


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
