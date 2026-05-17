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
    session_edge_pairs,
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
) -> LosoCalibrationResult:
    """Run calibrated global assignment with leave-one-subject-out model fitting.

    Hard-negative candidate limiting already changes the class prior seen by the
    logistic model. By default LOSO calibration therefore does not add
    inverse-frequency sample weights, and it disables PyRecEst's implicit
    ``class_weight='balanced'`` behavior unless the caller explicitly overrides
    ``model_kwargs['class_weight']``.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO calibration requires at least two subject directories")

    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 2),
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
            training_labels,
            sample_weight=sample_weight,
            strategy=sample_weight_strategy,
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
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=calibrated_model,
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
                    variant="Calibrated costs + LOSO global assignment",
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
