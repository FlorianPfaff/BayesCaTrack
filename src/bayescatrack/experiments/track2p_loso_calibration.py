"""Leave-one-subject-out calibrated Track2p benchmark folds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferenceTrainingOptions,
    collect_reference_training_examples,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession, load_track2p_subject
from bayescatrack.evaluation.complete_track_scores import normalize_track_matrix, score_track_matrices
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    discover_subject_dirs,
    solve_configured_global_assignment,
)
from bayescatrack.reference import Track2pReference, load_aligned_subject_reference, load_track2p_reference


@dataclass(frozen=True)
class SubjectCalibrationData:
    """Loaded sessions and Track2p reference identities for one subject."""

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
    model_kwargs: Mapping[str, Any] | None = None,
) -> LosoCalibrationResult:
    """Run calibrated global assignment with leave-one-subject-out model fitting."""

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError("LOSO calibration requires method='global-assignment' and cost='calibrated'")

    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO calibration requires at least two subject directories")

    total_steps = len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 1)
    progress = ProgressReporter(total_steps, enabled=config.progress, label="LOSO")
    subject_data: list[SubjectCalibrationData] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subject_data.append(_load_subject_calibration_data(subject_dir, config=config))
    subjects = tuple(subject_data)
    feature_names = tuple(feature_names)
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(subject for index, subject in enumerate(subjects) if index != held_out_index)
        training_features, training_labels = _collect_training_examples(
            training_subjects,
            config=config,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out.subject_name,
        )
        progress.step(f"fitting model for {held_out.subject_name}")
        calibrated_model = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=feature_names,
            sample_weight=sample_weight,
            model_kwargs=model_kwargs,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=calibrated_model,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(assignment.result.tracks, held_out.sessions)
        scores = score_track_matrices(predicted_matrix, _reference_matrix(held_out.reference, curated_only=config.curated_only))
        scores = {
            **scores,
            "training_examples": int(training_labels.shape[0]),
            "positive_examples": int(np.sum(training_labels)),
            "negative_examples": int(training_labels.shape[0] - np.sum(training_labels)),
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(subject.subject_name for subject in training_subjects),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant="Calibrated costs + LOSO global assignment",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=int(training_labels.shape[0]),
                positive_examples=int(np.sum(training_labels)),
            )
        )

    return LosoCalibrationResult(folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap))


def _load_subject_calibration_data(subject_dir: Path, *, config: Track2pBenchmarkConfig) -> SubjectCalibrationData:
    sessions = tuple(
        load_track2p_subject(
            subject_dir,
            plane_name=config.plane_name,
            input_format=config.input_format,
            include_behavior=config.include_behavior,
            include_non_cells=config.include_non_cells,
            cell_probability_threshold=config.cell_probability_threshold,
            weighted_masks=config.weighted_masks,
            exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        )
    )
    reference = _load_training_reference(subject_dir, config=config)
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            f"Subject {subject_dir.name!r} has {len(sessions)} loaded sessions but "
            f"{reference.n_sessions} reference sessions"
        )
    return SubjectCalibrationData(subject_dir=subject_dir, sessions=sessions, reference=reference)


def _load_training_reference(subject_dir: Path, *, config: Track2pBenchmarkConfig) -> Track2pReference:
    if config.reference is None:
        if not (subject_dir / "track2p" / "track_ops.npy").exists():
            return load_aligned_subject_reference(
                subject_dir,
                plane_name=config.plane_name,
                input_format=config.input_format,
            )
        return load_track2p_reference(subject_dir / "track2p", plane_name=config.plane_name)

    for candidate in _reference_candidates(subject_dir, config.reference):
        if (candidate / "track_ops.npy").exists() or (candidate / "track2p" / "track_ops.npy").exists():
            return load_track2p_reference(candidate, plane_name=config.plane_name)
    raise FileNotFoundError(f"Could not find Track2p reference for subject {subject_dir.name!r}")


def _reference_candidates(subject_dir: Path, reference_root: Path) -> tuple[Path, ...]:
    return (
        reference_root,
        reference_root / subject_dir.name,
        reference_root / subject_dir.name / "track2p",
        reference_root / "track2p",
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
    for subject in training_subjects:
        if progress is not None:
            progress.step(f"collecting {subject.subject_name} training features for {held_out_subject}")
        features, labels = collect_reference_training_examples(
            subject.sessions,
            subject.reference,
            session_edges=session_edge_pairs(len(subject.sessions), max_gap=config.max_gap),
            options=training_options,
        )
        feature_blocks.append(features)
        label_blocks.append(labels)

    if not feature_blocks:
        raise ValueError("At least one training subject is required")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def _reference_training_options(config: Track2pBenchmarkConfig, feature_names: Sequence[str]) -> ReferenceTrainingOptions:
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


def _reference_matrix(reference: Track2pReference, *, curated_only: bool) -> np.ndarray:
    matrix = normalize_track_matrix(reference.suite2p_indices)
    if not curated_only:
        return matrix
    if reference.curated_mask is None:
        raise ValueError("curated_only was requested, but the reference has no curation mask")
    return matrix[np.asarray(reference.curated_mask, dtype=bool)]
