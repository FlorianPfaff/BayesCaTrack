"""Leave-one-subject-out calibrated Track2p benchmark folds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
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
    MonotoneRankerTrainingOptions,
    balanced_binary_sample_weights,
    collect_candidate_limited_training_examples,
    fit_monotone_hard_negative_ranker,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _activity_tie_breaker_metadata,
    _activity_variant_suffix,
    _score_prediction_against_reference,
    _assignment_registration_summary,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    _variant_name,
    discover_subject_dirs,
    solve_configured_global_assignment,
)
from bayescatrack.reference import Track2pReference

SampleWeightStrategy = Literal["none", "balanced"]
CalibrationModelKind = Literal["monotone-ranker", "logistic"]
CalibrationFeatureSet = Literal["default", "local-evidence", "default+local-evidence"]
_LOCAL_EVIDENCE_COMPONENT_KWARGS = frozenset(
    {
        "local_evidence_components",
        "weighted_dice_weight",
        "overlap_fraction_weight",
        "containment_weight",
        "distance_transform_weight",
        "image_patch_weight",
        "neighbor_constellation_weight",
        "centroid_rank_weight",
    }
)


def calibration_feature_names(
    feature_set: CalibrationFeatureSet | str = "default",
) -> tuple[str, ...]:
    """Return the calibrated-association feature names for a named preset."""

    if feature_set == "default":
        return tuple(DEFAULT_ASSOCIATION_FEATURES)
    if feature_set == "local-evidence":
        return tuple(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    if feature_set == "default+local-evidence":
        return tuple(
            dict.fromkeys(
                (*DEFAULT_ASSOCIATION_FEATURES, *LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
            )
        )
    raise ValueError(
        "calibration feature set must be 'default', 'local-evidence', or 'default+local-evidence'"
    )


def pairwise_cost_kwargs_for_calibration_features(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
    feature_names: Sequence[str],
) -> dict[str, Any] | None:
    """Return pairwise-cost kwargs needed to materialize calibrated features.

    Local-evidence calibrated features are optional-zero transforms for backward
    compatibility. They become useful only when the bundle builder is asked to
    collect the corresponding local components. This helper keeps LOSO training
    and held-out solving synchronized: selecting a local-evidence feature preset
    automatically turns on ``local_evidence_components`` while preserving any
    caller-provided pairwise-cost overrides.
    """

    kwargs = dict(pairwise_cost_kwargs or {})
    if _uses_local_evidence_features(feature_names):
        kwargs.setdefault("local_evidence_components", True)
    return kwargs or None


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
    feature_names: Sequence[str] | None = None,
    calibration_model: CalibrationModelKind | str | None = None,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kwargs: Mapping[str, Any] | None = None,
    hard_negative_options: CandidateHardNegativeOptions | None = None,
) -> LosoCalibrationResult:
    """Run calibrated global assignment with leave-one-subject-out model fitting.

    Hard-negative candidate limiting already changes the class prior seen by the
    logistic model. By default LOSO calibration therefore does not add
    inverse-frequency sample weights, and it disables PyRecEst's implicit
    ``class_weight='balanced'`` behavior for logistic calibration unless the
    caller explicitly overrides ``model_kwargs['class_weight']``.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    feature_names = tuple(
        _feature_names_from_config(config) if feature_names is None else feature_names
    )
    calibration_model = _validate_calibration_model(
        getattr(config, "calibration_model", "monotone-ranker")
        if calibration_model is None
        else calibration_model
    )
    config = _config_with_pairwise_kwargs_for_features(config, feature_names)
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
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    if calibration_model != "logistic" and (
        sample_weight is not None or sample_weight_strategy != "none"
    ):
        raise ValueError(
            "sample weights are only supported for calibration_model='logistic'"
        )
    logistic_model_kwargs = _loso_logistic_model_kwargs(model_kwargs)
    hard_negative_options = hard_negative_options or _hard_negative_options_from_config(
        config
    )
    folds: list[LosoCalibrationFold] = []

    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        training_pairwise_blocks = _collect_training_pairwise_blocks(
            training_subjects,
            config=config,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out.subject_name,
        )
        training_features, training_labels = (
            collect_candidate_limited_training_examples(
                training_pairwise_blocks, options=hard_negative_options
            )
        )
        weights = None
        if calibration_model == "logistic":
            weights = _training_sample_weight(
                training_labels,
                sample_weight=sample_weight,
                strategy=sample_weight_strategy,
            )
        progress.step(f"fitting model for {held_out.subject_name}")
        if calibration_model == "logistic":
            calibrated_model = fit_logistic_association_model(
                training_features,
                training_labels,
                feature_names=feature_names,
                sample_weight=weights,
                activity_feature_weight=config.activity_feature_weight,
                model_kwargs=logistic_model_kwargs,
            )
        else:
            calibrated_model = fit_monotone_hard_negative_ranker(
                training_pairwise_blocks,
                feature_names=feature_names,
                options=MonotoneRankerTrainingOptions(
                    hard_negative_options=hard_negative_options
                ),
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
        registration_summary = _assignment_registration_summary(assignment)
        positives = int(np.sum(training_labels))
        scores: dict[str, float | int | str] = {
            **base_scores,
            "training_examples": int(training_labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(training_labels.shape[0] - positives),
            "calibration_feature_set": _feature_set_label(config, feature_names),
            "calibration_feature_count": int(len(feature_names)),
            "calibration_model": calibration_model,
            "calibration_sample_weight_strategy": sample_weight_strategy,
            "calibration_class_weight": _stringify_class_weight(
                logistic_model_kwargs.get("class_weight")
            ),
            "calibration_hard_negative_ratio": (
                hard_negative_options.negative_to_positive_ratio
            ),
            "calibration_hard_negative_top_k": _stringify_optional_int(
                hard_negative_options.candidate_top_k_per_anchor
            ),
            "calibration_hard_negative_include_columns": str(
                hard_negative_options.include_column_candidates
            ),
            "calibration_hard_negative_features": ",".join(
                hard_negative_options.hardness_feature_names
            ),
            **_activity_tie_breaker_metadata(config),
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
                        _calibration_variant_name(
                            calibration_model,
                            higher_order_consistency_config=(
                                config.higher_order_consistency_config
                            ),
                        )
                        + _activity_variant_suffix(config)
                    ),
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                    registration_backend=registration_summary["registration_backend"],
                    registration_transform_type=registration_summary["registration_transform_type"],
                ),
                training_examples=int(training_labels.shape[0]),
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def _feature_names_from_config(config: Track2pBenchmarkConfig) -> tuple[str, ...]:
    feature_set = getattr(config, "calibration_feature_set", "default")
    if feature_set != "default":
        return calibration_feature_names(str(feature_set))
    if _pairwise_kwargs_request_local_evidence(config.pairwise_cost_kwargs):
        return calibration_feature_names("default+local-evidence")
    return calibration_feature_names("default")


def _feature_set_label(
    config: Track2pBenchmarkConfig, feature_names: Sequence[str]
) -> str:
    configured = str(getattr(config, "calibration_feature_set", "default"))
    if configured != "default":
        return configured
    if tuple(feature_names) == calibration_feature_names("default+local-evidence"):
        return "default+local-evidence"
    if tuple(feature_names) == calibration_feature_names("local-evidence"):
        return "local-evidence"
    return "custom" if tuple(feature_names) != calibration_feature_names("default") else "default"


def _pairwise_kwargs_request_local_evidence(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
) -> bool:
    if not pairwise_cost_kwargs:
        return False
    for key in _LOCAL_EVIDENCE_COMPONENT_KWARGS:
        value = pairwise_cost_kwargs.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif value is not None and float(value) > 0.0:
            return True
    return False


def _uses_local_evidence_features(feature_names: Sequence[str]) -> bool:
    local_evidence_features = set(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    return any(feature_name in local_evidence_features for feature_name in feature_names)


def _config_with_pairwise_kwargs_for_features(
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = pairwise_cost_kwargs_for_calibration_features(
        config.pairwise_cost_kwargs, feature_names
    )
    if pairwise_cost_kwargs == config.pairwise_cost_kwargs:
        return config
    return replace(config, pairwise_cost_kwargs=pairwise_cost_kwargs)


def _validate_sample_weight_strategy(strategy: str) -> SampleWeightStrategy:
    if strategy not in {"none", "balanced"}:
        raise ValueError("sample_weight_strategy must be either 'none' or 'balanced'")
    return cast(SampleWeightStrategy, strategy)


def _validate_calibration_model(model: str) -> CalibrationModelKind:
    if model not in {"monotone-ranker", "logistic"}:
        raise ValueError(
            "calibration_model must be either 'monotone-ranker' or 'logistic'"
        )
    return cast(CalibrationModelKind, model)


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


def _stringify_optional_int(value: int | None) -> str:
    return "None" if value is None else str(int(value))


def _hard_negative_options_from_config(
    config: Track2pBenchmarkConfig,
) -> CandidateHardNegativeOptions:
    return CandidateHardNegativeOptions(
        negative_to_positive_ratio=float(
            getattr(config, "calibration_hard_negative_ratio", 4.0)
        ),
        candidate_top_k_per_anchor=getattr(
            config, "calibration_hard_negative_top_k", 20
        ),
        include_column_candidates=bool(
            getattr(config, "calibration_hard_negative_include_columns", True)
        ),
        hardness_feature_names=_tuple_from_names(
            getattr(config, "calibration_hard_negative_features", ())
        ),
    )


def _tuple_from_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_names = value.split(",")
    else:
        raw_names = value
    return tuple(str(name).strip() for name in raw_names if str(name).strip())


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
    hard_negative_options: CandidateHardNegativeOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pairwise_blocks = _collect_training_pairwise_blocks(
        training_subjects,
        config=config,
        feature_names=feature_names,
        progress=progress,
        held_out_subject=held_out_subject,
    )
    hard_negative_options = hard_negative_options or _hard_negative_options_from_config(
        config
    )
    return collect_candidate_limited_training_examples(
        pairwise_blocks, options=hard_negative_options
    )


def _collect_training_pairwise_blocks(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    progress: ProgressReporter | None = None,
    held_out_subject: str | None = None,
) -> tuple[Any, ...]:
    blocks: list[Any] = []
    training_options = _reference_training_options(config, feature_names)
    for subject in training_subjects:
        if progress is not None:
            progress.step(
                f"collecting {subject.subject_name} training features for {held_out_subject}"
            )
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                subject.sessions,
                subject.reference,
                session_edges=session_edge_pairs(
                    len(subject.sessions), max_gap=config.max_gap
                ),
                options=training_options,
            )
        )
    if not blocks:
        raise ValueError("At least one training subject is required")
    return tuple(blocks)


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
        activity_feature_weight=config.activity_feature_weight,
        pairwise_cost_kwargs=pairwise_cost_kwargs_for_calibration_features(
            config.pairwise_cost_kwargs, feature_names
        ),
    )


def _calibration_variant_name(
    calibration_model: CalibrationModelKind,
    *,
    higher_order_consistency_config: Mapping[str, Any] | None = None,
) -> str:
    if calibration_model == "logistic":
        return _variant_name(
            "calibrated",
            higher_order_consistency_config=higher_order_consistency_config,
            assignment_label="LOSO global assignment",
        )
    label = "Monotone-ranker calibrated costs + LOSO global assignment"
    if higher_order_consistency_config is not None:
        label = f"{label} + higher-order consistency"
    return label
