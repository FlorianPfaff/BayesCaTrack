"""Calibrated pairwise association costs for Track2p-style ROI linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.association._pyrecest_feature_compat import (
    CalibratedPairwiseAssociationModel,
    FeatureTransform,
    NamedPairwiseFeatureSchema,
    load_logistic_pairwise_association_model,
)
from bayescatrack.association._pyrecest_feature_compat import (
    pairwise_feature_tensor as pyrecest_pairwise_feature_tensor,
)
from bayescatrack.association.activity_similarity import (
    add_activity_similarity_components,
)
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.core.bridge import (
    SessionAssociationBundle,
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.reference import Track2pReference
from bayescatrack.track2p_registration import register_plane_pair

_ACTIVITY_FEATURES = {
    "activity_correlation",
    "activity_similarity",
    "activity_similarity_cost",
    "activity_similarity_available",
}

DEFAULT_ASSOCIATION_FEATURES = (
    "centroid_distance",
    "mahalanobis_centroid_distance",
    "one_minus_iou",
    "one_minus_mask_cosine",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity_cost",
    "activity_similarity_available",
    "session_gap",
)


@dataclass(frozen=True)
class CalibratedAssociationModel:
    """PyRecEst logistic model plus the feature schema used to fit it."""

    model: Any
    feature_names: tuple[str, ...] = DEFAULT_ASSOCIATION_FEATURES

    @property
    def schema(self) -> NamedPairwiseFeatureSchema:
        """Named PyRecEst feature schema used by this calibrated model."""

        return pairwise_feature_schema(self.feature_names)

    def _pyrecest_model(self) -> CalibratedPairwiseAssociationModel:
        return CalibratedPairwiseAssociationModel(self.model, schema=self.schema)

    def pairwise_cost_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        return np.asarray(
            self._pyrecest_model().pairwise_cost_matrix_from_components(
                pairwise_components
            ),
            dtype=float,
        )

    def pairwise_cost_matrix_from_bundle(
        self,
        bundle: SessionAssociationBundle,
        *,
        session_gap: int | float = 1.0,
    ) -> np.ndarray:
        """Convert a full association bundle into calibrated assignment costs."""

        return self.pairwise_cost_matrix_from_components(
            pairwise_components_from_bundle(bundle, session_gap=session_gap)
        )

    def pairwise_probability_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        """Convert pairwise components into calibrated match probabilities."""

        return np.asarray(
            self._pyrecest_model().pairwise_probability_matrix_from_components(
                pairwise_components
            ),
            dtype=float,
        )

    def pairwise_probability_matrix_from_bundle(
        self,
        bundle: SessionAssociationBundle,
        *,
        session_gap: int | float = 1.0,
    ) -> np.ndarray:
        """Convert a full association bundle into calibrated match probabilities."""

        return self.pairwise_probability_matrix_from_components(
            pairwise_components_from_bundle(bundle, session_gap=session_gap)
        )

    def predict_match_probability(self, features: Any) -> np.ndarray:
        """Return calibrated match probabilities for feature vectors or pairwise tensors."""

        return np.asarray(
            self._pyrecest_model().predict_match_probability(features), dtype=float
        )


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class ReferenceTrainingOptions:
    """Registration and feature options used to collect calibrated training examples."""

    curated_only: bool = False
    transform_type: str = "affine"
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    feature_names: tuple[str, ...] = DEFAULT_ASSOCIATION_FEATURES
    pairwise_cost_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReferencePairwiseExamples:
    """Labeled candidate ROI pairs for one registered session edge."""

    session_a: int
    session_b: int
    features: np.ndarray
    labels: np.ndarray
    reference_roi_indices: np.ndarray
    measurement_roi_indices: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def gap(self) -> int:
        return int(self.session_b - self.session_a)


def pairwise_feature_schema(
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
) -> NamedPairwiseFeatureSchema:
    """Return the PyRecEst named feature schema for BayesCaTrack components."""

    names = tuple(feature_names)
    return NamedPairwiseFeatureSchema(names, transforms=_feature_transforms_for(names))


def pairwise_components_from_bundle(
    bundle: SessionAssociationBundle,
    *,
    covariance_epsilon: float = 1.0e-6,
    session_gap: int | float = 1.0,
) -> dict[str, np.ndarray]:
    """Return base pairwise components augmented with covariance-shape and session-gap terms."""

    components = {
        key: np.asarray(value) for key, value in bundle.pairwise_components.items()
    }
    covariance_shape_cost, covariance_logdet_cost, covariance_shape_similarity = (
        _pairwise_covariance_shape_components(
            _reference_position_covariances(bundle.reference_state_covariances),
            _measurement_position_covariances(bundle.measurement_covariances),
            epsilon=covariance_epsilon,
        )
    )
    components.setdefault("covariance_shape_cost", covariance_shape_cost)
    components.setdefault("covariance_logdet_cost", covariance_logdet_cost)
    components.setdefault("covariance_shape_similarity", covariance_shape_similarity)
    return with_session_gap_component(components, session_gap=session_gap)


def with_session_gap_component(
    pairwise_components: Mapping[str, Any],
    *,
    session_gap: int | float,
) -> dict[str, np.ndarray]:
    """Return pairwise components augmented with a constant positive session-gap plane."""

    session_gap = float(session_gap)
    if session_gap <= 0.0:
        raise ValueError("session_gap must be positive")
    components = {key: np.asarray(value) for key, value in pairwise_components.items()}
    if not components:
        raise ValueError(
            "At least one pairwise component is required to infer the session-gap shape"
        )
    reference_shape = next(iter(components.values())).shape
    if len(reference_shape) != 2:
        raise ValueError("Pairwise components must be two-dimensional")
    components["session_gap"] = np.full(reference_shape, session_gap, dtype=float)
    return components


def pairwise_feature_tensor(
    pairwise_components: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
) -> np.ndarray:
    """Build a ``(n_reference, n_measurement, n_features)`` tensor from pairwise components."""

    return np.asarray(
        pyrecest_pairwise_feature_tensor(
            pairwise_components, pairwise_feature_schema(feature_names)
        ),
        dtype=float,
    )


def label_matrix_from_reference(
    reference: Track2pReference,
    session_a: int,
    session_b: int,
    *,
    reference_roi_indices: Any,
    measurement_roi_indices: Any,
    curated_only: bool = False,
) -> np.ndarray:
    """Return a binary match-label matrix in loaded-ROI coordinates."""

    reference_indices = np.asarray(reference_roi_indices, dtype=int).reshape(-1)
    measurement_indices = np.asarray(measurement_roi_indices, dtype=int).reshape(-1)
    labels = np.zeros(
        (reference_indices.shape[0], measurement_indices.shape[0]), dtype=int
    )
    reference_lookup = {
        int(roi_index): row for row, roi_index in enumerate(reference_indices)
    }
    measurement_lookup = {
        int(roi_index): col for col, roi_index in enumerate(measurement_indices)
    }
    for roi_a, roi_b in reference.pairwise_matches(
        session_a, session_b, curated_only=curated_only
    ):
        row = reference_lookup.get(int(roi_a))
        col = measurement_lookup.get(int(roi_b))
        if row is not None and col is not None:
            labels[row, col] = 1
    return labels


# pylint: disable=too-many-arguments
def collect_reference_training_examples(
    sessions: Sequence[Track2pSession],
    reference: Track2pReference,
    *,
    session_edges: Sequence[tuple[int, int]],
    options: ReferenceTrainingOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect pairwise feature vectors and binary labels from Track2p reference identities."""

    example_blocks = collect_reference_pairwise_example_blocks(
        sessions,
        reference,
        session_edges=session_edges,
        options=options,
    )
    feature_blocks = [
        block.features.reshape(-1, block.features.shape[-1]) for block in example_blocks
    ]
    label_blocks = [block.labels.reshape(-1) for block in example_blocks]
    if not feature_blocks:
        raise ValueError("At least one training edge is required")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def collect_reference_pairwise_example_blocks(
    sessions: Sequence[Track2pSession],
    reference: Track2pReference,
    *,
    session_edges: Sequence[tuple[int, int]],
    options: ReferenceTrainingOptions | None = None,
) -> tuple[ReferencePairwiseExamples, ...]:
    """Collect labeled pairwise feature tensors with ROI and session metadata."""

    sessions = list(sessions)
    options = options or ReferenceTrainingOptions()
    blocks: list[ReferencePairwiseExamples] = []
    for session_a, session_b in session_edges:
        if session_a < 0 or session_b >= len(sessions) or session_a >= session_b:
            raise ValueError(f"Invalid training edge {(session_a, session_b)}")
        bundle = _build_training_bundle(sessions, session_a, session_b, options)
        components = pairwise_components_from_bundle(
            bundle, session_gap=session_b - session_a
        )
        features = pairwise_feature_tensor(
            components, feature_names=options.feature_names
        )
        labels = label_matrix_from_reference(
            reference,
            session_a,
            session_b,
            reference_roi_indices=bundle.reference_roi_indices,
            measurement_roi_indices=bundle.measurement_roi_indices,
            curated_only=options.curated_only,
        )
        blocks.append(
            ReferencePairwiseExamples(
                session_a=int(session_a),
                session_b=int(session_b),
                features=features,
                labels=labels,
                reference_roi_indices=np.asarray(
                    bundle.reference_roi_indices, dtype=int
                ).reshape(-1),
                measurement_roi_indices=np.asarray(
                    bundle.measurement_roi_indices, dtype=int
                ).reshape(-1),
                feature_names=tuple(options.feature_names),
            )
        )
    if not blocks:
        raise ValueError("At least one training edge is required")
    return tuple(blocks)


def fit_logistic_association_model(
    features: Any,
    labels: Any,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> CalibratedAssociationModel:
    """Fit PyRecEst's logistic pairwise association model and keep its feature schema."""

    model_class = load_logistic_pairwise_association_model()
    model = model_class(**dict(model_kwargs or {}))
    model.fit(features, labels, sample_weight=sample_weight)
    return CalibratedAssociationModel(model=model, feature_names=tuple(feature_names))


# pylint: disable=too-many-arguments
def fit_logistic_association_model_from_reference(
    sessions: Sequence[Track2pSession],
    reference: Track2pReference,
    *,
    session_edges: Sequence[tuple[int, int]],
    options: ReferenceTrainingOptions | None = None,
    sample_weight: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> CalibratedAssociationModel:
    """Fit a calibrated association model from Track2p reference identities."""

    options = options or ReferenceTrainingOptions()
    features, labels = collect_reference_training_examples(
        sessions, reference, session_edges=session_edges, options=options
    )
    return fit_logistic_association_model(
        features,
        labels,
        feature_names=options.feature_names,
        sample_weight=sample_weight,
        model_kwargs=model_kwargs,
    )


def calibrated_cost_matrix_from_bundle(
    bundle: SessionAssociationBundle,
    calibrated_model: CalibratedAssociationModel,
    *,
    session_gap: int | float = 1.0,
) -> np.ndarray:
    """Return calibrated assignment costs for one registration-aware association bundle."""

    return calibrated_model.pairwise_cost_matrix_from_bundle(
        bundle, session_gap=session_gap
    )


def _feature_transforms_for(
    feature_names: Sequence[str],
) -> dict[str, FeatureTransform]:
    transforms: dict[str, FeatureTransform] = {}
    for feature_name in feature_names:
        if feature_name == "one_minus_iou":
            transforms[feature_name] = lambda components: 1.0 - _finite_component(
                components, "iou"
            )
        elif feature_name == "one_minus_mask_cosine":
            transforms[feature_name] = lambda components: 1.0 - _finite_component(
                components, "mask_cosine_similarity"
            )
        elif feature_name == "one_minus_covariance_shape_similarity":
            transforms[feature_name] = lambda components: 1.0 - _finite_component(
                components, "covariance_shape_similarity"
            )
        elif feature_name == "one_minus_activity_similarity":
            transforms[feature_name] = lambda components: 1.0 - _finite_component(
                components, "activity_similarity"
            )
        elif feature_name in _ACTIVITY_FEATURES:
            transforms[feature_name] = _optional_zero_component_transform(feature_name)
        elif feature_name == "session_gap":
            transforms[feature_name] = _session_gap_transform
    return transforms


def _optional_zero_component_transform(
    feature_name: str,
) -> FeatureTransform:
    def transform(pairwise_components: Mapping[str, Any]) -> np.ndarray:
        if feature_name not in pairwise_components:
            return _zero_like_pairwise_component(pairwise_components)
        return _finite_component(pairwise_components, feature_name)

    return transform


def _session_gap_transform(pairwise_components: Mapping[str, Any]) -> np.ndarray:
    if "session_gap" not in pairwise_components:
        return np.ones_like(_zero_like_pairwise_component(pairwise_components))
    return _finite_component(pairwise_components, "session_gap")


def _build_training_bundle(
    sessions: Sequence[Track2pSession],
    session_a: int,
    session_b: int,
    options: ReferenceTrainingOptions,
) -> SessionAssociationBundle:
    registered_measurement_plane = register_plane_pair(
        sessions[session_a].plane_data,
        sessions[session_b].plane_data,
        transform_type=options.transform_type,
    )
    registered_measurement_plane, _ = replace_empty_registered_masks(
        registered_measurement_plane
    )
    bundle = build_session_pair_association_bundle(
        sessions[session_a],
        sessions[session_b],
        measurement_plane_in_reference_frame=registered_measurement_plane,
        order=options.order,
        weighted_centroids=options.weighted_centroids,
        velocity_variance=options.velocity_variance,
        regularization=options.regularization,
        pairwise_cost_kwargs=options.pairwise_cost_kwargs,
        return_pairwise_components=True,
    )
    add_activity_similarity_components(
        bundle.pairwise_components,
        sessions[session_a].plane_data,
        registered_measurement_plane,
    )
    return bundle


def _finite_component(
    pairwise_components: Mapping[str, Any], component_name: str
) -> np.ndarray:
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"Pairwise component {component_name!r} must be two-dimensional"
        )
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _zero_like_pairwise_component(pairwise_components: Mapping[str, Any]) -> np.ndarray:
    for values in pairwise_components.values():
        array_values = np.asarray(values)
        if array_values.ndim == 2:
            return np.zeros(array_values.shape, dtype=float)
    raise KeyError(
        "Pairwise components do not contain any two-dimensional matrix to infer feature shape"
    )


def _reference_position_covariances(reference_state_covariances: Any) -> np.ndarray:
    covariances = np.asarray(reference_state_covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[:2] != (4, 4):
        raise ValueError("reference_state_covariances must have shape (4, 4, n_roi)")
    return covariances[[0, 2], :, :][:, [0, 2], :]


def _measurement_position_covariances(measurement_covariances: Any) -> np.ndarray:
    covariances = np.asarray(measurement_covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[:2] != (2, 2):
        raise ValueError("measurement_covariances must have shape (2, 2, n_roi)")
    return covariances


def _pairwise_covariance_shape_components(
    covariances_self: np.ndarray, covariances_other: np.ndarray, *, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return covariance-based ROI shape and scale-difference components."""

    covariances_self = np.asarray(covariances_self, dtype=float)
    covariances_other = np.asarray(covariances_other, dtype=float)
    if covariances_self.ndim != 3 or covariances_self.shape[:2] != (2, 2):
        raise ValueError("covariances_self must have shape (2, 2, n_roi)")
    if covariances_other.ndim != 3 or covariances_other.shape[:2] != (2, 2):
        raise ValueError("covariances_other must have shape (2, 2, n_roi)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    n_self = covariances_self.shape[2]
    n_other = covariances_other.shape[2]
    if n_self == 0 or n_other == 0:
        empty = np.zeros((n_self, n_other), dtype=float)
        return empty, empty.copy(), empty.copy()
    covariances_self = 0.5 * (covariances_self + np.swapaxes(covariances_self, 0, 1))
    covariances_other = 0.5 * (covariances_other + np.swapaxes(covariances_other, 0, 1))
    moved_self = np.moveaxis(covariances_self, -1, 0)
    moved_other = np.moveaxis(covariances_other, -1, 0)
    traces_self = np.maximum(np.trace(moved_self, axis1=1, axis2=2), epsilon)
    traces_other = np.maximum(np.trace(moved_other, axis1=1, axis2=2), epsilon)
    normalized_self = moved_self / traces_self[:, None, None]
    normalized_other = moved_other / traces_other[:, None, None]
    shape_diffs = normalized_self[:, None, :, :] - normalized_other[None, :, :, :]
    covariance_shape_cost = np.linalg.norm(shape_diffs, axis=(2, 3)) / np.sqrt(2.0)
    covariance_shape_similarity = np.exp(-covariance_shape_cost)
    determinants_self = np.maximum(np.linalg.det(moved_self), epsilon)
    determinants_other = np.maximum(np.linalg.det(moved_other), epsilon)
    covariance_logdet_cost = np.abs(
        np.log(determinants_self[:, None] / determinants_other[None, :])
    )
    return (
        np.nan_to_num(covariance_shape_cost, nan=0.0, posinf=1.0e6, neginf=0.0),
        np.nan_to_num(covariance_logdet_cost, nan=0.0, posinf=1.0e6, neginf=0.0),
        np.nan_to_num(covariance_shape_similarity, nan=0.0, posinf=1.0, neginf=0.0),
    )
