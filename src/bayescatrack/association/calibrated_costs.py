"""Calibrated pairwise association costs for Track2p-style ROI linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayescatrack.core.bridge import (
    SessionAssociationBundle,
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.reference import Track2pReference
from bayescatrack.track2p_registration import register_plane_pair

DEFAULT_ASSOCIATION_FEATURES = (
    "centroid_distance",
    "mahalanobis_centroid_distance",
    "one_minus_iou",
    "one_minus_mask_cosine",
    "area_ratio_cost",
    "roi_feature_cost",
    "cell_probability_cost",
)


@dataclass(frozen=True)
class CalibratedAssociationModel:
    """PyRecEst logistic model plus the feature schema used to fit it."""

    model: Any
    feature_names: tuple[str, ...] = DEFAULT_ASSOCIATION_FEATURES

    def pairwise_cost_matrix_from_components(self, pairwise_components: Mapping[str, Any]) -> np.ndarray:
        """Convert BayesCaTrack pairwise components into calibrated assignment costs."""

        features = pairwise_feature_tensor(pairwise_components, feature_names=self.feature_names)
        return np.asarray(self.model.pairwise_cost_matrix(features), dtype=float)


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
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


def pairwise_feature_tensor(
    pairwise_components: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
) -> np.ndarray:
    """Build a ``(n_reference, n_measurement, n_features)`` tensor from pairwise components."""

    feature_planes = [_component_feature(pairwise_components, feature_name) for feature_name in feature_names]
    if not feature_planes:
        raise ValueError("At least one feature is required")
    reference_shape = feature_planes[0].shape
    for feature_name, feature_plane in zip(feature_names, feature_planes):
        if feature_plane.shape != reference_shape:
            raise ValueError(f"Feature {feature_name!r} has shape {feature_plane.shape}, expected {reference_shape}")
    return np.stack(feature_planes, axis=-1)


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
    labels = np.zeros((reference_indices.shape[0], measurement_indices.shape[0]), dtype=int)
    reference_lookup = {int(roi_index): row for row, roi_index in enumerate(reference_indices)}
    measurement_lookup = {int(roi_index): col for col, roi_index in enumerate(measurement_indices)}

    for roi_a, roi_b in reference.pairwise_matches(session_a, session_b, curated_only=curated_only):
        row = reference_lookup.get(int(roi_a))
        col = measurement_lookup.get(int(roi_b))
        if row is None or col is None:
            continue
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

    sessions = list(sessions)
    options = options or ReferenceTrainingOptions()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for session_a, session_b in session_edges:
        if session_a < 0 or session_b >= len(sessions) or session_a >= session_b:
            raise ValueError(f"Invalid training edge {(session_a, session_b)}")
        bundle = _build_training_bundle(sessions, session_a, session_b, options)
        features = pairwise_feature_tensor(bundle.pairwise_components, feature_names=options.feature_names)
        labels = label_matrix_from_reference(
            reference,
            session_a,
            session_b,
            reference_roi_indices=bundle.reference_roi_indices,
            measurement_roi_indices=bundle.measurement_roi_indices,
            curated_only=options.curated_only,
        )
        feature_blocks.append(features.reshape(-1, features.shape[-1]))
        label_blocks.append(labels.reshape(-1))

    if not feature_blocks:
        raise ValueError("At least one training edge is required")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def fit_logistic_association_model(
    features: Any,
    labels: Any,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> CalibratedAssociationModel:
    """Fit PyRecEst's logistic pairwise association model and keep its feature schema."""

    try:
        from pyrecest.utils.association_models import LogisticPairwiseAssociationModel
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments without PyRecEst
        raise ImportError(
            "PyRecEst with pyrecest.utils.association_models is required to fit calibrated association costs."
        ) from exc

    model = LogisticPairwiseAssociationModel(**dict(model_kwargs or {}))
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
        sessions,
        reference,
        session_edges=session_edges,
        options=options,
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
) -> np.ndarray:
    """Return calibrated assignment costs for one registration-aware association bundle."""

    return calibrated_model.pairwise_cost_matrix_from_components(bundle.pairwise_components)


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
    return build_session_pair_association_bundle(
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


def _component_feature(pairwise_components: Mapping[str, Any], feature_name: str) -> np.ndarray:
    if feature_name == "one_minus_iou":
        return 1.0 - _finite_component(pairwise_components, "iou")
    if feature_name == "one_minus_mask_cosine":
        return 1.0 - _finite_component(pairwise_components, "mask_cosine_similarity")
    return _finite_component(pairwise_components, feature_name)


def _finite_component(pairwise_components: Mapping[str, Any], component_name: str) -> np.ndarray:
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Pairwise component {component_name!r} must be two-dimensional")
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
