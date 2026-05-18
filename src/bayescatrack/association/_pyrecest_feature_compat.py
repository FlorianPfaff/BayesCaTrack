"""Compatibility helpers for PyRecEst named pairwise association features."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FeatureTransform = Callable[[Mapping[str, Any]], Any]

try:
    from pyrecest.utils import (
        pairwise_feature_tensor as _pyrecest_pairwise_feature_tensor,
    )
except ImportError:  # pragma: no cover - exercised when PyRecEst grows this API
    _pyrecest_pairwise_feature_tensor = None


@dataclass(frozen=True)
class NamedPairwiseFeatureSchema:
    """Minimal named pairwise feature schema compatible with PyRecEst's planned API."""

    feature_names: tuple[str, ...]
    transforms: Mapping[str, FeatureTransform] | None = None

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        transforms: Mapping[str, FeatureTransform] | None = None,
    ) -> None:
        object.__setattr__(self, "feature_names", tuple(feature_names))
        object.__setattr__(self, "transforms", dict(transforms or {}))

    def build_tensor(self, pairwise_components: Mapping[str, Any]) -> np.ndarray:
        """Build a feature tensor from named pairwise components."""

        return pairwise_feature_tensor(pairwise_components, self)


def pairwise_feature_tensor(
    pairwise_components: Mapping[str, Any],
    schema: NamedPairwiseFeatureSchema,
) -> np.ndarray:
    """Build a pairwise feature tensor using PyRecEst when available."""

    if _pyrecest_pairwise_feature_tensor is not None:
        try:
            return np.asarray(
                _pyrecest_pairwise_feature_tensor(pairwise_components, schema),
                dtype=float,
            )
        except (AttributeError, TypeError):
            pass
    return _local_pairwise_feature_tensor(pairwise_components, schema)


@dataclass(frozen=True)
class CalibratedPairwiseAssociationModel:
    """Compatibility wrapper for calibrated pairwise association models."""

    model: Any
    schema: NamedPairwiseFeatureSchema

    def pairwise_cost_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        features = self.schema.build_tensor(pairwise_components)
        if hasattr(self.model, "pairwise_cost_matrix"):
            return np.asarray(self.model.pairwise_cost_matrix(features), dtype=float)
        probabilities = self.predict_match_probability(features)
        probabilities = np.clip(probabilities, 1.0e-12, 1.0)
        return -np.log(probabilities)

    def pairwise_probability_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        return self.predict_match_probability(
            self.schema.build_tensor(pairwise_components)
        )

    def predict_match_probability(self, features: Any) -> np.ndarray:
        if hasattr(self.model, "predict_match_probability"):
            probabilities = self.model.predict_match_probability(features)
        elif hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
            if probabilities.ndim >= 1 and probabilities.shape[-1] == 2:
                probabilities = probabilities[..., 1]
        else:
            costs = np.asarray(self.model.pairwise_cost_matrix(features), dtype=float)
            probabilities = np.exp(-costs)
        return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def load_logistic_pairwise_association_model() -> type[Any]:
    """Return PyRecEst's logistic pairwise association model class."""

    try:
        from pyrecest.utils import LogisticPairwiseAssociationModel

        return LogisticPairwiseAssociationModel
    except ImportError:
        pass

    try:
        from pyrecest.utils.association_models import LogisticPairwiseAssociationModel

        return LogisticPairwiseAssociationModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyRecEst with LogisticPairwiseAssociationModel is required to fit calibrated association costs."
        ) from exc


def _local_pairwise_feature_tensor(
    pairwise_components: Mapping[str, Any],
    schema: NamedPairwiseFeatureSchema,
) -> np.ndarray:
    feature_planes = [
        _schema_feature(pairwise_components, schema, feature_name)
        for feature_name in schema.feature_names
    ]
    if not feature_planes:
        raise ValueError("At least one feature is required")
    reference_shape = feature_planes[0].shape
    for feature_name, feature_plane in zip(schema.feature_names, feature_planes):
        if feature_plane.shape != reference_shape:
            raise ValueError(
                f"Feature {feature_name!r} has shape {feature_plane.shape}, expected {reference_shape}"
            )
    return np.stack(feature_planes, axis=-1)


def _schema_feature(
    pairwise_components: Mapping[str, Any],
    schema: NamedPairwiseFeatureSchema,
    feature_name: str,
) -> np.ndarray:
    transforms = getattr(schema, "transforms", None) or {}
    transform = transforms.get(feature_name)
    if transform is not None:
        values = transform(pairwise_components)
    else:
        values = pairwise_components[feature_name]
    return _finite_feature(values, feature_name)


def _finite_feature(values: Any, feature_name: str) -> np.ndarray:
    feature = np.asarray(values, dtype=float)
    if feature.ndim != 2:
        raise ValueError(f"Feature {feature_name!r} must be two-dimensional")
    return np.nan_to_num(feature, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
