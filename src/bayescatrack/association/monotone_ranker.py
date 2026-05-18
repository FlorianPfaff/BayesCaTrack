"""Monotone ranking association model for Track2p ROI linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferencePairwiseExamples,
    pairwise_components_from_bundle,
    pairwise_feature_tensor,
)

DEFAULT_MONOTONE_BADNESS_FEATURES = tuple(
    name
    for name in DEFAULT_ASSOCIATION_FEATURES
    if name != "activity_similarity_available"
)


@dataclass(frozen=True)
class MonotoneRankerOptions:
    """Projected-gradient options for the monotone hard-negative ranker."""

    monotone_feature_names: tuple[str, ...] = ()
    margin: float = 1.0
    max_negatives_per_positive: int = 16
    include_row_negatives: bool = True
    include_column_negatives: bool = True
    max_iter: int = 800
    learning_rate: float = 0.05
    l2_regularization: float = 1.0e-3
    binary_loss_weight: float = 0.05
    tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "monotone_feature_names", tuple(self.monotone_feature_names or ())
        )
        if self.margin <= 0.0:
            raise ValueError("margin must be positive")
        if self.max_negatives_per_positive <= 0:
            raise ValueError("max_negatives_per_positive must be positive")
        if not self.include_row_negatives and not self.include_column_negatives:
            raise ValueError("At least one negative source must be enabled")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative")
        if self.binary_loss_weight < 0.0:
            raise ValueError("binary_loss_weight must be non-negative")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")


@dataclass(frozen=True)
class MonotoneRankingAssociationModel:
    """Association model whose cost is monotone in badness features."""

    feature_names: tuple[str, ...]
    monotone_feature_names: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    decision_offset: float
    training_rank_loss: float
    training_binary_loss: float
    n_rank_constraints: int
    n_training_examples: int
    n_positive_examples: int

    @property
    def model(self) -> "MonotoneRankingAssociationModel":
        """Compatibility with wrappers that access ``calibrated_model.model``."""

        return self

    def raw_cost_score(self, features: Any) -> np.ndarray:
        """Return the normalized linear badness score."""

        normalized = self._normalized_features(features)
        return np.tensordot(normalized, self.weights, axes=([-1], [0]))

    def pairwise_cost_matrix(self, features: Any) -> np.ndarray:
        """Return non-negative assignment costs."""

        return _softplus(self.raw_cost_score(features) - self.decision_offset)

    def predict_match_probability(self, features: Any) -> np.ndarray:
        """Return sigmoid match probabilities."""

        return _sigmoid(self.decision_offset - self.raw_cost_score(features))

    def pairwise_cost_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        return self.pairwise_cost_matrix(
            pairwise_feature_tensor(pairwise_components, feature_names=self.feature_names)
        )

    def pairwise_probability_matrix_from_components(
        self, pairwise_components: Mapping[str, Any]
    ) -> np.ndarray:
        return self.predict_match_probability(
            pairwise_feature_tensor(pairwise_components, feature_names=self.feature_names)
        )

    def pairwise_cost_matrix_from_bundle(
        self, bundle: Any, *, session_gap: int | float = 1.0
    ) -> np.ndarray:
        return self.pairwise_cost_matrix_from_components(
            pairwise_components_from_bundle(bundle, session_gap=session_gap)
        )

    def pairwise_probability_matrix_from_bundle(
        self, bundle: Any, *, session_gap: int | float = 1.0
    ) -> np.ndarray:
        return self.pairwise_probability_matrix_from_components(
            pairwise_components_from_bundle(bundle, session_gap=session_gap)
        )

    def _normalized_features(self, features: Any) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("Feature tensor does not match model feature schema")
        indices = [self.feature_names.index(name) for name in self.monotone_feature_names]
        selected = np.nan_to_num(
            array[..., indices], nan=0.0, posinf=1.0e6, neginf=-1.0e6
        )
        return (selected - self.feature_center) / self.feature_scale


# pylint: disable=too-many-locals
def fit_monotone_ranking_association_model_from_blocks(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    options: MonotoneRankerOptions | None = None,
) -> MonotoneRankingAssociationModel:
    """Fit a non-negative hard-negative ranking model from pairwise GT blocks."""

    blocks = tuple(example_blocks)
    if not blocks:
        raise ValueError("At least one pairwise example block is required")
    options = options or MonotoneRankerOptions()
    feature_names = _shared_feature_names(blocks)
    monotone_names = _resolve_monotone_feature_names(
        feature_names, options.monotone_feature_names
    )
    feature_indices = np.asarray(
        [feature_names.index(name) for name in monotone_names], dtype=int
    )
    raw_examples, labels, raw_constraints = _collect_training_arrays(
        blocks, feature_indices=feature_indices, options=options
    )
    center, scale = _robust_normalizer(raw_examples)
    examples = (raw_examples - center) / scale
    constraints = raw_constraints / scale
    weights, offset, rank_loss, binary_loss = _fit_projected_ranker(
        examples, labels, constraints, options=options
    )
    return MonotoneRankingAssociationModel(
        feature_names=feature_names,
        monotone_feature_names=monotone_names,
        feature_center=center,
        feature_scale=scale,
        weights=weights,
        decision_offset=offset,
        training_rank_loss=rank_loss,
        training_binary_loss=binary_loss,
        n_rank_constraints=int(constraints.shape[0]),
        n_training_examples=int(labels.size),
        n_positive_examples=int(np.sum(labels != 0)),
    )


def _shared_feature_names(blocks: Sequence[ReferencePairwiseExamples]) -> tuple[str, ...]:
    feature_names = tuple(blocks[0].feature_names)
    if not feature_names:
        raise ValueError("At least one feature is required")
    for block in blocks[1:]:
        if tuple(block.feature_names) != feature_names:
            raise ValueError("All example blocks must use the same feature schema")
    return feature_names


def _resolve_monotone_feature_names(
    feature_names: Sequence[str], requested: Sequence[str]
) -> tuple[str, ...]:
    if requested:
        selected = tuple(requested)
    else:
        selected = tuple(
            name for name in DEFAULT_MONOTONE_BADNESS_FEATURES if name in feature_names
        )
        if not selected:
            selected = tuple(
                name
                for name in feature_names
                if name != "session_gap" and not name.endswith("_available")
            )
    missing = [name for name in selected if name not in feature_names]
    if missing:
        raise ValueError(
            "Monotone features are absent from the feature tensor: "
            + ", ".join(missing)
        )
    if not selected:
        raise ValueError("At least one monotone badness feature is required")
    return selected


# pylint: disable=too-many-locals
def _collect_training_arrays(
    blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_indices: np.ndarray,
    options: MonotoneRankerOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples: list[np.ndarray] = []
    labels_out: list[int] = []
    constraints: list[np.ndarray] = []
    seen_examples: set[tuple[int, int, int]] = set()
    for block_index, block in enumerate(blocks):
        features = np.nan_to_num(
            np.asarray(block.features, dtype=float), nan=0.0, posinf=1.0e6, neginf=-1.0e6
        )
        labels = np.asarray(block.labels, dtype=int)
        if features.ndim != 3 or labels.shape != features.shape[:2]:
            raise ValueError("Pairwise features and labels have incompatible shapes")
        selected = features[..., feature_indices]
        hardness = np.mean(selected, axis=-1)
        for positive_row, positive_col in np.argwhere(labels != 0):
            positive_features = selected[positive_row, positive_col]
            key = (block_index, int(positive_row), int(positive_col))
            if key not in seen_examples:
                seen_examples.add(key)
                examples.append(positive_features)
                labels_out.append(1)
            for negative_row, negative_col in _hard_negative_positions(
                labels, hardness, int(positive_row), int(positive_col), options
            ):
                negative_features = selected[negative_row, negative_col]
                key = (block_index, int(negative_row), int(negative_col))
                if key not in seen_examples:
                    seen_examples.add(key)
                    examples.append(negative_features)
                    labels_out.append(0)
                constraints.append(positive_features - negative_features)
    if not examples:
        raise ValueError("No positive reference examples were found")
    if not constraints:
        raise ValueError("No hard-negative ranking constraints were found")
    return (
        np.asarray(examples, dtype=float),
        np.asarray(labels_out, dtype=int),
        np.asarray(constraints, dtype=float),
    )


def _hard_negative_positions(
    labels: np.ndarray,
    hardness: np.ndarray,
    positive_row: int,
    positive_col: int,
    options: MonotoneRankerOptions,
) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_candidates(candidates: Sequence[tuple[int, int]]) -> None:
        for row, col in candidates:
            key = (int(row), int(col))
            if key in seen:
                continue
            seen.add(key)
            positions.append(key)
            if len(positions) >= options.max_negatives_per_positive:
                return

    if options.include_row_negatives:
        columns = np.flatnonzero(labels[positive_row] == 0)
        order = np.lexsort((columns, hardness[positive_row, columns]))
        add_candidates([(positive_row, int(col)) for col in columns[order]])
    if len(positions) < options.max_negatives_per_positive and options.include_column_negatives:
        rows = np.flatnonzero(labels[:, positive_col] == 0)
        order = np.lexsort((rows, hardness[rows, positive_col]))
        add_candidates([(int(row), positive_col) for row in rows[order]])
    return tuple(positions)


def _robust_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    center = np.median(values, axis=0)
    q75, q25 = np.percentile(values, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return center.astype(float), scale.astype(float)


# pylint: disable=too-many-locals
def _fit_projected_ranker(
    examples: np.ndarray,
    labels: np.ndarray,
    constraints: np.ndarray,
    *,
    options: MonotoneRankerOptions,
) -> tuple[np.ndarray, float, float, float]:
    weights = np.full((examples.shape[1],), 1.0 / examples.shape[1], dtype=float)
    offset = _initial_offset(examples @ weights, labels)
    sample_weights = _balanced_binary_weights(labels)
    previous_loss = np.inf
    rank_loss = np.inf
    binary_loss = np.inf
    for iteration in range(options.max_iter):
        deltas = constraints @ weights
        violations = np.maximum(0.0, options.margin + deltas)
        rank_loss = float(np.mean(violations * violations))
        rank_gradient = 2.0 * np.mean(violations[:, None] * constraints, axis=0)
        scores = examples @ weights
        probabilities = _sigmoid(offset - scores)
        binary_loss = _weighted_log_loss(probabilities, labels, sample_weights)
        residual = sample_weights * (labels.astype(float) - probabilities)
        denom = max(float(np.sum(sample_weights)), 1.0)
        binary_gradient = (examples.T @ residual) / denom
        offset_gradient = float(
            np.sum(sample_weights * (probabilities - labels.astype(float))) / denom
        )
        gradient = (
            rank_gradient
            + options.binary_loss_weight * binary_gradient
            + options.l2_regularization * weights
        )
        step = options.learning_rate / np.sqrt(1.0 + iteration / 50.0)
        weights = np.maximum(0.0, weights - step * gradient)
        offset -= step * options.binary_loss_weight * offset_gradient
        loss = (
            rank_loss
            + options.binary_loss_weight * binary_loss
            + 0.5 * options.l2_regularization * float(weights @ weights)
        )
        if np.isfinite(previous_loss) and abs(previous_loss - loss) <= options.tolerance:
            break
        previous_loss = loss
    scores = examples @ weights
    offset = _fit_offset(scores, labels, sample_weights, initial_offset=offset)
    probabilities = _sigmoid(offset - scores)
    binary_loss = _weighted_log_loss(probabilities, labels, sample_weights)
    rank_loss = float(np.mean(np.maximum(0.0, options.margin + constraints @ weights) ** 2))
    return weights.astype(float), float(offset), rank_loss, binary_loss


def _initial_offset(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels != 0]
    negative = scores[labels == 0]
    if positive.size and negative.size:
        return float(0.5 * (np.median(positive) + np.median(negative)))
    return float(np.median(scores)) if scores.size else 0.0


def _fit_offset(
    scores: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray, *, initial_offset: float
) -> float:
    offset = float(initial_offset)
    targets = labels.astype(float)
    for _ in range(50):
        probabilities = _sigmoid(offset - scores)
        gradient = float(np.sum(sample_weights * (probabilities - targets)))
        hessian = float(np.sum(sample_weights * probabilities * (1.0 - probabilities)))
        if hessian <= 1.0e-12:
            break
        step = gradient / hessian
        offset -= step
        if abs(step) <= 1.0e-10 * max(1.0, abs(offset)):
            break
    return float(offset)


def _balanced_binary_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    weights = np.ones(labels.shape, dtype=float)
    for is_positive in (False, True):
        mask = labels != 0 if is_positive else labels == 0
        count = int(np.sum(mask))
        if count:
            weights[mask] = labels.size / (2.0 * count)
    return weights


def _weighted_log_loss(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    y = np.asarray(labels, dtype=float)
    return float(np.sum(weights * (-y * np.log(p) - (1.0 - y) * np.log(1.0 - p))) / max(float(np.sum(weights)), 1.0))


def _sigmoid(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _softplus(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)
