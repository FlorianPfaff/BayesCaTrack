"""Monotone pairwise ranking costs for calibrated ROI association.

The model is trained from manual-reference pairwise examples. Every labelled
positive edge is constrained to have a lower assignment cost than hard row and
column alternatives from the same registered session-pair block. Feature signs
turn benefit-like components into costs, and projected gradient updates keep all
learned feature weights non-negative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    ReferencePairwiseExamples,
)

_EPSILON = 1.0e-12
_DEFAULT_HARDNESS_FEATURES = (
    "mahalanobis_centroid_distance",
    "centroid_distance",
    "one_minus_iou",
    "one_minus_mask_cosine",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity_cost",
)


@dataclass(frozen=True)
class MonotoneRankerOptions:
    """Training knobs for the monotone pairwise ranking model."""

    row_negatives_per_positive: int = 20
    column_negatives_per_positive: int = 20
    max_preference_pairs: int | None = 200_000
    learning_rate: float = 0.05
    max_iter: int = 750
    l2_regularization: float = 1.0e-3
    random_seed: int = 0
    hardness_feature_names: tuple[str, ...] = ()
    feature_directions: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        for name in ("row_negatives_per_positive", "column_negatives_per_positive"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.max_preference_pairs is not None:
            max_pairs = int(self.max_preference_pairs)
            if max_pairs <= 0:
                raise ValueError("max_preference_pairs must be positive or None")
            object.__setattr__(self, "max_preference_pairs", max_pairs)
        if self.learning_rate <= 0.0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.l2_regularization < 0.0 or not np.isfinite(self.l2_regularization):
            raise ValueError("l2_regularization must be finite and non-negative")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(
            self,
            "hardness_feature_names",
            tuple(
                ()
                if self.hardness_feature_names is None
                else self.hardness_feature_names
            ),
        )
        if self.feature_directions is not None:
            directions = {
                str(key): float(value) for key, value in self.feature_directions.items()
            }
            if any(
                (not np.isfinite(value)) or value == 0.0
                for value in directions.values()
            ):
                raise ValueError(
                    "feature_directions values must be finite and non-zero"
                )
            object.__setattr__(self, "feature_directions", directions)


@dataclass(frozen=True)
class MonotonePairwiseRanker:
    """Calibrated monotone linear ranking model; lower scores are better."""

    feature_names: tuple[str, ...]
    feature_directions: tuple[float, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    probability_intercept: float
    probability_score_scale: float
    training_examples: int
    positive_examples: int
    preference_pairs: int

    @property
    def negative_examples(self) -> int:
        return int(self.training_examples - self.positive_examples)

    @property
    def nonzero_weights(self) -> int:
        return int(np.sum(np.asarray(self.weights, dtype=float) > 0.0))

    def predict_score(self, features: Any) -> np.ndarray:
        """Return the uncalibrated ranking score; lower is more match-like."""

        normalized = self._normalized_features(features)
        return np.tensordot(
            normalized, np.asarray(self.weights, dtype=float), axes=([-1], [0])
        )

    def predict_match_probability(self, features: Any) -> np.ndarray:
        """Return monotone match probabilities derived from the learned score."""

        score = np.asarray(self.predict_score(features), dtype=float)
        logits = float(self.probability_intercept) - score / float(
            self.probability_score_scale
        )
        return _sigmoid(logits)

    def pairwise_cost_matrix(self, features: Any) -> np.ndarray:
        """Return assignment costs compatible with calibrated global assignment."""

        probabilities = np.clip(self.predict_match_probability(features), _EPSILON, 1.0)
        return -np.log(probabilities)

    def coefficient_rows(self) -> list[dict[str, float | str]]:
        """Return interpretable feature-direction and weight diagnostics."""

        return [
            {
                "feature": name,
                "direction": "cost" if direction > 0.0 else "benefit",
                "weight": float(weight),
                "center": float(center),
                "scale": float(scale),
            }
            for name, direction, weight, center, scale in zip(
                self.feature_names,
                self.feature_directions,
                np.asarray(self.weights, dtype=float),
                np.asarray(self.feature_center, dtype=float),
                np.asarray(self.feature_scale, dtype=float),
            )
        ]

    def _normalized_features(self, features: Any) -> np.ndarray:
        feature_array = np.asarray(features, dtype=float)
        if feature_array.shape[-1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {feature_array.shape[-1]}"
            )
        finite = np.nan_to_num(feature_array, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        signed = finite * np.asarray(self.feature_directions, dtype=float)
        return (signed - np.asarray(self.feature_center, dtype=float)) / np.asarray(
            self.feature_scale, dtype=float
        )


def fit_monotone_ranked_association_model(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_names: Sequence[str] | None = None,
    options: MonotoneRankerOptions | None = None,
) -> CalibratedAssociationModel:
    """Fit a calibrated association model from manual-GT ranking constraints."""

    options = options or MonotoneRankerOptions()
    feature_names_tuple = _validated_feature_names(example_blocks, feature_names)
    directions = np.asarray(
        [
            _feature_direction(name, options.feature_directions)
            for name in feature_names_tuple
        ],
        dtype=float,
    )
    binary_features, binary_labels, differences = _collect_training_arrays(
        example_blocks,
        feature_names=feature_names_tuple,
        directions=directions,
        options=options,
    )
    signed_binary = _finite_features(binary_features) * directions
    center, scale = _robust_center_scale(
        signed_binary.reshape(-1, signed_binary.shape[-1])
    )
    normalized_differences = differences / scale
    weights = _fit_projected_logistic_ranker(normalized_differences, options)
    ranker = MonotonePairwiseRanker(
        feature_names=feature_names_tuple,
        feature_directions=tuple(float(value) for value in directions),
        feature_center=center,
        feature_scale=scale,
        weights=weights,
        probability_intercept=_logit_prior(binary_labels),
        probability_score_scale=_score_scale(signed_binary, center, scale, weights),
        training_examples=int(binary_labels.size),
        positive_examples=int(np.sum(binary_labels != 0)),
        preference_pairs=int(normalized_differences.shape[0]),
    )
    return CalibratedAssociationModel(model=ranker, feature_names=feature_names_tuple)


def _validated_feature_names(
    blocks: Sequence[ReferencePairwiseExamples], feature_names: Sequence[str] | None
) -> tuple[str, ...]:
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("At least one pairwise example block is required")
    names = tuple(blocks[0].feature_names if feature_names is None else feature_names)
    if not names:
        raise ValueError("At least one feature is required")
    for block in blocks:
        missing = [name for name in names if name not in block.feature_names]
        if missing:
            raise ValueError("Missing monotone-ranker features: " + ", ".join(missing))
    return names


def _collect_training_arrays(
    blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_names: tuple[str, ...],
    directions: np.ndarray,
    options: MonotoneRankerOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary_features: list[np.ndarray] = []
    binary_labels: list[int] = []
    differences: list[np.ndarray] = []
    for block in blocks:
        features = _selected_block_features(block, feature_names)
        labels = np.asarray(block.labels, dtype=int)
        if labels.shape != features.shape[:2]:
            raise ValueError("Pairwise labels must match feature tensor shape")
        hardness = _hardness_score(features, feature_names, directions, options)
        for row, col in np.argwhere(labels != 0):
            positive = features[row, col]
            binary_features.append(positive)
            binary_labels.append(1)
            negatives = _hard_negative_indices(
                labels, hardness, int(row), int(col), options=options
            )
            for neg_row, neg_col in negatives:
                negative = features[neg_row, neg_col]
                binary_features.append(negative)
                binary_labels.append(0)
                differences.append((positive - negative) * directions)
    if not differences:
        raise ValueError("No monotone ranking preference pairs were generated")
    feature_array = np.asarray(binary_features, dtype=float)
    label_array = np.asarray(binary_labels, dtype=int)
    difference_array = np.asarray(differences, dtype=float)
    if (
        options.max_preference_pairs is not None
        and difference_array.shape[0] > options.max_preference_pairs
    ):
        rng = np.random.default_rng(options.random_seed)
        keep = np.sort(
            rng.choice(
                difference_array.shape[0],
                size=options.max_preference_pairs,
                replace=False,
            )
        )
        difference_array = difference_array[keep]
    return feature_array, label_array, difference_array


def _selected_block_features(
    block: ReferencePairwiseExamples, feature_names: tuple[str, ...]
) -> np.ndarray:
    feature_indices = [tuple(block.feature_names).index(name) for name in feature_names]
    return _finite_features(
        np.asarray(block.features, dtype=float)[..., feature_indices]
    )


def _hard_negative_indices(
    labels: np.ndarray,
    hardness: np.ndarray,
    row: int,
    col: int,
    *,
    options: MonotoneRankerOptions,
) -> list[tuple[int, int]]:
    negatives: list[tuple[int, int]] = []
    if options.row_negatives_per_positive:
        row_candidates = np.flatnonzero(labels[row] == 0)
        ordered_cols = row_candidates[
            np.argsort(hardness[row, row_candidates], kind="mergesort")
        ]
        negatives.extend(
            (row, int(candidate_col))
            for candidate_col in ordered_cols[: options.row_negatives_per_positive]
        )
    if options.column_negatives_per_positive:
        col_candidates = np.flatnonzero(labels[:, col] == 0)
        ordered_rows = col_candidates[
            np.argsort(hardness[col_candidates, col], kind="mergesort")
        ]
        negatives.extend(
            (int(candidate_row), col)
            for candidate_row in ordered_rows[: options.column_negatives_per_positive]
        )
    return list(dict.fromkeys(negatives))


def _hardness_score(
    selected_features: np.ndarray,
    feature_names: tuple[str, ...],
    directions: np.ndarray,
    options: MonotoneRankerOptions,
) -> np.ndarray:
    if options.hardness_feature_names:
        names = tuple(options.hardness_feature_names)
    else:
        names = tuple(
            name for name in _DEFAULT_HARDNESS_FEATURES if name in feature_names
        )
    if not names:
        signed = selected_features * directions
        return np.mean(_robust_normalize_planes(signed), axis=-1)
    indices = [feature_names.index(name) for name in names if name in feature_names]
    if not indices:
        return np.zeros(selected_features.shape[:2], dtype=float)
    return np.mean(
        _robust_normalize_planes(selected_features[..., indices] * directions[indices]),
        axis=-1,
    )


def _fit_projected_logistic_ranker(
    differences: np.ndarray, options: MonotoneRankerOptions
) -> np.ndarray:
    differences = np.asarray(differences, dtype=float)
    weights = np.zeros((differences.shape[1],), dtype=float)
    learning_rate = float(options.learning_rate)
    for iteration in range(int(options.max_iter)):
        margins = np.clip(differences @ weights, -60.0, 60.0)
        gradient = (differences.T @ _sigmoid(margins)) / differences.shape[0]
        gradient += float(options.l2_regularization) * weights
        step = learning_rate / np.sqrt(1.0 + iteration / 50.0)
        weights = np.maximum(0.0, weights - step * gradient)
    return weights


def _feature_direction(
    feature_name: str, overrides: Mapping[str, float] | None
) -> float:
    if overrides is not None and feature_name in overrides:
        return 1.0 if float(overrides[feature_name]) > 0.0 else -1.0
    lower = feature_name.lower()
    benefit_tokens = ("iou", "similarity", "correlation", "probability")
    cost_tokens = ("one_minus", "cost", "distance", "gap", "ratio", "logdet")
    if lower.startswith("one_minus") or any(token in lower for token in cost_tokens):
        return 1.0
    if any(token in lower for token in benefit_tokens):
        return -1.0
    return 1.0


def _finite_features(features: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(features, dtype=float), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_features(values)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return center.astype(float), scale.astype(float)


def _robust_normalize_planes(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    center, scale = _robust_center_scale(flat)
    return (values - center) / scale


def _score_scale(
    signed_binary: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> float:
    scores = ((signed_binary - center) / scale) @ weights
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return 1.0
    spread = float(np.std(finite_scores))
    if not np.isfinite(spread) or spread <= 1.0e-12:
        return 1.0
    return spread


def _logit_prior(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    positives = float(np.sum(labels != 0))
    negatives = float(labels.size - positives)
    probability = (positives + 1.0) / (positives + negatives + 2.0)
    return float(np.log(probability / (1.0 - probability)))


def _sigmoid(values: Any) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    return np.where(
        values_array >= 0.0,
        1.0 / (1.0 + np.exp(-values_array)),
        np.exp(values_array) / (1.0 + np.exp(values_array)),
    )
