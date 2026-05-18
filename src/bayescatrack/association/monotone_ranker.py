"""Monotone pairwise ranking costs for ROI association.

The ranker learns non-negative weights on cost-like pairwise features from
manual-GT positive edges versus row/column hard negatives. Lower scores are
better, so the learned model can be used as a calibrated-cost backend by the
existing global-assignment solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    ReferencePairwiseExamples,
)

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

_DEFAULT_COST_LIKE_FEATURES = frozenset(
    {
        "centroid_distance",
        "mahalanobis_centroid_distance",
        "one_minus_iou",
        "one_minus_mask_cosine",
        "one_minus_covariance_shape_similarity",
        "one_minus_activity_similarity",
        "area_ratio_cost",
        "covariance_shape_cost",
        "covariance_logdet_cost",
        "roi_feature_cost",
        "cell_probability_cost",
        "activity_similarity_cost",
        "session_gap",
    }
)


@dataclass(frozen=True)
class MonotoneRankerTrainingOptions:
    """Hyperparameters for the manual-GT pairwise ranking objective."""

    row_negatives_per_positive: int = 8
    column_negatives_per_positive: int = 8
    max_preference_pairs: int | None = 100_000
    learning_rate: float = 0.1
    max_iter: int = 500
    tolerance: float = 1.0e-7
    l2_regularization: float = 1.0e-3
    random_state: int = 0
    hardness_feature_names: tuple[str, ...] = ()
    cost_like_feature_names: tuple[str, ...] = ()
    positive_cost_target: float = 1.0

    def __post_init__(self) -> None:
        row_negatives = int(self.row_negatives_per_positive)
        column_negatives = int(self.column_negatives_per_positive)
        if row_negatives < 0 or column_negatives < 0:
            raise ValueError("row/column negatives per positive must be non-negative")
        if row_negatives + column_negatives <= 0:
            raise ValueError("At least one row or column hard negative is required")
        object.__setattr__(self, "row_negatives_per_positive", row_negatives)
        object.__setattr__(self, "column_negatives_per_positive", column_negatives)
        if self.max_preference_pairs is not None:
            max_pairs = int(self.max_preference_pairs)
            if max_pairs <= 0:
                raise ValueError("max_preference_pairs must be positive or None")
            object.__setattr__(self, "max_preference_pairs", max_pairs)
        for name in ("learning_rate", "tolerance", "l2_regularization", "positive_cost_target"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (name in {"learning_rate", "positive_cost_target"} and value <= 0.0):
                raise ValueError(f"{name} has an invalid value")
            object.__setattr__(self, name, value)
        max_iter = int(self.max_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "hardness_feature_names", tuple(self.hardness_feature_names or ()))
        object.__setattr__(self, "cost_like_feature_names", tuple(self.cost_like_feature_names or ()))


@dataclass(frozen=True)
class MonotonePairwiseRanker:
    """Non-negative additive lower-is-better pairwise cost model."""

    feature_names: tuple[str, ...]
    weights: np.ndarray
    impute_values: np.ndarray
    feature_scales: np.ndarray
    cost_scale: float
    decision_cost: float
    probability_temperature: float
    training_loss: float
    n_preference_pairs: int
    n_iterations: int
    trainable_feature_names: tuple[str, ...]

    @property
    def weights_(self) -> np.ndarray:
        """Sklearn-style alias for learned non-negative feature weights."""

        return np.asarray(self.weights, dtype=float)

    def pairwise_cost_matrix(self, features: Any) -> np.ndarray:
        """Return lower-is-better assignment costs for pairwise feature tensors."""

        prepared = self._prepare_features(features)
        scores = np.sum(prepared * self.weights, axis=-1)
        costs = scores / max(float(self.cost_scale), 1.0e-12)
        return np.maximum(np.nan_to_num(costs, nan=1.0e6, posinf=1.0e6, neginf=0.0), 0.0)

    def predict_match_probability(self, features: Any) -> np.ndarray:
        """Return a monotone ranking-confidence diagnostic, not a calibrated posterior."""

        costs = self.pairwise_cost_matrix(features)
        logits = (float(self.decision_cost) - costs) / max(float(self.probability_temperature), 1.0e-12)
        return _sigmoid(logits)

    def diagnostics(self, *, prefix: str = "monotone") -> dict[str, float | int]:
        """Return compact training diagnostics for benchmark output rows."""

        rows: dict[str, float | int] = {
            f"{prefix}_preference_pairs": int(self.n_preference_pairs),
            f"{prefix}_iterations": int(self.n_iterations),
            f"{prefix}_training_loss": float(self.training_loss),
            f"{prefix}_cost_scale": float(self.cost_scale),
            f"{prefix}_decision_cost": float(self.decision_cost),
        }
        for feature_name, weight in zip(self.feature_names, self.weights):
            rows[f"{prefix}_weight_{_metric_safe_name(feature_name)}"] = float(weight)
        return rows

    def _prepare_features(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim < 1:
            raise ValueError("features must have a final feature dimension")
        if values.shape[-1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {values.shape[-1]}")
        clean = np.where(np.isfinite(values), values, self.impute_values)
        return clean / self.feature_scales


def fit_monotone_ranked_association_model_from_blocks(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_names: Sequence[str] | None = None,
    options: MonotoneRankerTrainingOptions | None = None,
) -> CalibratedAssociationModel:
    """Fit a calibrated-association wrapper backed by a monotone ranker."""

    blocks = tuple(example_blocks)
    if not blocks:
        raise ValueError("At least one pairwise example block is required")
    names = tuple(feature_names or blocks[0].feature_names)
    for block in blocks:
        if tuple(block.feature_names) != names:
            raise ValueError("All pairwise example blocks must use the same feature_names")
    positive_features, negative_features = collect_monotone_preference_training_pairs(blocks, options=options)
    model = fit_monotone_pairwise_ranker(positive_features, negative_features, feature_names=names, options=options)
    return CalibratedAssociationModel(model=model, feature_names=names)


def collect_monotone_preference_training_pairs(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    options: MonotoneRankerTrainingOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect positive-vs-hard-negative feature pairs from labeled blocks."""

    options = options or MonotoneRankerTrainingOptions()
    positive_blocks: list[np.ndarray] = []
    negative_blocks: list[np.ndarray] = []
    for block in example_blocks:
        features, labels = _validated_pairwise_block_arrays(block)
        hardness = _pairwise_hardness_score(block, options.hardness_feature_names)
        positive_rows, positive_cols = np.nonzero(labels != 0)
        for row, col in zip(positive_rows, positive_cols):
            positive = features[row, col, :]
            for neg_col in _ordered_limited_indices(hardness[row, :], np.flatnonzero(labels[row, :] == 0), options.row_negatives_per_positive):
                positive_blocks.append(positive)
                negative_blocks.append(features[row, neg_col, :])
            for neg_row in _ordered_limited_indices(hardness[:, col], np.flatnonzero(labels[:, col] == 0), options.column_negatives_per_positive):
                positive_blocks.append(positive)
                negative_blocks.append(features[neg_row, col, :])
    if not positive_blocks:
        raise ValueError("No monotone ranking preference pairs were collected")
    positives = np.asarray(positive_blocks, dtype=float)
    negatives = np.asarray(negative_blocks, dtype=float)
    if options.max_preference_pairs is not None and positives.shape[0] > options.max_preference_pairs:
        rng = np.random.default_rng(int(options.random_state))
        keep = np.sort(rng.choice(positives.shape[0], size=int(options.max_preference_pairs), replace=False))
        positives = positives[keep]
        negatives = negatives[keep]
    return positives, negatives


def fit_monotone_pairwise_ranker(
    positive_features: Any,
    negative_features: Any,
    *,
    feature_names: Sequence[str],
    options: MonotoneRankerTrainingOptions | None = None,
) -> MonotonePairwiseRanker:
    """Fit non-negative weights from positive-lower-than-negative preferences."""

    options = options or MonotoneRankerTrainingOptions()
    names = tuple(feature_names)
    positives = np.asarray(positive_features, dtype=float)
    negatives = np.asarray(negative_features, dtype=float)
    if positives.shape != negatives.shape or positives.ndim != 2:
        raise ValueError("positive_features and negative_features must share shape (n_pairs, n_features)")
    if positives.shape[1] != len(names):
        raise ValueError("feature_names length does not match feature matrix width")
    finite_stack = np.concatenate([positives, negatives], axis=0)
    impute_values = _finite_column_median(finite_stack)
    scales = _robust_feature_scales(finite_stack, impute_values)
    pos = np.where(np.isfinite(positives), positives, impute_values) / scales
    neg = np.where(np.isfinite(negatives), negatives, impute_values) / scales
    diffs = pos - neg
    trainable = _trainable_feature_mask(names, options.cost_like_feature_names)
    if not np.any(trainable):
        raise ValueError("No cost-like trainable feature was selected")
    weights = np.zeros(len(names), dtype=float)
    weights[trainable] = 1.0 / np.sqrt(float(np.sum(trainable)))
    last_loss = np.inf
    iterations = 0
    for iteration in range(int(options.max_iter)):
        margins = diffs @ weights
        sigma = _sigmoid(margins)
        grad = np.mean(diffs * sigma[:, None], axis=0) + float(options.l2_regularization) * weights
        grad[~trainable] = 0.0
        weights[trainable] = np.maximum(weights[trainable] - float(options.learning_rate) * grad[trainable], 0.0)
        loss = float(np.mean(np.logaddexp(0.0, diffs @ weights)) + 0.5 * float(options.l2_regularization) * np.dot(weights, weights))
        iterations = iteration + 1
        if abs(last_loss - loss) <= float(options.tolerance):
            last_loss = loss
            break
        last_loss = loss
    raw_pos = pos @ weights
    raw_neg = neg @ weights
    positive_median = _finite_median(raw_pos, default=1.0)
    negative_median = _finite_median(raw_neg, default=positive_median + 1.0)
    cost_scale = positive_median / float(options.positive_cost_target) if positive_median > 1.0e-12 else max(negative_median / 4.0, 1.0)
    scaled_pos = raw_pos / max(cost_scale, 1.0e-12)
    scaled_neg = raw_neg / max(cost_scale, 1.0e-12)
    pos_med = _finite_median(scaled_pos, default=1.0)
    neg_med = _finite_median(scaled_neg, default=pos_med + 1.0)
    decision_cost = 0.5 * (pos_med + neg_med)
    probability_temperature = max(abs(neg_med - pos_med), 1.0)
    return MonotonePairwiseRanker(
        feature_names=names,
        weights=weights,
        impute_values=impute_values,
        feature_scales=scales,
        cost_scale=float(cost_scale),
        decision_cost=float(decision_cost),
        probability_temperature=float(probability_temperature),
        training_loss=float(last_loss),
        n_preference_pairs=int(positives.shape[0]),
        n_iterations=int(iterations),
        trainable_feature_names=tuple(name for name, is_trainable in zip(names, trainable) if is_trainable),
    )


def _validated_pairwise_block_arrays(block: ReferencePairwiseExamples) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(block.features, dtype=float)
    labels = np.asarray(block.labels, dtype=int)
    if features.ndim != 3:
        raise ValueError("Pairwise training features must be three-dimensional")
    if labels.shape != features.shape[:2]:
        raise ValueError("Pairwise labels must match the first two feature dimensions")
    if features.shape[-1] != len(block.feature_names):
        raise ValueError("feature_names length does not match feature tensor width")
    return features, labels


def _pairwise_hardness_score(block: ReferencePairwiseExamples, hardness_feature_names: Sequence[str]) -> np.ndarray:
    features, labels = _validated_pairwise_block_arrays(block)
    names = tuple(block.feature_names)
    selected = tuple(hardness_feature_names) or tuple(name for name in _DEFAULT_HARDNESS_FEATURES if name in names)
    if not selected:
        selected = tuple(name for name in names if name != "session_gap" and not name.endswith("_available"))
    if not selected:
        return np.zeros(labels.shape, dtype=float)
    planes = []
    for name in selected:
        if name not in names:
            raise ValueError(f"Hard-negative feature {name!r} is missing")
        planes.append(_robust_normalized_cost_plane(features[..., names.index(name)]))
    return np.mean(np.stack(planes, axis=0), axis=0)


def _ordered_limited_indices(scores: np.ndarray, candidate_indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or candidate_indices.size == 0:
        return np.empty((0,), dtype=int)
    ordered = candidate_indices[np.lexsort((candidate_indices, np.asarray(scores, dtype=float)[candidate_indices]))]
    return ordered[:limit]


def _trainable_feature_mask(feature_names: Sequence[str], explicit_cost_like_names: Sequence[str]) -> np.ndarray:
    names = tuple(feature_names)
    if explicit_cost_like_names:
        explicit = set(explicit_cost_like_names)
        missing = sorted(explicit - set(names))
        if missing:
            raise ValueError("cost_like_feature_names are missing: " + ", ".join(missing))
        return np.asarray([name in explicit for name in names], dtype=bool)
    return np.asarray([name in _DEFAULT_COST_LIKE_FEATURES for name in names], dtype=bool)


def _finite_column_median(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[1],), dtype=float)
    for col in range(values.shape[1]):
        finite = values[:, col][np.isfinite(values[:, col])]
        result[col] = float(np.median(finite)) if finite.size else 0.0
    return result


def _robust_feature_scales(values: np.ndarray, impute_values: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, impute_values)
    q25 = np.percentile(clean, 25.0, axis=0)
    q75 = np.percentile(clean, 75.0, axis=0)
    scales = q75 - q25
    std = np.std(clean, axis=0)
    scales = np.where(scales > 1.0e-12, scales, std)
    return np.where(scales > 1.0e-12, scales, 1.0)


def _robust_normalized_cost_plane(values: np.ndarray) -> np.ndarray:
    plane = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=float)
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return (plane - center) / scale


def _finite_median(values: np.ndarray, *, default: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0.0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def _metric_safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
