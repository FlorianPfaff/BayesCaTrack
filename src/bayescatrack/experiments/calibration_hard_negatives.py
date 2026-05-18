"""Candidate-limited hard-negative sampling for Track2p calibration folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples

_HARD_NEGATIVE_DEFAULT_FEATURES = (
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
class CandidateHardNegativeOptions:
    """Candidate-limited hard-negative selection for calibrated training."""

    negative_to_positive_ratio: float = 4.0
    candidate_top_k_per_anchor: int | None = 20
    include_column_candidates: bool = True
    hardness_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ratio = float(self.negative_to_positive_ratio)
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError(
                "negative_to_positive_ratio must be finite and non-negative"
            )
        object.__setattr__(self, "negative_to_positive_ratio", ratio)
        if self.candidate_top_k_per_anchor is not None:
            top_k = int(self.candidate_top_k_per_anchor)
            if top_k <= 0:
                raise ValueError("candidate_top_k_per_anchor must be positive or None")
            object.__setattr__(self, "candidate_top_k_per_anchor", top_k)
        object.__setattr__(
            self,
            "hardness_feature_names",
            tuple(
                ()
                if self.hardness_feature_names is None
                else self.hardness_feature_names
            ),
        )


def collect_candidate_limited_training_examples(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    options: CandidateHardNegativeOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all positives and a bounded set of hard negatives from pairwise blocks."""

    options = options or CandidateHardNegativeOptions()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for block in example_blocks:
        features, labels = _validated_pairwise_block_arrays(block)
        selected_mask = _candidate_hard_negative_selection_mask(block, options)
        selected_labels = labels[selected_mask].reshape(-1)
        if selected_labels.size == 0:
            continue
        feature_blocks.append(features[selected_mask].reshape(-1, features.shape[-1]))
        label_blocks.append(selected_labels)
    if not feature_blocks:
        raise ValueError("No calibration training examples were selected")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def balanced_binary_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for binary calibration labels."""

    labels = np.asarray(labels, dtype=int).reshape(-1)
    weights = np.ones(labels.shape, dtype=float)
    n_examples = int(labels.size)
    if n_examples == 0:
        return weights
    positive = labels != 0
    negative = ~positive
    for mask in (positive, negative):
        n_class = int(np.sum(mask))
        if n_class:
            weights[mask] = n_examples / (2.0 * n_class)
    return weights


def _validated_pairwise_block_arrays(
    block: ReferencePairwiseExamples,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(block.features, dtype=float)
    labels = np.asarray(block.labels, dtype=int)
    if features.ndim != 3:
        raise ValueError("Pairwise training features must be three-dimensional")
    if labels.shape != features.shape[:2]:
        raise ValueError("Pairwise labels must match the first two feature dimensions")
    if features.shape[-1] != len(block.feature_names):
        raise ValueError("Pairwise feature-name count does not match feature tensor")
    return features, labels


def _candidate_hard_negative_selection_mask(
    block: ReferencePairwiseExamples,
    options: CandidateHardNegativeOptions,
) -> np.ndarray:
    _features, labels = _validated_pairwise_block_arrays(block)
    positive_mask = labels != 0
    candidate_mask = _top_k_candidate_negative_mask(block, options)
    selected_negative_mask = np.zeros(labels.shape, dtype=bool)
    n_positive = int(np.sum(positive_mask))
    if n_positive > 0 and options.negative_to_positive_ratio > 0.0:
        max_negatives = int(np.ceil(options.negative_to_positive_ratio * n_positive))
        flat_candidates = np.flatnonzero(candidate_mask.reshape(-1))
        if flat_candidates.size:
            hardness = _pairwise_hardness_score(
                block, hardness_feature_names=options.hardness_feature_names
            )
            order = np.lexsort((flat_candidates, hardness.reshape(-1)[flat_candidates]))
            selected_flat = flat_candidates[order[:max_negatives]]
            selected_negative_mask.reshape(-1)[selected_flat] = True
    return positive_mask | selected_negative_mask


def _top_k_candidate_negative_mask(
    block: ReferencePairwiseExamples,
    options: CandidateHardNegativeOptions,
) -> np.ndarray:
    _features, labels = _validated_pairwise_block_arrays(block)
    candidate_mask = labels == 0
    if options.candidate_top_k_per_anchor is None or not np.any(candidate_mask):
        return candidate_mask

    hardness = _pairwise_hardness_score(
        block, hardness_feature_names=options.hardness_feature_names
    )
    top_mask = np.zeros_like(candidate_mask)
    top_k = int(options.candidate_top_k_per_anchor)
    for row_index in range(candidate_mask.shape[0]):
        candidate_cols = np.flatnonzero(candidate_mask[row_index])
        if candidate_cols.size:
            ordered_cols = _ordered_candidate_indices(
                hardness[row_index, candidate_cols], candidate_cols
            )
            top_mask[row_index, ordered_cols[:top_k]] = True
    if options.include_column_candidates:
        for col_index in range(candidate_mask.shape[1]):
            candidate_rows = np.flatnonzero(candidate_mask[:, col_index])
            if candidate_rows.size:
                ordered_rows = _ordered_candidate_indices(
                    hardness[candidate_rows, col_index], candidate_rows
                )
                top_mask[ordered_rows[:top_k], col_index] = True
    return top_mask


def _ordered_candidate_indices(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return indices[np.lexsort((indices, np.asarray(scores, dtype=float)))]


def _pairwise_hardness_score(
    block: ReferencePairwiseExamples,
    *,
    hardness_feature_names: Sequence[str] = (),
) -> np.ndarray:
    feature_names = tuple(block.feature_names)
    if hardness_feature_names:
        selected_names = tuple(hardness_feature_names)
        missing = [name for name in selected_names if name not in feature_names]
        if missing:
            raise ValueError(
                "Hard-negative feature names are missing from the feature tensor: "
                + ", ".join(missing)
            )
    else:
        selected_names = tuple(
            name for name in _HARD_NEGATIVE_DEFAULT_FEATURES if name in feature_names
        )
        if not selected_names:
            selected_names = tuple(
                name
                for name in feature_names
                if name != "session_gap" and not name.endswith("_available")
            )
    if not selected_names:
        return np.zeros(np.asarray(block.labels).shape, dtype=float)
    planes = [
        _robust_normalized_cost_plane(_required_feature_plane(block, feature_name))
        for feature_name in selected_names
    ]
    return np.mean(np.stack(planes, axis=0), axis=0)


def _required_feature_plane(
    block: ReferencePairwiseExamples, feature_name: str
) -> np.ndarray:
    feature_names = tuple(block.feature_names)
    if feature_name not in feature_names:
        raise ValueError(f"Feature {feature_name!r} is required for hard negatives")
    features, _labels = _validated_pairwise_block_arrays(block)
    return features[..., feature_names.index(feature_name)]


def _robust_normalized_cost_plane(values: np.ndarray) -> np.ndarray:
    plane = np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
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
