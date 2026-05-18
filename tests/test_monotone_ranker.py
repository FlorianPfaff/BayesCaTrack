from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerTrainingOptions,
    collect_monotone_preference_training_pairs,
    fit_monotone_ranked_association_model_from_blocks,
)


def _toy_rank_block() -> ReferencePairwiseExamples:
    features = np.zeros((2, 2, 2), dtype=float)
    features[..., 0] = np.array([[0.0, 4.0], [4.0, 0.0]])
    features[..., 1] = 1.0
    return ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=np.eye(2, dtype=int),
        reference_roi_indices=np.arange(2),
        measurement_roi_indices=np.arange(2),
        feature_names=("one_minus_iou", "session_gap"),
    )


def test_collect_monotone_preference_pairs_uses_row_and_column_hard_negatives():
    positives, negatives = collect_monotone_preference_training_pairs(
        (_toy_rank_block(),),
        options=MonotoneRankerTrainingOptions(
            row_negatives_per_positive=1,
            column_negatives_per_positive=1,
            max_preference_pairs=None,
        ),
    )

    assert positives.shape == negatives.shape == (4, 2)
    assert np.all(positives[:, 0] == 0.0)
    assert np.all(negatives[:, 0] == 4.0)


def test_monotone_ranker_ranks_gt_edges_above_row_and_column_hard_negatives():
    block = _toy_rank_block()

    calibrated_model = fit_monotone_ranked_association_model_from_blocks(
        (block,),
        options=MonotoneRankerTrainingOptions(
            row_negatives_per_positive=1,
            column_negatives_per_positive=1,
            max_preference_pairs=None,
            learning_rate=0.2,
            max_iter=200,
            l2_regularization=1.0e-4,
        ),
    )
    model = calibrated_model.model
    costs = model.pairwise_cost_matrix(block.features)
    probabilities = model.predict_match_probability(block.features)

    assert np.all(model.weights >= 0.0)
    assert model.n_preference_pairs == 4
    assert costs[0, 0] < costs[0, 1]
    assert costs[1, 1] < costs[1, 0]
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_monotone_ranker_ignores_non_cost_like_features_by_default():
    block = _toy_rank_block()
    features = np.concatenate(
        [
            block.features,
            np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=float),
        ],
        axis=-1,
    )
    block = ReferencePairwiseExamples(
        session_a=block.session_a,
        session_b=block.session_b,
        features=features,
        labels=block.labels,
        reference_roi_indices=block.reference_roi_indices,
        measurement_roi_indices=block.measurement_roi_indices,
        feature_names=("one_minus_iou", "session_gap", "activity_similarity_available"),
    )

    calibrated_model = fit_monotone_ranked_association_model_from_blocks((block,))
    model = calibrated_model.model

    assert "activity_similarity_available" not in model.trainable_feature_names
    assert model.weights[2] == pytest.approx(0.0)
