from __future__ import annotations

import numpy as np

from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)


def _example_block() -> ReferencePairwiseExamples:
    feature_names = ("centroid_distance", "one_minus_iou", "session_gap")
    centroid = np.array(
        [
            [0.0, 3.0, 8.0],
            [4.0, 0.0, 7.0],
            [8.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    one_minus_iou = np.array(
        [
            [0.0, 0.8, 1.0],
            [0.7, 0.0, 1.0],
            [1.0, 0.9, 0.0],
        ],
        dtype=float,
    )
    features = np.stack([centroid, one_minus_iou, np.ones_like(centroid)], axis=-1)
    labels = np.eye(3, dtype=int)
    return ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=labels,
        reference_roi_indices=np.array([0, 1, 2], dtype=int),
        measurement_roi_indices=np.array([0, 1, 2], dtype=int),
        feature_names=feature_names,
    )


def test_monotone_ranker_learns_nonnegative_weights_and_ranks_gt_edges():
    block = _example_block()

    model = fit_monotone_ranking_association_model_from_blocks(
        [block],
        options=MonotoneRankerOptions(
            max_iter=500,
            learning_rate=0.1,
            binary_loss_weight=0.05,
        ),
    )

    costs = model.pairwise_cost_matrix(block.features)
    row_negative_min = np.min(np.where(block.labels == 0, costs, np.inf), axis=1)

    assert np.all(model.weights >= 0.0)
    assert int(model.n_rank_constraints) > 0
    assert np.all(np.diag(costs) < row_negative_min)
    assert np.all(np.diag(model.predict_match_probability(block.features)) > 0.5)


def test_monotone_ranker_cost_is_monotone_in_badness_features():
    model = fit_monotone_ranking_association_model_from_blocks(
        [_example_block()],
        options=MonotoneRankerOptions(max_iter=500, learning_rate=0.1),
    )

    ordered_features = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.5, 1.0],
        ],
        dtype=float,
    )
    costs = model.pairwise_cost_matrix(ordered_features)

    assert costs[0] <= costs[1] <= costs[2]
