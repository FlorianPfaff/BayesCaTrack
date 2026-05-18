from __future__ import annotations

import numpy as np

from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranking_costs import (
    MonotoneRankerOptions,
    fit_monotone_ranked_association_model,
)


def test_monotone_ranker_orders_positive_edges_above_hard_negatives():
    feature_names = ("centroid_distance", "one_minus_iou")
    features = np.array(
        [
            [[0.1, 0.1], [2.0, 0.8], [3.0, 0.9]],
            [[2.0, 0.7], [0.1, 0.1], [3.0, 0.9]],
            [[2.0, 0.8], [3.0, 0.9], [0.1, 0.1]],
        ],
        dtype=float,
    )
    labels = np.eye(3, dtype=int)
    block = ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=labels,
        reference_roi_indices=np.arange(3),
        measurement_roi_indices=np.arange(3),
        feature_names=feature_names,
    )

    calibrated_model = fit_monotone_ranked_association_model(
        [block],
        options=MonotoneRankerOptions(max_iter=200),
    )
    ranker = calibrated_model.model

    assert np.all(ranker.weights >= 0.0)
    assert ranker.preference_pairs == 12
    assert ranker.positive_examples == 3
    scores = ranker.predict_score(features)
    costs = ranker.pairwise_cost_matrix(features)
    probabilities = ranker.predict_match_probability(features)

    for index in range(3):
        false_columns = [column for column in range(3) if column != index]
        assert scores[index, index] < np.min(scores[index, false_columns])
        assert costs[index, index] < np.min(costs[index, false_columns])
        assert probabilities[index, index] > np.max(probabilities[index, false_columns])


def test_monotone_ranker_treats_raw_similarity_features_as_benefits():
    feature_names = ("mask_cosine_similarity",)
    features = np.array([[[0.9], [0.2]], [[0.1], [0.8]]], dtype=float)
    labels = np.eye(2, dtype=int)
    block = ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=labels,
        reference_roi_indices=np.arange(2),
        measurement_roi_indices=np.arange(2),
        feature_names=feature_names,
    )

    calibrated_model = fit_monotone_ranked_association_model(
        [block],
        options=MonotoneRankerOptions(max_iter=200),
    )
    ranker = calibrated_model.model

    assert ranker.feature_directions == (-1.0,)
    assert ranker.predict_score(features)[0, 0] < ranker.predict_score(features)[0, 1]
    assert ranker.predict_score(features)[1, 1] < ranker.predict_score(features)[1, 0]
