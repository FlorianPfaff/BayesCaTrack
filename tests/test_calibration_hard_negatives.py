from __future__ import annotations

import numpy as np
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    balanced_binary_sample_weights,
    collect_candidate_limited_training_examples,
)


def _example_block() -> ReferencePairwiseExamples:
    feature_names = ("centroid_distance", "one_minus_iou", "session_gap")
    centroid = np.array(
        [
            [0.0, 0.1, 9.0],
            [0.2, 0.0, 8.0],
            [7.0, 0.3, 0.0],
        ],
        dtype=float,
    )
    one_minus_iou = np.array(
        [
            [0.0, 0.2, 1.0],
            [0.3, 0.0, 1.0],
            [1.0, 0.4, 0.0],
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


def test_candidate_limited_hard_negatives_keep_all_positives_and_hard_negatives():
    features, labels = collect_candidate_limited_training_examples(
        [_example_block()],
        options=CandidateHardNegativeOptions(
            negative_to_positive_ratio=1.0,
            candidate_top_k_per_anchor=1,
            include_column_candidates=False,
            hardness_feature_names=("centroid_distance",),
        ),
    )

    assert int(np.sum(labels == 1)) == 3
    assert int(np.sum(labels == 0)) == 3
    assert features.shape == (6, 3)
    selected_centroid_distances = sorted(
        float(value) for value in features[labels == 0, 0]
    )
    assert selected_centroid_distances == [0.1, 0.2, 0.3]


def test_balanced_binary_sample_weights_are_inverse_frequency():
    weights = balanced_binary_sample_weights(np.array([1, 0, 0, 0], dtype=int))

    assert weights[0] == 2.0
    assert np.allclose(weights[1:], 2.0 / 3.0)
    assert np.isclose(float(np.sum(weights)), 4.0)
