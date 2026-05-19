"""Tests for calibrated activity feature down-weighting."""

import numpy as np
import pytest

from bayescatrack.association.calibrated_costs import pairwise_feature_tensor


def test_activity_feature_weight_scales_continuous_activity_features_only() -> None:
    components = {
        "centroid_distance": np.array([[2.0]]),
        "activity_similarity_cost": np.array([[0.8]]),
        "activity_similarity_available": np.array([[1.0]]),
    }

    features = pairwise_feature_tensor(
        components,
        feature_names=(
            "centroid_distance",
            "activity_similarity_cost",
            "activity_similarity_available",
        ),
        activity_feature_weight=0.25,
    )

    np.testing.assert_allclose(features[0, 0], np.array([2.0, 0.2, 1.0]))


def test_one_minus_activity_similarity_is_scaled() -> None:
    components = {
        "activity_similarity": np.array([[0.75]]),
    }

    features = pairwise_feature_tensor(
        components,
        feature_names=("one_minus_activity_similarity",),
        activity_feature_weight=0.2,
    )

    np.testing.assert_allclose(features[0, 0], np.array([0.05]))


def test_activity_feature_weight_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_feature_tensor(
            {"activity_similarity_cost": np.array([[0.5]])},
            feature_names=("activity_similarity_cost",),
            activity_feature_weight=-1.0,
        )
