"""Tests for shifted-overlap calibrated-association feature presets."""

import numpy as np
import numpy.testing as npt

from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS,
    SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
    pairwise_feature_tensor,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    calibration_feature_names,
    pairwise_cost_kwargs_for_calibration_features,
)


def test_shifted_overlap_feature_preset_extends_default_features():
    feature_names = calibration_feature_names("default+shifted-overlap")

    assert (
        feature_names[: len(DEFAULT_ASSOCIATION_FEATURES)]
        == DEFAULT_ASSOCIATION_FEATURES
    )
    assert set(SHIFTED_OVERLAP_ASSOCIATION_FEATURES).issubset(feature_names)

    pairwise_kwargs = pairwise_cost_kwargs_for_calibration_features(None, feature_names)

    assert pairwise_kwargs == DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS


def test_shifted_overlap_features_are_optional_zero_without_components():
    components = {"centroid_distance": np.ones((2, 3), dtype=float)}

    features = pairwise_feature_tensor(
        components, feature_names=SHIFTED_OVERLAP_ASSOCIATION_FEATURES
    )

    assert features.shape == (2, 3, len(SHIFTED_OVERLAP_ASSOCIATION_FEATURES))
    npt.assert_allclose(features, 0.0)
