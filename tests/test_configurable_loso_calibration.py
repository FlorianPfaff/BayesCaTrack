from __future__ import annotations

# pylint: disable=protected-access,too-few-public-methods

import numpy as np
from bayescatrack.experiments import track2p_configurable_loso_calibration as loso


def test_configurable_loso_parser_builds_hard_negative_options_and_model_kwargs():
    args = loso.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--hard-negative-ratio",
            "7.5",
            "--hard-negative-top-k",
            "none",
            "--no-hard-negative-column-candidates",
            "--hard-negative-features",
            "centroid_distance,one_minus_iou",
            "--sample-weight-strategy",
            "balanced",
            "--calibration-model",
            "hist-gradient-boosting",
            "--calibration-model-kwargs-json",
            '{"max_iter":25}',
        ]
    )

    options = loso._hard_negative_options(args)

    assert args.sample_weight_strategy == "balanced"
    assert args.calibration_model == "hist-gradient-boosting"
    assert loso._json_object(args.calibration_model_kwargs_json, "model") == {
        "max_iter": 25
    }
    assert options.negative_to_positive_ratio == 7.5
    assert options.candidate_top_k_per_anchor is None
    assert not options.include_column_candidates
    assert options.hardness_feature_names == ("centroid_distance", "one_minus_iou")


def test_sklearn_probability_adapter_preserves_pairwise_tensor_shape():
    class DummyEstimator:
        def __init__(self):
            self.classes_ = None
            self.n_fit_examples = 0
            self.received_sample_weight = None

        def fit(self, features, _labels, **kwargs):
            self.classes_ = np.array([0, 1], dtype=int)
            self.n_fit_examples = int(features.shape[0])
            self.received_sample_weight = kwargs.get("sample_weight")
            return self

        def predict_proba(self, features):
            probabilities = np.linspace(0.1, 0.9, num=features.shape[0])
            return np.column_stack([1.0 - probabilities, probabilities])

    adapter = loso.SklearnPairwiseProbabilityAdapter(DummyEstimator())
    features = np.zeros((2, 3, 4), dtype=float)
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
    sample_weight = np.ones(labels.size, dtype=float)

    adapter.fit(features, labels, sample_weight=sample_weight)
    probabilities = adapter.predict_match_probability(features)

    assert adapter.estimator.n_fit_examples == 6
    assert adapter.estimator.received_sample_weight.shape == (6,)
    assert probabilities.shape == (2, 3)
    assert np.isclose(probabilities[0, 0], 0.1)
    assert np.isclose(probabilities[-1, -1], 0.9)
