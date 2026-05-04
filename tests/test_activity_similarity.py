from __future__ import annotations

import numpy as np
import numpy.testing as npt
from bayescatrack.association.activity_similarity import (
    activity_similarity_components,
    add_activity_similarity_components,
)
from bayescatrack.association.calibrated_costs import pairwise_feature_tensor
from bayescatrack.core.bridge import CalciumPlaneData


def _masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    return masks


def test_activity_similarity_favors_matching_trace_structure() -> None:
    masks = _masks()
    traces = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces.copy())

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(np.diag(components["activity_correlation"]), np.ones(2))
    npt.assert_allclose(np.diag(components["activity_similarity"]), np.ones(2))
    npt.assert_allclose(np.diag(components["activity_similarity_cost"]), np.zeros(2))
    assert (
        components["activity_similarity"][0, 0]
        > components["activity_similarity"][0, 1]
    )
    assert (
        components["activity_similarity"][1, 1]
        > components["activity_similarity"][1, 0]
    )
    npt.assert_allclose(components["activity_similarity_available"], np.ones((2, 2)))


def test_activity_similarity_prefers_spike_traces_in_auto_mode() -> None:
    masks = _masks()
    fluorescence = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    spikes = fluorescence[::-1]
    reference = CalciumPlaneData(
        roi_masks=masks, traces=fluorescence, spike_traces=spikes
    )
    measurement = CalciumPlaneData(
        roi_masks=masks, traces=fluorescence, spike_traces=spikes
    )

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(np.diag(components["activity_similarity"]), np.ones(2))
    assert (
        components["activity_similarity"][0, 0]
        > components["activity_similarity"][0, 1]
    )


def test_activity_similarity_hook_is_neutral_when_traces_are_missing() -> None:
    masks = _masks()
    reference = CalciumPlaneData(roi_masks=masks)
    measurement = CalciumPlaneData(roi_masks=masks)

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(components["activity_correlation"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_similarity"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_similarity_cost"], np.full((2, 2), 0.5))
    npt.assert_allclose(components["activity_similarity_available"], np.zeros((2, 2)))


def test_activity_similarity_components_can_be_added_to_pairwise_component_dict() -> (
    None
):
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces)
    pairwise_components = {"pairwise_cost_matrix": np.zeros((2, 2), dtype=float)}

    add_activity_similarity_components(pairwise_components, reference, measurement)
    features = pairwise_feature_tensor(
        pairwise_components,
        feature_names=("activity_similarity_cost", "activity_similarity_available"),
    )

    assert features.shape == (2, 2, 2)
    npt.assert_allclose(
        features[:, :, 0], pairwise_components["activity_similarity_cost"]
    )
    npt.assert_allclose(
        features[:, :, 1], pairwise_components["activity_similarity_available"]
    )


def test_pairwise_feature_tensor_accepts_missing_activity_hook_components() -> None:
    features = pairwise_feature_tensor(
        {"pairwise_cost_matrix": np.zeros((2, 3), dtype=float)},
        feature_names=("activity_similarity_cost", "activity_similarity_available"),
    )

    assert features.shape == (2, 3, 2)
    npt.assert_allclose(features, np.zeros((2, 3, 2)))
