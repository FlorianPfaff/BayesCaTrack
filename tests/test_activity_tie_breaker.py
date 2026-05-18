from __future__ import annotations

import numpy as np
import numpy.testing as npt
from bayescatrack.association.activity_similarity import (
    ACTIVITY_TIE_BREAKER_FEATURES,
    ACTIVITY_TIEBREAKER_FEATURES,
    activity_similarity_components,
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.calibrated_costs import pairwise_feature_tensor
from bayescatrack.core.bridge import CalciumPlaneData


def _masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    return masks


def test_activity_tie_breaker_components_expose_weak_activity_cues() -> None:
    masks = _masks()
    fluorescence = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    spikes = np.array(
        [
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    neuropil = np.array(
        [
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reference = CalciumPlaneData(
        roi_masks=masks,
        traces=fluorescence,
        spike_traces=spikes,
        neuropil_traces=neuropil,
    )
    measurement = CalciumPlaneData(
        roi_masks=masks,
        traces=fluorescence.copy(),
        spike_traces=spikes.copy(),
        neuropil_traces=neuropil.copy(),
    )

    components = activity_similarity_components(reference, measurement)

    for component_name in (
        "fluorescence_activity_similarity_cost",
        "spike_activity_similarity_cost",
        "activity_event_rate_absdiff",
        "activity_trace_std_absdiff",
        "activity_trace_skew_absdiff",
        "activity_neuropil_ratio_absdiff",
        "activity_tiebreaker_cost",
    ):
        npt.assert_allclose(np.diag(components[component_name]), np.zeros(2))
        assert components[component_name][0, 1] > 0.0

    for availability_name in (
        "fluorescence_activity_similarity_available",
        "spike_activity_similarity_available",
        "activity_event_rate_available",
        "activity_trace_std_available",
        "activity_trace_skew_available",
        "activity_neuropil_ratio_available",
        "activity_available_indicator",
    ):
        npt.assert_allclose(components[availability_name], np.ones((2, 2)))


def test_activity_tie_breaker_is_neutral_and_flagged_when_traces_are_missing() -> None:
    masks = _masks()
    reference = CalciumPlaneData(roi_masks=masks)
    measurement = CalciumPlaneData(roi_masks=masks)

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(components["activity_similarity_cost"], np.full((2, 2), 0.5))
    npt.assert_allclose(
        components["fluorescence_activity_similarity_cost"], np.full((2, 2), 0.5)
    )
    npt.assert_allclose(
        components["spike_activity_similarity_cost"], np.full((2, 2), 0.5)
    )
    npt.assert_allclose(components["activity_available_indicator"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_missing"], np.ones((2, 2)))
    npt.assert_allclose(components["activity_tiebreaker_cost"], np.full((2, 2), 0.5))
    npt.assert_allclose(components["activity_tiebreaker_missing"], np.ones((2, 2)))


def test_activity_tie_breaker_cost_matrix_scales_selected_component() -> None:
    pairwise_components = {
        "activity_similarity_cost": np.array(
            [[0.0, 0.5], [1.0, np.nan]], dtype=float
        )
    }

    tie_breaker_cost = activity_tie_breaker_cost_matrix(
        pairwise_components, weight=0.1
    )

    npt.assert_allclose(tie_breaker_cost, np.array([[0.0, 0.05], [0.1, 0.05]]))


def test_activity_tie_breaker_features_work_with_calibrated_feature_tensor() -> None:
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces)
    components = {
        "pairwise_cost_matrix": np.zeros((2, 2), dtype=float),
        **activity_similarity_components(reference, measurement),
    }

    for feature_names in (ACTIVITY_TIE_BREAKER_FEATURES, ACTIVITY_TIEBREAKER_FEATURES):
        features = pairwise_feature_tensor(components, feature_names=feature_names)
        assert features.shape == (2, 2, len(feature_names))
        assert np.all(np.isfinite(features))
