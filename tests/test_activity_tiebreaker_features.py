from __future__ import annotations

import numpy as np
import numpy.testing as npt
from bayescatrack.association.activity_similarity import (
    ACTIVITY_TIEBREAKER_FEATURES,
    activity_similarity_components,
)
from bayescatrack.association.calibrated_costs import pairwise_feature_tensor
from bayescatrack.core.bridge import CalciumPlaneData


def _masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    return masks


def test_activity_tiebreaker_components_expose_weak_activity_cues() -> None:
    masks = _masks()
    traces = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    spikes = np.array(
        [
            [0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    neuropil = np.array(
        [
            [0.0, 0.1, 0.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reference = CalciumPlaneData(
        roi_masks=masks,
        traces=traces,
        spike_traces=spikes,
        neuropil_traces=neuropil,
    )
    measurement = CalciumPlaneData(
        roi_masks=masks,
        traces=traces.copy(),
        spike_traces=spikes.copy(),
        neuropil_traces=neuropil.copy(),
    )

    components = activity_similarity_components(reference, measurement)

    for component_name in (
        "fluorescence_similarity_cost",
        "spike_similarity_cost",
        "trace_std_absdiff",
        "trace_skew_absdiff",
        "event_rate_absdiff",
        "neuropil_ratio_absdiff",
        "activity_tiebreaker_cost",
    ):
        npt.assert_allclose(np.diag(components[component_name]), np.zeros(2))
        assert components[component_name][0, 1] > 0.0

    for availability_name in (
        "fluorescence_similarity_available",
        "spike_similarity_available",
        "trace_std_available",
        "trace_skew_available",
        "event_rate_available",
        "neuropil_ratio_available",
        "activity_tiebreaker_available",
    ):
        npt.assert_allclose(components[availability_name], np.ones((2, 2)))


def test_activity_tiebreaker_explicitly_flags_missing_optional_sources() -> None:
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces)

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(
        components["fluorescence_similarity_available"], np.ones((2, 2))
    )
    npt.assert_allclose(components["spike_similarity_available"], np.zeros((2, 2)))
    npt.assert_allclose(components["event_rate_available"], np.zeros((2, 2)))
    npt.assert_allclose(components["neuropil_ratio_available"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_tiebreaker_available"], np.ones((2, 2)))
    npt.assert_allclose(components["activity_tiebreaker_missing"], np.zeros((2, 2)))


def test_activity_tiebreaker_features_work_with_calibrated_feature_tensor() -> None:
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces)
    components = {
        "pairwise_cost_matrix": np.zeros((2, 2), dtype=float),
        **activity_similarity_components(reference, measurement),
    }

    features = pairwise_feature_tensor(
        components,
        feature_names=ACTIVITY_TIEBREAKER_FEATURES,
    )

    assert features.shape == (2, 2, len(ACTIVITY_TIEBREAKER_FEATURES))
    assert np.all(np.isfinite(features))
