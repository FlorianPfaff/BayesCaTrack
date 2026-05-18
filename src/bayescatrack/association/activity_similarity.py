"""Trace similarity components for calibrated ROI association."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

_TRACE_FIELDS = ("spike_traces", "traces", "neuropil_traces")
_AUTO_TRACE_FIELDS = ("spike_traces", "traces")
_SOURCE_COMPONENTS = {
    "traces": "fluorescence",
    "spike_traces": "spike",
    "neuropil_traces": "neuropil",
}
ACTIVITY_TIEBREAKER_FEATURES = (
    "activity_tiebreaker_cost",
    "activity_tiebreaker_available",
    "activity_tiebreaker_missing",
    "fluorescence_similarity_cost",
    "fluorescence_similarity_available",
    "spike_similarity_cost",
    "spike_similarity_available",
    "trace_std_absdiff",
    "trace_std_available",
    "trace_skew_absdiff",
    "trace_skew_available",
    "event_rate_absdiff",
    "event_rate_available",
    "neuropil_ratio_absdiff",
    "neuropil_ratio_available",
)


def add_activity_similarity_components(
    pairwise_components: MutableMapping[str, np.ndarray],
    reference_plane: Any,
    measurement_plane: Any,
    *,
    trace_source: str = "auto",
    similarity_epsilon: float = 1.0e-12,
    event_threshold: float = 0.0,
) -> MutableMapping[str, np.ndarray]:
    """Add optional pairwise trace-similarity matrices in place."""

    pairwise_components.update(
        activity_similarity_components(
            reference_plane,
            measurement_plane,
            trace_source=trace_source,
            similarity_epsilon=similarity_epsilon,
            event_threshold=event_threshold,
        )
    )
    return pairwise_components


# pylint: disable=too-many-locals
def activity_similarity_components(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    trace_source: str = "auto",
    similarity_epsilon: float = 1.0e-12,
    event_threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return pairwise activity tie-breaker components for two ROI planes.

    The legacy ``activity_*`` components are still computed from ``trace_source``
    for backward compatibility. Additional named components expose fluorescence,
    spike, neuropil, event-rate, trace-shape, and neuropil-ratio cues separately
    with explicit availability indicators. Missing activity data is represented
    by a neutral cost and an availability value of zero rather than by a fake
    zero-similarity observation.
    """

    if similarity_epsilon <= 0.0:
        raise ValueError("similarity_epsilon must be strictly positive")

    shape = (int(reference_plane.n_rois), int(measurement_plane.n_rois))
    components: dict[str, np.ndarray] = {}

    reference_traces, measurement_traces = _resolve_trace_arrays(
        reference_plane,
        measurement_plane,
        trace_source=trace_source,
    )
    components.update(
        _trace_correlation_components(
            "activity",
            reference_traces,
            measurement_traces,
            shape,
            similarity_epsilon=similarity_epsilon,
        )
    )

    for field_name, prefix in _SOURCE_COMPONENTS.items():
        components.update(
            _trace_correlation_components(
                prefix,
                getattr(reference_plane, field_name, None),
                getattr(measurement_plane, field_name, None),
                shape,
                similarity_epsilon=similarity_epsilon,
            )
        )

    fluorescence_reference = _as_trace_matrix(getattr(reference_plane, "traces", None))
    fluorescence_measurement = _as_trace_matrix(
        getattr(measurement_plane, "traces", None)
    )
    spike_reference = _as_trace_matrix(getattr(reference_plane, "spike_traces", None))
    spike_measurement = _as_trace_matrix(getattr(measurement_plane, "spike_traces", None))
    neuropil_reference = _as_trace_matrix(
        getattr(reference_plane, "neuropil_traces", None)
    )
    neuropil_measurement = _as_trace_matrix(
        getattr(measurement_plane, "neuropil_traces", None)
    )

    components.update(
        _scaled_pairwise_absdiff_components(
            "trace_std_absdiff",
            _per_roi_trace_std(fluorescence_reference),
            _per_roi_trace_std(fluorescence_measurement),
            shape,
            availability_name="trace_std_available",
            scale_epsilon=similarity_epsilon,
        )
    )
    components.update(
        _scaled_pairwise_absdiff_components(
            "trace_skew_absdiff",
            _per_roi_trace_skew(
                fluorescence_reference, similarity_epsilon=similarity_epsilon
            ),
            _per_roi_trace_skew(
                fluorescence_measurement, similarity_epsilon=similarity_epsilon
            ),
            shape,
            availability_name="trace_skew_available",
            scale_epsilon=similarity_epsilon,
        )
    )
    components.update(
        _scaled_pairwise_absdiff_components(
            "event_rate_absdiff",
            _per_roi_event_rate(spike_reference, event_threshold=event_threshold),
            _per_roi_event_rate(spike_measurement, event_threshold=event_threshold),
            shape,
            availability_name="event_rate_available",
            scale_epsilon=similarity_epsilon,
        )
    )
    components.update(
        _scaled_pairwise_absdiff_components(
            "neuropil_ratio_absdiff",
            _per_roi_neuropil_ratio(
                fluorescence_reference,
                neuropil_reference,
                similarity_epsilon=similarity_epsilon,
            ),
            _per_roi_neuropil_ratio(
                fluorescence_measurement,
                neuropil_measurement,
                similarity_epsilon=similarity_epsilon,
            ),
            shape,
            availability_name="neuropil_ratio_available",
            scale_epsilon=similarity_epsilon,
        )
    )
    components.update(_activity_tiebreaker_components(components, shape))
    return components


def _resolve_trace_arrays(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    trace_source: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if trace_source == "auto":
        for field_name in _AUTO_TRACE_FIELDS:
            reference_traces = getattr(reference_plane, field_name, None)
            measurement_traces = getattr(measurement_plane, field_name, None)
            if reference_traces is not None and measurement_traces is not None:
                return reference_traces, measurement_traces
        return None, None

    if trace_source not in _TRACE_FIELDS:
        raise ValueError(f"Unsupported trace_source: {trace_source!r}")
    return getattr(reference_plane, trace_source, None), getattr(
        measurement_plane, trace_source, None
    )


def _trace_correlation_components(
    prefix: str,
    reference_traces: Any,
    measurement_traces: Any,
    shape: tuple[int, int],
    *,
    similarity_epsilon: float,
) -> dict[str, np.ndarray]:
    reference_matrix = _as_trace_matrix(reference_traces)
    measurement_matrix = _as_trace_matrix(measurement_traces)
    if reference_matrix is None or measurement_matrix is None:
        return _neutral_similarity_components(prefix, shape)

    if reference_matrix.shape[0] != shape[0] or measurement_matrix.shape[0] != shape[1]:
        return _neutral_similarity_components(prefix, shape)

    n_timepoints = min(reference_matrix.shape[1], measurement_matrix.shape[1])
    if n_timepoints <= 0:
        return _neutral_similarity_components(prefix, shape)

    reference_unit, reference_valid = _row_normalized_trace_vectors(
        reference_matrix[:, :n_timepoints],
        similarity_epsilon=similarity_epsilon,
    )
    measurement_unit, measurement_valid = _row_normalized_trace_vectors(
        measurement_matrix[:, :n_timepoints],
        similarity_epsilon=similarity_epsilon,
    )

    correlations = np.clip(reference_unit @ measurement_unit.T, -1.0, 1.0)
    available = reference_valid[:, None] & measurement_valid[None, :]
    similarity = np.where(available, 0.5 * (correlations + 1.0), 0.0)
    cost = np.where(available, 1.0 - similarity, 0.5)

    return {
        f"{prefix}_correlation": correlations,
        f"{prefix}_similarity": similarity,
        f"{prefix}_similarity_cost": cost,
        f"{prefix}_similarity_available": available.astype(float),
    }


def _as_trace_matrix(traces: Any) -> np.ndarray | None:
    if traces is None:
        return None
    trace_matrix = np.asarray(traces, dtype=float)
    if trace_matrix.ndim != 2:
        return None
    return trace_matrix


def _row_normalized_trace_vectors(
    traces: np.ndarray,
    *,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=float)
    finite_rows = np.all(np.isfinite(traces), axis=1)
    traces = np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = finite_rows & (norms > similarity_epsilon)
    normalized = np.zeros_like(centered, dtype=float)
    normalized[valid] = centered[valid] / norms[valid, None]
    return normalized, valid


def _per_roi_trace_std(traces: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if traces is None:
        return None
    traces = np.asarray(traces, dtype=float)
    means, variances, valid = _finite_row_mean_and_variance(traces)
    del means
    values = np.sqrt(np.maximum(variances, 0.0))
    return values.astype(float), valid.astype(bool)


def _per_roi_trace_skew(
    traces: np.ndarray | None,
    *,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if traces is None:
        return None
    traces = np.asarray(traces, dtype=float)
    means, variances, valid = _finite_row_mean_and_variance(traces)
    finite = np.isfinite(traces)
    centered = np.where(finite, traces - means[:, None], 0.0)
    counts = np.sum(finite, axis=1)
    third_moments = np.zeros(traces.shape[0], dtype=float)
    third_moments[valid] = np.sum(centered[valid] ** 3, axis=1) / counts[valid]
    stds = np.sqrt(np.maximum(variances, 0.0))
    values = np.zeros(traces.shape[0], dtype=float)
    non_constant = valid & (stds > similarity_epsilon)
    values[non_constant] = third_moments[non_constant] / (stds[non_constant] ** 3)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid.astype(bool)


def _finite_row_mean_and_variance(
    traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(traces)
    counts = np.sum(finite, axis=1)
    valid = counts > 0
    values = np.where(finite, traces, 0.0)
    means = np.zeros(traces.shape[0], dtype=float)
    means[valid] = np.sum(values[valid], axis=1) / counts[valid]
    centered = np.where(finite, traces - means[:, None], 0.0)
    variances = np.zeros(traces.shape[0], dtype=float)
    variances[valid] = np.sum(centered[valid] ** 2, axis=1) / counts[valid]
    return means, variances, valid.astype(bool)


def _per_roi_event_rate(
    traces: np.ndarray | None,
    *,
    event_threshold: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if traces is None:
        return None
    traces = np.asarray(traces, dtype=float)
    finite = np.isfinite(traces)
    counts = np.sum(finite, axis=1)
    valid = counts > 0
    events = (traces > event_threshold) & finite
    values = np.zeros(traces.shape[0], dtype=float)
    values[valid] = np.sum(events[valid], axis=1) / counts[valid]
    return values, valid.astype(bool)


def _per_roi_neuropil_ratio(
    fluorescence_traces: np.ndarray | None,
    neuropil_traces: np.ndarray | None,
    *,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if fluorescence_traces is None or neuropil_traces is None:
        return None
    n_timepoints = min(fluorescence_traces.shape[1], neuropil_traces.shape[1])
    if n_timepoints <= 0:
        return None
    fluorescence = fluorescence_traces[:, :n_timepoints]
    neuropil = neuropil_traces[:, :n_timepoints]
    finite = np.isfinite(fluorescence) & np.isfinite(neuropil)
    counts = np.sum(finite, axis=1)
    valid = counts > 0
    fluorescence_values = np.where(finite, fluorescence, 0.0)
    neuropil_values = np.where(finite, neuropil, 0.0)
    fluorescence_mean = np.zeros(fluorescence.shape[0], dtype=float)
    neuropil_mean = np.zeros(neuropil.shape[0], dtype=float)
    fluorescence_mean[valid] = np.sum(fluorescence_values[valid], axis=1) / counts[valid]
    neuropil_mean[valid] = np.sum(neuropil_values[valid], axis=1) / counts[valid]
    denominator = np.maximum(np.abs(fluorescence_mean), similarity_epsilon)
    values = neuropil_mean / denominator
    valid = valid & np.isfinite(values)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid.astype(bool)


def _scaled_pairwise_absdiff_components(
    value_name: str,
    reference_feature: tuple[np.ndarray, np.ndarray] | None,
    measurement_feature: tuple[np.ndarray, np.ndarray] | None,
    shape: tuple[int, int],
    *,
    availability_name: str,
    scale_epsilon: float,
) -> dict[str, np.ndarray]:
    if reference_feature is None or measurement_feature is None:
        return {
            value_name: np.zeros(shape, dtype=float),
            availability_name: np.zeros(shape, dtype=float),
        }
    reference_values, reference_valid = reference_feature
    measurement_values, measurement_valid = measurement_feature
    reference_values = np.asarray(reference_values, dtype=float).reshape(-1)
    measurement_values = np.asarray(measurement_values, dtype=float).reshape(-1)
    reference_valid = np.asarray(reference_valid, dtype=bool).reshape(-1)
    measurement_valid = np.asarray(measurement_valid, dtype=bool).reshape(-1)
    if reference_values.shape != (shape[0],) or measurement_values.shape != (shape[1],):
        raise ValueError(f"Feature {value_name!r} does not match pairwise shape")
    available = reference_valid[:, None] & measurement_valid[None, :]
    diff = np.abs(reference_values[:, None] - measurement_values[None, :])
    scale = _pooled_robust_scale(
        reference_values[reference_valid],
        measurement_values[measurement_valid],
        scale_epsilon=scale_epsilon,
    )
    cost = np.zeros(shape, dtype=float)
    cost[available] = diff[available] / scale
    return {
        value_name: np.nan_to_num(cost, nan=0.0, posinf=1.0e6, neginf=0.0),
        availability_name: available.astype(float),
    }


def _pooled_robust_scale(
    reference_values: np.ndarray,
    measurement_values: np.ndarray,
    *,
    scale_epsilon: float,
) -> float:
    pooled = np.concatenate(
        [np.asarray(reference_values, dtype=float), np.asarray(measurement_values, dtype=float)]
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size <= 1:
        return 1.0
    q75, q25 = np.percentile(pooled, [75.0, 25.0])
    robust_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    std_scale = float(np.std(pooled))
    scale = max(robust_scale, std_scale, scale_epsilon)
    return 1.0 if scale <= scale_epsilon else scale


def _activity_tiebreaker_components(
    components: dict[str, np.ndarray], shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    cost_names = (
        "fluorescence_similarity_cost",
        "spike_similarity_cost",
        "trace_std_absdiff",
        "trace_skew_absdiff",
        "event_rate_absdiff",
        "neuropil_ratio_absdiff",
    )
    availability_names = (
        "fluorescence_similarity_available",
        "spike_similarity_available",
        "trace_std_available",
        "trace_skew_available",
        "event_rate_available",
        "neuropil_ratio_available",
    )
    weighted_sum = np.zeros(shape, dtype=float)
    weights = np.zeros(shape, dtype=float)
    for cost_name, availability_name in zip(cost_names, availability_names):
        available = np.asarray(components[availability_name], dtype=float) > 0.0
        cost = np.asarray(components[cost_name], dtype=float)
        weighted_sum[available] += np.clip(cost[available], 0.0, 1.0)
        weights[available] += 1.0
    available = weights > 0.0
    tiebreaker_cost = np.full(shape, 0.5, dtype=float)
    tiebreaker_cost[available] = weighted_sum[available] / weights[available]
    return {
        "activity_tiebreaker_cost": tiebreaker_cost,
        "activity_tiebreaker_available": available.astype(float),
        "activity_tiebreaker_missing": (~available).astype(float),
    }


def _neutral_similarity_components(prefix: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_correlation": np.zeros(shape, dtype=float),
        f"{prefix}_similarity": np.zeros(shape, dtype=float),
        f"{prefix}_similarity_cost": np.full(shape, 0.5, dtype=float),
        f"{prefix}_similarity_available": np.zeros(shape, dtype=float),
    }


def _neutral_activity_components(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return _neutral_similarity_components("activity", shape)
