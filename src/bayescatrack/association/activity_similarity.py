"""Trace similarity components for calibrated ROI association."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

_TRACE_FIELDS = ("spike_traces", "traces", "neuropil_traces")
_AUTO_TRACE_FIELDS = ("spike_traces", "traces")

ACTIVITY_TIE_BREAKER_FEATURES = (
    "activity_similarity_cost",
    "activity_similarity_available",
    "fluorescence_activity_similarity_cost",
    "fluorescence_activity_similarity_available",
    "spike_activity_similarity_cost",
    "spike_activity_similarity_available",
    "activity_event_rate_absdiff",
    "activity_event_rate_available",
    "activity_trace_std_absdiff",
    "activity_trace_std_available",
    "activity_trace_skew_absdiff",
    "activity_trace_skew_available",
    "activity_neuropil_ratio_absdiff",
    "activity_neuropil_ratio_available",
    "activity_available_indicator",
)

# Backward-compatible shorter names used by earlier experiments.
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


# pylint: disable=too-many-arguments
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
    """Return pairwise activity components for two ROI planes.

    The legacy ``activity_*`` components use ``trace_source`` and remain backward
    compatible with earlier calibrated models. Additional components expose
    fluorescence similarity, deconvolved-spike similarity, event-rate differences,
    trace moment differences, and neuropil-ratio differences as weak tie-breaker
    features. Missing trace families are represented by neutral cost planes plus
    explicit availability indicators rather than by perfect matches.
    """

    if similarity_epsilon <= 0.0:
        raise ValueError("similarity_epsilon must be strictly positive")
    if not np.isfinite(event_threshold):
        raise ValueError("event_threshold must be finite")

    shape = (int(reference_plane.n_rois), int(measurement_plane.n_rois))
    reference_traces, measurement_traces = _resolve_trace_arrays(
        reference_plane,
        measurement_plane,
        trace_source=trace_source,
    )

    components = _trace_similarity_components(
        reference_traces,
        measurement_traces,
        shape,
        prefix="activity",
        similarity_epsilon=similarity_epsilon,
    )

    fluorescence_components = _trace_similarity_components(
        getattr(reference_plane, "traces", None),
        getattr(measurement_plane, "traces", None),
        shape,
        prefix="fluorescence_activity",
        similarity_epsilon=similarity_epsilon,
    )
    _add_similarity_aliases(fluorescence_components, "fluorescence_activity", "fluorescence")
    components.update(fluorescence_components)

    spike_components = _trace_similarity_components(
        getattr(reference_plane, "spike_traces", None),
        getattr(measurement_plane, "spike_traces", None),
        shape,
        prefix="spike_activity",
        similarity_epsilon=similarity_epsilon,
    )
    _add_similarity_aliases(spike_components, "spike_activity", "spike")
    components.update(spike_components)

    event_rate = _pairwise_absdiff_components(
        "activity_event_rate",
        _row_event_rates(getattr(reference_plane, "spike_traces", None), threshold=event_threshold),
        _row_event_rates(getattr(measurement_plane, "spike_traces", None), threshold=event_threshold),
        shape,
        scale_epsilon=similarity_epsilon,
    )
    _add_absdiff_aliases(event_rate, "activity_event_rate", "event_rate")
    components.update(event_rate)

    trace_std = _pairwise_absdiff_components(
        "activity_trace_std",
        _row_trace_stds(getattr(reference_plane, "traces", None)),
        _row_trace_stds(getattr(measurement_plane, "traces", None)),
        shape,
        scale_epsilon=similarity_epsilon,
    )
    _add_absdiff_aliases(trace_std, "activity_trace_std", "trace_std")
    components.update(trace_std)

    trace_skew = _pairwise_absdiff_components(
        "activity_trace_skew",
        _row_trace_skews(getattr(reference_plane, "traces", None), epsilon=similarity_epsilon),
        _row_trace_skews(getattr(measurement_plane, "traces", None), epsilon=similarity_epsilon),
        shape,
        scale_epsilon=similarity_epsilon,
    )
    _add_absdiff_aliases(trace_skew, "activity_trace_skew", "trace_skew")
    components.update(trace_skew)

    neuropil_ratio = _pairwise_absdiff_components(
        "activity_neuropil_ratio",
        _row_neuropil_ratios(
            getattr(reference_plane, "traces", None),
            getattr(reference_plane, "neuropil_traces", None),
            epsilon=similarity_epsilon,
        ),
        _row_neuropil_ratios(
            getattr(measurement_plane, "traces", None),
            getattr(measurement_plane, "neuropil_traces", None),
            epsilon=similarity_epsilon,
        ),
        shape,
        scale_epsilon=similarity_epsilon,
    )
    _add_absdiff_aliases(neuropil_ratio, "activity_neuropil_ratio", "neuropil_ratio")
    components.update(neuropil_ratio)

    components.update(_combined_activity_tie_breaker_components(components, shape))
    return components


def activity_tie_breaker_cost_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    component_name: str = "activity_similarity_cost",
    weight: float = 0.05,
) -> np.ndarray:
    """Return a low-weight activity cost plane for additive tie-breaking."""

    weight = float(weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Pairwise component {component_name!r} must be two-dimensional")
    return weight * np.nan_to_num(values, nan=0.5, posinf=1.0e6, neginf=0.0)


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


def _trace_similarity_components(
    reference_traces: Any,
    measurement_traces: Any,
    shape: tuple[int, int],
    *,
    prefix: str,
    similarity_epsilon: float,
) -> dict[str, np.ndarray]:
    reference_traces = _as_trace_matrix(reference_traces)
    measurement_traces = _as_trace_matrix(measurement_traces)
    if reference_traces is None or measurement_traces is None:
        return _neutral_similarity_components(prefix, shape)
    if reference_traces.shape[0] != shape[0] or measurement_traces.shape[0] != shape[1]:
        return _neutral_similarity_components(prefix, shape)

    n_timepoints = min(reference_traces.shape[1], measurement_traces.shape[1])
    if n_timepoints <= 0:
        return _neutral_similarity_components(prefix, shape)

    reference_unit, reference_valid = _row_normalized_trace_vectors(
        reference_traces[:, :n_timepoints], similarity_epsilon=similarity_epsilon
    )
    measurement_unit, measurement_valid = _row_normalized_trace_vectors(
        measurement_traces[:, :n_timepoints], similarity_epsilon=similarity_epsilon
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


def _row_event_rates(traces: Any, *, threshold: float) -> tuple[np.ndarray, np.ndarray] | None:
    traces = _as_trace_matrix(traces)
    if traces is None:
        return None
    finite = np.isfinite(traces)
    counts = np.sum(finite, axis=1)
    valid = counts > 0
    events = (traces > threshold) & finite
    values = np.zeros(traces.shape[0], dtype=float)
    values[valid] = np.sum(events[valid], axis=1) / counts[valid]
    return values, valid.astype(bool)


def _row_trace_stds(traces: Any) -> tuple[np.ndarray, np.ndarray] | None:
    traces = _as_trace_matrix(traces)
    if traces is None:
        return None
    means, variances, valid = _finite_row_mean_and_variance(traces)
    del means
    return np.sqrt(np.maximum(variances, 0.0)), valid.astype(bool)


def _row_trace_skews(
    traces: Any,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    traces = _as_trace_matrix(traces)
    if traces is None:
        return None
    means, variances, valid = _finite_row_mean_and_variance(traces)
    finite = np.isfinite(traces)
    counts = np.sum(finite, axis=1)
    centered = np.where(finite, traces - means[:, None], 0.0)
    third_moments = np.zeros(traces.shape[0], dtype=float)
    third_moments[valid] = np.sum(centered[valid] ** 3, axis=1) / counts[valid]
    stds = np.sqrt(np.maximum(variances, 0.0))
    values = np.zeros(traces.shape[0], dtype=float)
    non_constant = valid & (stds > epsilon)
    values[non_constant] = third_moments[non_constant] / (stds[non_constant] ** 3)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid.astype(bool)


def _row_neuropil_ratios(
    fluorescence_traces: Any,
    neuropil_traces: Any,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    fluorescence = _as_trace_matrix(fluorescence_traces)
    neuropil = _as_trace_matrix(neuropil_traces)
    if fluorescence is None or neuropil is None:
        return None
    if fluorescence.shape[0] != neuropil.shape[0]:
        return None
    n_timepoints = min(fluorescence.shape[1], neuropil.shape[1])
    if n_timepoints <= 0:
        return None
    fluorescence = fluorescence[:, :n_timepoints]
    neuropil = neuropil[:, :n_timepoints]
    finite = np.isfinite(fluorescence) & np.isfinite(neuropil)
    counts = np.sum(finite, axis=1)
    valid = counts > 0
    fluorescence_values = np.where(finite, fluorescence, 0.0)
    neuropil_values = np.where(finite, neuropil, 0.0)
    fluorescence_mean = np.zeros(fluorescence.shape[0], dtype=float)
    neuropil_mean = np.zeros(neuropil.shape[0], dtype=float)
    fluorescence_mean[valid] = np.sum(fluorescence_values[valid], axis=1) / counts[valid]
    neuropil_mean[valid] = np.sum(neuropil_values[valid], axis=1) / counts[valid]
    denominator = np.maximum(np.abs(fluorescence_mean), epsilon)
    ratios = neuropil_mean / denominator
    valid = valid & np.isfinite(ratios)
    return np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0), valid.astype(bool)


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


def _pairwise_absdiff_components(
    prefix: str,
    reference_feature: tuple[np.ndarray, np.ndarray] | None,
    measurement_feature: tuple[np.ndarray, np.ndarray] | None,
    shape: tuple[int, int],
    *,
    scale_epsilon: float,
) -> dict[str, np.ndarray]:
    value_name = f"{prefix}_absdiff"
    availability_name = f"{prefix}_available"
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


def _combined_activity_tie_breaker_components(
    components: Mapping[str, np.ndarray], shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    cost_names = (
        "fluorescence_activity_similarity_cost",
        "spike_activity_similarity_cost",
        "activity_trace_std_absdiff",
        "activity_trace_skew_absdiff",
        "activity_event_rate_absdiff",
        "activity_neuropil_ratio_absdiff",
    )
    availability_names = (
        "fluorescence_activity_similarity_available",
        "spike_activity_similarity_available",
        "activity_trace_std_available",
        "activity_trace_skew_available",
        "activity_event_rate_available",
        "activity_neuropil_ratio_available",
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
        "activity_any_available": available.astype(float),
        "activity_available_indicator": available.astype(float),
        "activity_missing": (~available).astype(float),
    }


def _add_similarity_aliases(components: dict[str, np.ndarray], source_prefix: str, alias_prefix: str) -> None:
    for suffix in ("correlation", "similarity", "similarity_cost", "similarity_available"):
        components[f"{alias_prefix}_{suffix}"] = components[f"{source_prefix}_{suffix}"]


def _add_absdiff_aliases(components: dict[str, np.ndarray], source_prefix: str, alias_prefix: str) -> None:
    components[f"{alias_prefix}_absdiff"] = components[f"{source_prefix}_absdiff"]
    components[f"{alias_prefix}_available"] = components[f"{source_prefix}_available"]


def _neutral_similarity_components(prefix: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_correlation": np.zeros(shape, dtype=float),
        f"{prefix}_similarity": np.zeros(shape, dtype=float),
        f"{prefix}_similarity_cost": np.full(shape, 0.5, dtype=float),
        f"{prefix}_similarity_available": np.zeros(shape, dtype=float),
    }


def _neutral_activity_components(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return _neutral_similarity_components("activity", shape)
