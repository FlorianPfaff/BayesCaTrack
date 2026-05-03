"""Trace similarity components for calibrated ROI association."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

_TRACE_FIELDS = ("spike_traces", "traces", "neuropil_traces")
_AUTO_TRACE_FIELDS = ("spike_traces", "traces")


def add_activity_similarity_components(
    pairwise_components: MutableMapping[str, np.ndarray],
    reference_plane: Any,
    measurement_plane: Any,
    *,
    trace_source: str = "auto",
    similarity_epsilon: float = 1.0e-12,
) -> MutableMapping[str, np.ndarray]:
    """Add optional pairwise trace-similarity matrices in place."""

    pairwise_components.update(
        activity_similarity_components(
            reference_plane,
            measurement_plane,
            trace_source=trace_source,
            similarity_epsilon=similarity_epsilon,
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
) -> dict[str, np.ndarray]:
    """Return pairwise trace-correlation components for two ROI planes."""

    if similarity_epsilon <= 0.0:
        raise ValueError("similarity_epsilon must be strictly positive")

    shape = (int(reference_plane.n_rois), int(measurement_plane.n_rois))
    reference_traces, measurement_traces = _resolve_trace_arrays(
        reference_plane,
        measurement_plane,
        trace_source=trace_source,
    )
    if reference_traces is None or measurement_traces is None:
        return _neutral_activity_components(shape)

    reference_traces = np.asarray(reference_traces, dtype=float)
    measurement_traces = np.asarray(measurement_traces, dtype=float)
    if reference_traces.ndim != 2 or measurement_traces.ndim != 2:
        return _neutral_activity_components(shape)

    n_timepoints = min(reference_traces.shape[1], measurement_traces.shape[1])
    if n_timepoints <= 0:
        return _neutral_activity_components(shape)

    reference_unit, reference_valid = _row_normalized_trace_vectors(
        reference_traces[:, :n_timepoints],
        similarity_epsilon=similarity_epsilon,
    )
    measurement_unit, measurement_valid = _row_normalized_trace_vectors(
        measurement_traces[:, :n_timepoints],
        similarity_epsilon=similarity_epsilon,
    )

    correlations = np.clip(reference_unit @ measurement_unit.T, -1.0, 1.0)
    available = reference_valid[:, None] & measurement_valid[None, :]
    similarity = np.where(available, 0.5 * (correlations + 1.0), 0.0)
    cost = np.where(available, 1.0 - similarity, 0.5)

    return {
        "activity_correlation": correlations,
        "activity_similarity": similarity,
        "activity_similarity_cost": cost,
        "activity_similarity_available": available.astype(float),
    }


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


def _row_normalized_trace_vectors(
    traces: np.ndarray,
    *,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.nan_to_num(
        np.asarray(traces, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > similarity_epsilon
    normalized = np.zeros_like(centered, dtype=float)
    normalized[valid] = centered[valid] / norms[valid, None]
    return normalized, valid


def _neutral_activity_components(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "activity_correlation": np.zeros(shape, dtype=float),
        "activity_similarity": np.zeros(shape, dtype=float),
        "activity_similarity_cost": np.full(shape, 0.5, dtype=float),
        "activity_similarity_available": np.zeros(shape, dtype=float),
    }
