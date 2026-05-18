# pylint: disable=duplicate-code
"""Compatibility accessors for PyRecEst pairwise covariance feature utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np

_pyrecest_utils = importlib.import_module("pyrecest.utils")
_pyrecest_pairwise_mahalanobis_distances: Callable[..., Any] | None = getattr(
    _pyrecest_utils,
    "pairwise_mahalanobis_distances",
    None,
)
_pyrecest_pairwise_covariance_shape_components: Callable[..., Any] | None = getattr(
    _pyrecest_utils,
    "pairwise_covariance_shape_components",
    None,
)


def pairwise_mahalanobis_distances(
    means_a: Any,
    covariances_a: Any,
    means_b: Any,
    covariances_b: Any,
    *,
    regularization: float = 0.0,
) -> np.ndarray:
    """Return PyRecEst pairwise Mahalanobis distances or a local compatibility result."""

    if _pyrecest_pairwise_mahalanobis_distances is not None:
        return np.asarray(
            _pyrecest_pairwise_mahalanobis_distances(
                means_a,
                covariances_a,
                means_b,
                covariances_b,
                regularization=regularization,
            ),
            dtype=float,
        )
    return _fallback_pairwise_mahalanobis_distances(
        means_a,
        covariances_a,
        means_b,
        covariances_b,
        regularization=regularization,
    )


def pairwise_covariance_shape_components(
    covariances_a: Any,
    covariances_b: Any,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PyRecEst covariance-shape components or a local compatibility result."""

    if _pyrecest_pairwise_covariance_shape_components is not None:
        shape_cost, logdet_cost, shape_similarity = (
            _pyrecest_pairwise_covariance_shape_components(
                covariances_a,
                covariances_b,
                epsilon=epsilon,
            )
        )
        return (
            np.asarray(shape_cost, dtype=float),
            np.asarray(logdet_cost, dtype=float),
            np.asarray(shape_similarity, dtype=float),
        )
    return _fallback_pairwise_covariance_shape_components(
        covariances_a,
        covariances_b,
        epsilon=epsilon,
    )


def _validate_means_and_covariances(
    means: Any,
    covariances: Any,
    *,
    means_name: str,
    covariances_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2:
        raise ValueError(f"{means_name} must have shape (dim, n_items)")
    if covariances.ndim != 3 or covariances.shape[:2] != (
        means.shape[0],
        means.shape[0],
    ):
        raise ValueError(
            f"{covariances_name} must have shape (dim, dim, n_items) matching {means_name}"
        )
    if covariances.shape[2] != means.shape[1]:
        raise ValueError(
            f"{covariances_name} must contain one covariance matrix per mean"
        )
    if not np.all(np.isfinite(means)):
        raise ValueError(f"{means_name} must be finite")
    if not np.all(np.isfinite(covariances)):
        raise ValueError(f"{covariances_name} must be finite")
    return means, covariances


def _validate_covariance_stack(name: str, covariances: Any) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[0] != covariances.shape[1]:
        raise ValueError(f"{name} must have shape (dim, dim, n_items)")
    if not np.all(np.isfinite(covariances)):
        raise ValueError(f"{name} must be finite")
    return covariances


def _fallback_pairwise_mahalanobis_distances(
    means_a: Any,
    covariances_a: Any,
    means_b: Any,
    covariances_b: Any,
    *,
    regularization: float,
) -> np.ndarray:
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    means_a, covariances_a = _validate_means_and_covariances(
        means_a,
        covariances_a,
        means_name="means_a",
        covariances_name="covariances_a",
    )
    means_b, covariances_b = _validate_means_and_covariances(
        means_b,
        covariances_b,
        means_name="means_b",
        covariances_name="covariances_b",
    )
    if means_a.shape[0] != means_b.shape[0]:
        raise ValueError("means_a and means_b must have the same leading dimension")

    n_a = means_a.shape[1]
    n_b = means_b.shape[1]
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=float)

    identity = np.eye(means_a.shape[0], dtype=float)
    distances = np.zeros((n_a, n_b), dtype=float)
    for index_a in range(n_a):
        for index_b in range(n_b):
            difference = means_a[:, index_a] - means_b[:, index_b]
            covariance = covariances_a[:, :, index_a] + covariances_b[:, :, index_b]
            if regularization > 0.0:
                covariance = covariance + float(regularization) * identity
            normalized = np.linalg.pinv(covariance) @ difference
            squared_distance = float(difference @ normalized)
            distances[index_a, index_b] = np.sqrt(max(squared_distance, 0.0))
    return distances


def _fallback_pairwise_covariance_shape_components(
    covariances_a: Any,
    covariances_b: Any,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariances_a = _validate_covariance_stack("covariances_a", covariances_a)
    covariances_b = _validate_covariance_stack("covariances_b", covariances_b)
    if covariances_a.shape[0] != covariances_b.shape[0]:
        raise ValueError("covariance stacks must have the same matrix dimension")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    n_a = covariances_a.shape[2]
    n_b = covariances_b.shape[2]
    if n_a == 0 or n_b == 0:
        empty = np.zeros((n_a, n_b), dtype=float)
        return empty, empty.copy(), empty.copy()

    covariances_a = 0.5 * (covariances_a + np.swapaxes(covariances_a, 0, 1))
    covariances_b = 0.5 * (covariances_b + np.swapaxes(covariances_b, 0, 1))
    moved_a = np.moveaxis(covariances_a, -1, 0)
    moved_b = np.moveaxis(covariances_b, -1, 0)
    traces_a = np.maximum(np.trace(moved_a, axis1=1, axis2=2), epsilon)
    traces_b = np.maximum(np.trace(moved_b, axis1=1, axis2=2), epsilon)
    normalized_a = moved_a / traces_a[:, None, None]
    normalized_b = moved_b / traces_b[:, None, None]
    shape_differences = normalized_a[:, None, :, :] - normalized_b[None, :, :, :]
    shape_cost = np.linalg.norm(shape_differences, axis=(2, 3)) / np.sqrt(2.0)
    shape_similarity = np.exp(-shape_cost)
    determinants_a = np.maximum(np.linalg.det(moved_a), epsilon)
    determinants_b = np.maximum(np.linalg.det(moved_b), epsilon)
    logdet_cost = np.abs(np.log(determinants_a[:, None] / determinants_b[None, :]))
    return shape_cost, logdet_cost, shape_similarity


__all__ = [
    "pairwise_covariance_shape_components",
    "pairwise_mahalanobis_distances",
]
