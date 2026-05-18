"""Weak additive activity tie-breaker costs for ROI association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def activity_tie_breaker_cost_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    component_name: str = "activity_tiebreaker_cost",
    weight: float = 0.05,
) -> np.ndarray:
    """Return a low-weight activity cost plane for additive tie-breaking.

    The selected component is intentionally scaled outside the calibrated model so
    spatial evidence remains dominant. Missing or undefined activity values are
    mapped to the neutral cost used by the activity feature extractor rather than
    to a perfect zero-cost match.
    """

    weight = float(weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Pairwise component {component_name!r} must be two-dimensional")
    return weight * np.nan_to_num(values, nan=0.5, posinf=1.0e6, neginf=0.0)
