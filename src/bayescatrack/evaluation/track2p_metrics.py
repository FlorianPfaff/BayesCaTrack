"""Track2p benchmark metric facade."""

# pylint: disable=wildcard-import,unused-wildcard-import

from __future__ import annotations

from typing import Any

import numpy as np

from bayescatrack.reference import Track2pReference

from .calibration_metrics import brier_score
from .complete_track_scores import *  # noqa: F401,F403
from .complete_track_scores import __all__ as _complete_track_score_exports
from .complete_track_scores import normalize_track_matrix, score_track_matrices

__all__ = [
    *_complete_track_score_exports,
    "brier_score",
    "score_track_matrix_against_reference",
]


def score_track_matrix_against_reference(
    predicted_track_matrix: Any,
    reference: Track2pReference,
    *,
    curated_only: bool = False,
) -> dict[str, float | int]:
    """Score a predicted Suite2p-index track matrix against a Track2p reference."""

    reference_matrix = normalize_track_matrix(reference.suite2p_indices)
    if curated_only:
        if reference.curated_mask is None:
            raise ValueError("curated_only=True requires a Track2p reference with a curated_mask")
        reference_matrix = reference_matrix[np.asarray(reference.curated_mask, dtype=bool)]
    return score_track_matrices(predicted_track_matrix, reference_matrix)
