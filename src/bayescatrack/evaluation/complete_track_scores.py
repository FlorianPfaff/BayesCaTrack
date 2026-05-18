"""Track-level scoring helpers for longitudinal ROI identity matrices.

The implementation is provided by :mod:`pyrecest.utils.track_evaluation`.
BayesCaTrack keeps this module as a compatibility import path for benchmark
code and downstream users.
"""

from __future__ import annotations

from pyrecest.utils.track_evaluation import (
    complete_track_set,
    normalize_track_matrix,
    pairwise_track_set,
    reference_fragment_counts,
    score_complete_tracks,
    score_false_continuations,
    score_fragmentation,
    score_pairwise_tracks,
    score_track_matrices,
    summarize_tracks,
    track_lengths,
)

__all__ = (
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
    "score_pairwise_tracks",
    "score_track_matrices",
    "summarize_tracks",
    "track_lengths",
)
