"""Track-level error ledgers for longitudinal ROI identity matrices.

The canonical implementation is provided by
:mod:`pyrecest.utils.track_evaluation`.  This module keeps BayesCaTrack's
historic import path and ROI-specific ledger aliases.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pyrecest.utils.track_evaluation import (
    summarize_track_errors as _summarize_track_errors,
)
from pyrecest.utils.track_evaluation import track_error_ledger as _track_error_ledger

Observation = tuple[int, int]
Link = tuple[int, int, int, int]

__all__ = ("summarize_track_errors", "track_error_ledger")


def track_error_ledger(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Return detailed track-level errors between predicted and reference identities.

    PyRecEst returns generic observation-keyed ledgers. BayesCaTrack augments
    those rows with the historical ``roi_*`` aliases so existing Track2p/Suite2p
    reporting code can continue to consume the same fields.
    """
    ledger = _track_error_ledger(
        predicted_track_matrix,
        reference_track_matrix,
        session_pairs=session_pairs,
    )
    _add_roi_aliases(ledger)
    return ledger


def summarize_track_errors(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, int | float]:
    """Return aggregate track-level error metrics for benchmark CSV tables."""
    return _summarize_track_errors(
        predicted_track_matrix,
        reference_track_matrix,
        session_pairs=session_pairs,
    )


def _add_roi_aliases(ledger: dict[str, Any]) -> None:
    for row in ledger.get("link_errors", ()):
        if "observation_a" in row and "roi_a" not in row:
            row["roi_a"] = row["observation_a"]
        if "observation_b" in row and "roi_b" not in row:
            row["roi_b"] = row["observation_b"]

    for row in ledger.get("duplicate_observations", ()):
        if "observation" in row and "roi" not in row:
            row["roi"] = row["observation"]
