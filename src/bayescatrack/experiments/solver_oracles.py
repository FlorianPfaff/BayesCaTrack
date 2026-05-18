"""Solver-oracle pairwise costs for Track2p benchmark diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    build_registered_pairwise_costs,
    registered_iou_cost_kwargs,
    roi_aware_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.core.bridge import Track2pSession, build_session_pair_association_bundle
from bayescatrack.experiments.track2p_benchmark import _is_valid_roi_index


def solve_from_pairwise_costs(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray],
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
) -> GlobalAssignmentRun:
    """Run the normal PyRecEst global solver on externally supplied costs."""
    from bayescatrack.association import pyrecest_global_assignment as ga

    sessions = list(sessions)
    sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    edges = session_edge_pairs(len(sessions), max_gap=max_gap)
    costs = {edge: np.asarray(pairwise_costs[edge], dtype=float) for edge in edges}
    for source, target in edges:
        expected = (sizes[source], sizes[target])
        if costs[(source, target)].shape != expected:
            raise ValueError(f"Cost matrix {(source, target)!r} has wrong shape")
    result = ga._load_pyrecest_multisession_solver()(  # pylint: disable=protected-access
        costs,
        session_sizes=sizes,
        start_cost=float(start_cost),
        end_cost=float(end_cost),
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )
    return GlobalAssignmentRun(result=result, pairwise_costs=costs, session_sizes=sizes, session_edges=edges)


def oracle_edge_costs(
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    *,
    max_gap: int = 2,
    large_cost: float = 1.0e6,
) -> dict[tuple[int, int], np.ndarray]:
    """Set GT edges to zero cost and all other admissible edges to large_cost."""
    sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    costs: dict[tuple[int, int], np.ndarray] = {}
    for source, target in session_edge_pairs(len(sessions), max_gap=max_gap):
        matrix = np.full((sizes[source], sizes[target]), float(large_cost), dtype=float)
        for row, col in manual_gt_local_links(sessions, reference_matrix, source, target):
            matrix[row, col] = 0.0
        costs[(source, target)] = matrix
    return costs


def oracle_rank_k_costs(
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    *,
    rank_k: int,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    transform_type: str = "affine",
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    large_cost: float = 1.0e6,
) -> dict[tuple[int, int], np.ndarray]:
    """Keep a GT edge only when its row rank under the base cost is <= rank_k."""
    if rank_k < 1:
        raise ValueError("rank_k must be at least 1")
    base_costs = build_registered_pairwise_costs(
        sessions,
        max_gap=max_gap,
        cost=cost,
        transform_type=transform_type,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    costs: dict[tuple[int, int], np.ndarray] = {}
    for source, target in session_edge_pairs(len(sessions), max_gap=max_gap):
        base = np.asarray(base_costs[(source, target)], dtype=float)
        matrix = np.full(base.shape, float(large_cost), dtype=float)
        for row, col in manual_gt_local_links(sessions, reference_matrix, source, target):
            if row_rank_with_pessimistic_ties(base, row, col) <= rank_k:
                matrix[row, col] = 0.0
        costs[(source, target)] = matrix
    return costs


def oracle_registration_costs(
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    large_cost: float = 1.0e6,
    min_fit_links: int = 3,
    require_full_rank: bool = True,
    ridge: float = 0.0,
) -> dict[tuple[int, int], np.ndarray]:
    """Fit a manual-GT affine warp per edge, then compute normal pairwise costs."""
    if cost == "registered-iou":
        kwargs = registered_iou_cost_kwargs()
    elif cost == "roi-aware":
        kwargs = roi_aware_cost_kwargs()
    else:
        raise ValueError("oracle-registration supports registered-iou or roi-aware")
    if pairwise_cost_kwargs:
        kwargs.update(dict(pairwise_cost_kwargs))
    kwargs.setdefault("large_cost", float(large_cost))
    costs: dict[tuple[int, int], np.ndarray] = {}
    for source, target in session_edge_pairs(len(sessions), max_gap=max_gap):
        registered = oracle_affine_registered_plane(
            sessions[source], sessions[target], reference_matrix, source, target,
            min_fit_links=min_fit_links, require_full_rank=require_full_rank, ridge=ridge,
        )
        registered, empty = replace_empty_registered_masks(registered)
        bundle = build_session_pair_association_bundle(
            sessions[source], sessions[target], measurement_plane_in_reference_frame=registered,
            order=order, weighted_centroids=weighted_centroids,
            velocity_variance=velocity_variance, regularization=regularization,
            pairwise_cost_kwargs=kwargs, return_pairwise_components=False,
        )
        matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float).copy()
        matrix[:, empty] = float(kwargs.get("large_cost", large_cost))
        costs[(source, target)] = matrix
    return costs


def manual_gt_local_links(
    sessions: Sequence[Track2pSession], reference_matrix: np.ndarray, source: int, target: int
) -> list[tuple[int, int]]:
    """Return manual-GT links in loaded local ROI coordinates."""
    source_lookup = loaded_roi_index_lookup(sessions[source])
    target_lookup = loaded_roi_index_lookup(sessions[target])
    links: list[tuple[int, int]] = []
    for track in reference_matrix:
        source_roi, target_roi = track[source], track[target]
        if not (_is_valid_roi_index(source_roi) and _is_valid_roi_index(target_roi)):
            continue
        source_idx, target_idx = int(cast(Any, source_roi)), int(cast(Any, target_roi))
        if source_idx in source_lookup and target_idx in target_lookup:
            links.append((source_lookup[source_idx], target_lookup[target_idx]))
    return links


def loaded_roi_index_lookup(session: Track2pSession) -> dict[int, int]:
    roi_indices = session.plane_data.roi_indices
    suite2p_indices = np.asarray(roi_indices, dtype=int) if roi_indices is not None else np.arange(session.plane_data.n_rois, dtype=int)
    lookup: dict[int, int] = {}
    for local, suite2p_index in enumerate(suite2p_indices):
        key = int(suite2p_index)
        if key in lookup:
            raise ValueError(f"Duplicate Suite2p ROI index {key} in session {session.session_name!r}")
        lookup[key] = int(local)
    return lookup


def row_rank_with_pessimistic_ties(cost_matrix: np.ndarray, row_index: int, col_index: int) -> float:
    row = np.asarray(cost_matrix[row_index], dtype=float)
    true_cost = float(row[col_index])
    if not np.isfinite(true_cost):
        return float("inf")
    finite = row[np.isfinite(row)]
    return float(np.count_nonzero(finite <= true_cost))


def oracle_affine_registered_plane(
    source_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    *,
    min_fit_links: int = 3,
    require_full_rank: bool = True,
    ridge: float = 0.0,
) -> Any:
    from bayescatrack.experiments.oracle_affine_registration_qa import (
        _fit_affine_xy,
        _manual_gt_links,
        _oracle_affine_registered_plane,
    )
    links = _manual_gt_links(source_session.plane_data, target_session.plane_data, reference_matrix, source_index, target_index)
    if len(links) < min_fit_links:
        raise ValueError(f"Not enough manual-GT links for oracle affine registration: got {len(links)}, need {min_fit_links}")
    fit = _fit_affine_xy(
        np.vstack([link[2] for link in links]),
        np.vstack([link[3] for link in links]),
        ridge=ridge,
        require_full_rank=require_full_rank,
    )
    return _oracle_affine_registered_plane(source_session.plane_data, target_session.plane_data, fit)
