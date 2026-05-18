"""Tests for Track2p teacher/debug disagreement diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from bayescatrack.experiments.track2p_teacher_debug import (
    _BayesEdgeCostLookup,
    _edge_pairs,
    _summary_rows,
    _track_matrix_edge_set,
    classify_teacher_edge,
)


def test_classify_teacher_edge_covers_all_disagreement_buckets() -> None:
    assert classify_teacher_edge(True, True, True) == "all_agree_manual_positive"
    assert classify_teacher_edge(True, True, False) == "bayes_missed_teacher_edge"
    assert classify_teacher_edge(True, False, True) == "bayes_found_track2p_missed_edge"
    assert classify_teacher_edge(True, False, False) == "both_missed_manual_edge"
    assert classify_teacher_edge(False, True, False) == "track2p_false_positive_bayes_rejected"
    assert classify_teacher_edge(False, False, True) == "bayes_hard_false_positive"
    assert classify_teacher_edge(False, True, True) == "teacher_and_bayes_false_positive"
    assert classify_teacher_edge(False, False, False) == "unobserved_true_negative"


def test_edge_pairs_support_solver_consecutive_and_all_pairs() -> None:
    assert _edge_pairs(4, max_gap=2, scope="solver") == (
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    assert _edge_pairs(4, max_gap=2, scope="consecutive") == ((0, 1), (1, 2), (2, 3))
    assert _edge_pairs(4, max_gap=2, scope="all-pairs") == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_track_matrix_edge_set_uses_requested_edges_and_skips_missing_rois() -> None:
    matrix = np.asarray(
        [
            [1, 2, 3],
            [4, -1, 6],
            [None, 8, 9],
        ],
        dtype=object,
    )

    assert _track_matrix_edge_set(matrix, edges=((0, 1), (1, 2))) == {
        (0, 1, 1, 2),
        (1, 2, 2, 3),
        (1, 2, 8, 9),
    }


def test_bayes_cost_lookup_reports_rank_threshold_and_missing_reason() -> None:
    sessions = [
        SimpleNamespace(plane_data=SimpleNamespace(roi_indices=np.asarray([10, 20]), n_rois=2)),
        SimpleNamespace(plane_data=SimpleNamespace(roi_indices=np.asarray([100, 200, 300]), n_rois=3)),
    ]
    lookup = _BayesEdgeCostLookup.from_sessions(
        cast(list[Any], sessions),
        {(0, 1): np.asarray([[3.0, 1.0, 5.0], [0.5, 2.0, 4.0]])},
        gap_penalty=0.0,
        cost_threshold=1.5,
    )

    row = lookup.lookup(0, 1, 10, 200)
    assert row["bayes_candidate_present"] is True
    assert row["bayes_eligible_after_threshold"] is True
    assert row["bayes_cost"] == 1.0
    assert row["bayes_adjusted_cost"] == 1.0
    assert row["bayes_row_rank"] == 1
    assert row["bayes_col_rank"] == 1
    assert row["bayes_mutual_top1"] is True
    assert row["bayes_best_target_suite2p"] == 200
    assert row["bayes_margin_to_best"] == 0.0

    missing = lookup.lookup(0, 1, 999, 200)
    assert missing["bayes_candidate_present"] is False
    assert missing["bayes_missing_reason"] == "roi_a_not_loaded"


def test_summary_rows_aggregate_dataset_subject_and_session_edge_levels() -> None:
    detail_rows = [
        {
            "subject": "jm038",
            "session_a": 0,
            "session_b": 1,
            "session_a_name": "s0",
            "session_b_name": "s1",
            "gap": 1,
            "category": "bayes_missed_teacher_edge",
            "manual_gt_label": True,
            "track2p_label": True,
            "bayescatrack_label": False,
            "bayes_candidate_present": True,
            "bayes_eligible_after_threshold": False,
            "bayes_row_rank": 3,
            "bayes_adjusted_cost": 2.5,
        },
        {
            "subject": "jm038",
            "session_a": 0,
            "session_b": 1,
            "session_a_name": "s0",
            "session_b_name": "s1",
            "gap": 1,
            "category": "bayes_missed_teacher_edge",
            "manual_gt_label": True,
            "track2p_label": True,
            "bayescatrack_label": False,
            "bayes_candidate_present": False,
            "bayes_eligible_after_threshold": False,
            "bayes_row_rank": None,
            "bayes_adjusted_cost": None,
        },
    ]

    summary = _summary_rows(
        cast(list[dict[str, str | int | float | bool | None]], detail_rows)
    )
    dataset_row = next(row for row in summary if row["scope"] == "dataset")
    assert dataset_row["count"] == 2
    assert dataset_row["candidate_present_rate"] == 0.5
    assert dataset_row["eligible_after_threshold_rate"] == 0.0
    assert dataset_row["median_bayes_row_rank"] == 3.0
    assert dataset_row["median_bayes_adjusted_cost"] == 2.5
    assert {row["scope"] for row in summary} == {"dataset", "subject", "session_edge"}
