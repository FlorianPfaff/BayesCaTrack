from __future__ import annotations

import csv

import numpy as np

from bayescatrack.evaluation.track2p_teacher_audit import (
    audit_track2p_teacher_edges,
    build_track_edge_index,
    write_teacher_audit_rows_csv,
    write_teacher_audit_summary_csv,
)


def test_build_track_edge_index_handles_missing_values_and_max_gap() -> None:
    matrix = np.asarray(
        [
            [1, 2, 3],
            [4, None, 6],
            [7, -1, np.nan],
        ],
        dtype=object,
    )

    all_edges = build_track_edge_index(matrix)
    consecutive_edges = build_track_edge_index(matrix, max_gap=1)

    assert len(all_edges.edges) == 4
    assert len(consecutive_edges.edges) == 2
    assert all(edge.gap == 1 for edge in consecutive_edges.edges)


def test_teacher_audit_emits_requested_debug_categories() -> None:
    manual_gt = np.asarray(
        [
            [1, 2, 3],
            [4, 5, -1],
        ],
        dtype=object,
    )
    track2p = np.asarray(
        [
            [1, 2, 3],
            [4, 99, -1],
        ],
        dtype=object,
    )
    bayes = np.asarray(
        [
            [1, 9, 3],
            [4, 5, -1],
        ],
        dtype=object,
    )

    result = audit_track2p_teacher_edges(
        manual_gt,
        track2p,
        bayes,
        subject="jm_test",
        session_names=("s0", "s1", "s2"),
        max_gap=1,
    )

    counts: dict[str, int] = {}
    for row in result.rows:
        counts[row.category] = counts.get(row.category, 0) + 1

    assert counts["GT+Track2p+Bayes-"] == 2
    assert counts["GT+Track2p-Bayes+"] == 1
    assert counts["GT-Track2p+Bayes-"] == 1
    assert counts["GT-Track2p-Bayes+"] == 2
    assert result.summary["gt_track2p_found_bayes_missed"] == 2
    assert result.summary["gt_track2p_missed_bayes_found"] == 1
    assert result.summary["track2p_f1"] == 2 / 3
    assert result.summary["bayes_f1"] == 1 / 3

    teacher_found_bayes_missed = [row for row in result.rows if row.category == "GT+Track2p+Bayes-"]
    assert teacher_found_bayes_missed[0].bayes_targets_for_source == (9,)


def test_teacher_audit_csv_writers(tmp_path) -> None:
    matrix = np.asarray([[1, 2]], dtype=object)
    result = audit_track2p_teacher_edges(matrix, matrix, matrix, subject="jm_test", session_names=("s0", "s1"))
    rows_path = tmp_path / "edges.csv"
    summary_path = tmp_path / "summary.csv"

    write_teacher_audit_rows_csv(result.rows, rows_path)
    write_teacher_audit_summary_csv([result.summary], summary_path)

    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))

    assert rows[0]["category"] == "GT+Track2p+Bayes+"
    assert summaries[0]["subject"] == "jm_test"
