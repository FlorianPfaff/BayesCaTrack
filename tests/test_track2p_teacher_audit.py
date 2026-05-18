from __future__ import annotations

import csv

import numpy as np
import pytest
from bayescatrack.experiments.track2p_teacher_audit import (
    TRACK2P_TEACHER_MISS_CATEGORY,
    audit_track_matrices,
    format_teacher_audit_table,
    teacher_training_rows,
    write_edge_rows,
)


def test_audit_track_matrices_reports_teacher_debug_buckets():
    result = audit_track_matrices(
        subject="jm001",
        session_names=("2024-05-01_a", "2024-05-02_a"),
        ground_truth_tracks=np.array([[0, 0], [1, 1]], dtype=object),
        track2p_tracks=np.array([[0, 0], [1, 1], [2, 2]], dtype=object),
        bayes_tracks=np.array([[0, 0], [1, 2]], dtype=object),
        restrict_to_reference_seed_rois=False,
    )

    summary = result.summary_rows[0]
    assert summary["ground_truth_edges"] == 2
    assert summary["track2p_edges"] == 3
    assert summary["bayes_edges"] == 2
    assert summary["edges_gt_track2p_bayes"] == 1
    assert summary["edges_gt_track2p_not_bayes"] == 1
    assert summary["edges_not_gt_track2p_not_bayes"] == 1
    assert summary["edges_not_gt_not_track2p_bayes"] == 1
    assert summary["track2p_vs_gt_precision"] == pytest.approx(2.0 / 3.0)
    assert summary["track2p_vs_gt_recall"] == pytest.approx(1.0)
    assert summary["bayes_vs_gt_precision"] == pytest.approx(0.5)
    assert summary["bayes_vs_gt_recall"] == pytest.approx(0.5)

    teacher_misses = [
        row
        for row in result.edge_rows
        if row["category"] == TRACK2P_TEACHER_MISS_CATEGORY
    ]
    assert len(teacher_misses) == 1
    assert teacher_misses[0]["roi_a"] == 1
    assert teacher_misses[0]["roi_b"] == 1
    assert teacher_misses[0]["in_ground_truth"] is True
    assert teacher_misses[0]["in_track2p"] is True
    assert teacher_misses[0]["in_bayes"] is False


def test_audit_pair_mode_max_gap_and_seed_filter():
    result = audit_track_matrices(
        subject="jm001",
        session_names=("s0", "s1", "s2"),
        ground_truth_tracks=np.array([[0, 0, 0], [5, 5, 5]], dtype=object),
        track2p_tracks=np.array([[0, 0, 0], [5, 5, 5], [9, 9, 9]], dtype=object),
        bayes_tracks=np.array([[0, 0, 0], [9, 9, 9]], dtype=object),
        pair_mode="max-gap",
        max_gap=1,
        restrict_to_reference_seed_rois=True,
    )

    summary = result.summary_rows[0]
    assert summary["reference_seed_rois"] == 2
    assert summary["ground_truth_edges"] == 4
    assert summary["track2p_edges"] == 4
    assert summary["bayes_edges"] == 2
    assert all(row["gap"] == 1 for row in result.edge_rows)
    assert all(row["roi_a"] != 9 for row in result.edge_rows)


def test_teacher_training_rows_and_csv_output(tmp_path):
    result = audit_track_matrices(
        subject="jm001",
        session_names=("s0", "s1"),
        ground_truth_tracks=[[0, 0]],
        track2p_tracks=[[0, 0], [1, 1]],
        bayes_tracks=[[0, 0], [1, 2]],
        restrict_to_reference_seed_rois=False,
    )

    teacher_rows = teacher_training_rows(result.edge_rows)
    by_edge = {(row["roi_a"], row["roi_b"]): row for row in teacher_rows}
    assert by_edge[(0, 0)]["teacher_label"] == 1
    assert by_edge[(0, 0)]["manual_gt_label"] == 1
    assert by_edge[(1, 1)]["teacher_label"] == 1
    assert by_edge[(1, 1)]["manual_gt_label"] == 0
    assert by_edge[(1, 2)]["teacher_label"] == 0
    assert by_edge[(1, 2)]["bayes_label"] == 1

    output_path = tmp_path / "teacher_edges.csv"
    write_edge_rows(teacher_rows, output_path)
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["teacher_label"] for row in rows} == {"0", "1"}
    assert "teacher_label_source" in rows[0]


def test_format_teacher_audit_table_contains_key_debug_metric():
    result = audit_track_matrices(
        subject="jm001",
        session_names=("s0", "s1"),
        ground_truth_tracks=[[0, 0]],
        track2p_tracks=[[0, 0]],
        bayes_tracks=[[0, 1]],
    )

    table = format_teacher_audit_table(result.summary_rows)
    assert "edges_gt_track2p_not_bayes" in table
    assert "jm001" in table
