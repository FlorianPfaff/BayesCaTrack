from __future__ import annotations

import csv

import pytest
from bayescatrack.experiments.benchmark_comparison import (
    ComparisonInput,
    aggregate_rows,
    format_markdown_table,
    load_labeled_rows,
)


def _write_result_csv(path, rows):
    fieldnames = [
        "subject",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_true_positives",
        "pairwise_false_positives",
        "pairwise_false_negatives",
        "complete_track_true_positives",
        "complete_track_false_positives",
        "complete_track_false_negatives",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_rows_reports_macro_and_micro_f1(tmp_path):
    result_path = tmp_path / "approach.csv"
    _write_result_csv(
        result_path,
        [
            {
                "subject": "jm001",
                "pairwise_f1": "1.0",
                "complete_track_f1": "0.5",
                "pairwise_true_positives": "10",
                "pairwise_false_positives": "0",
                "pairwise_false_negatives": "0",
                "complete_track_true_positives": "1",
                "complete_track_false_positives": "1",
                "complete_track_false_negatives": "1",
            },
            {
                "subject": "jm002",
                "pairwise_f1": "0.5",
                "complete_track_f1": "1.0",
                "pairwise_true_positives": "1",
                "pairwise_false_positives": "1",
                "pairwise_false_negatives": "1",
                "complete_track_true_positives": "10",
                "complete_track_false_positives": "0",
                "complete_track_false_negatives": "0",
            },
        ],
    )

    rows = aggregate_rows(
        load_labeled_rows([ComparisonInput("test approach", result_path)])
    )

    assert rows[0]["approach"] == "test approach"
    assert rows[0]["subjects"] == 2
    assert rows[0]["pairwise_f1_macro"] == pytest.approx(0.75)
    assert rows[0]["pairwise_f1_micro"] == pytest.approx(22 / 24)
    assert rows[0]["complete_track_f1_macro"] == pytest.approx(0.75)
    assert rows[0]["complete_track_f1_micro"] == pytest.approx(22 / 24)
    assert "test approach" in format_markdown_table(rows)


def test_markdown_table_highlights_best_cells():
    rows = [
        {
            "approach": "Base",
            "subjects": 10,
            "pairwise_f1_macro": 0.60,
            "pairwise_f1_sd": 0.01,
            "pairwise_f1_micro": 0.50,
            "complete_track_f1_macro": 0.40,
            "complete_track_f1_sd": 0.03,
            "complete_track_f1_micro": 0.20,
        },
        {
            "approach": "Bayes",
            "subjects": 10,
            "pairwise_f1_macro": 0.80,
            "pairwise_f1_sd": 0.02,
            "pairwise_f1_micro": 0.90,
            "complete_track_f1_macro": 0.70,
            "complete_track_f1_sd": 0.04,
            "complete_track_f1_micro": 0.88,
        },
    ]

    table = format_markdown_table(rows, highlight_best=True)

    assert "**0.800**" in table
    assert "**0.900**" in table
    assert "**0.700**" in table
    assert "**0.880**" in table
