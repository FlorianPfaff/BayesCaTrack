from __future__ import annotations

import csv

import pytest
from bayescatrack.experiments.benchmark_comparison import (
    ComparisonInput,
    aggregate_rows,
    build_metric_rows,
    build_reference_gap_rows,
    build_subject_gap_summary_rows,
    build_subject_metric_rows,
    format_best_summary,
    format_markdown_table,
    format_reference_gap_summary,
    format_subject_gap_summary,
    load_labeled_rows,
    write_metric_csv,
    write_reference_gap_csv,
    write_subject_gap_summary,
    write_subject_metric_csv,
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


def _aggregate_summary_row(
    approach: str,
    *,
    pairwise_f1_macro: float,
    pairwise_f1_micro: float,
    complete_track_f1_macro: float,
    complete_track_f1_micro: float,
    subjects: int = 2,
    pairwise_f1_sd: float = 0.02,
    complete_track_f1_sd: float = 0.04,
) -> dict[str, float | int | str]:
    return {
        "approach": approach,
        "subjects": subjects,
        "pairwise_f1_macro": pairwise_f1_macro,
        "pairwise_f1_sd": pairwise_f1_sd,
        "pairwise_f1_micro": pairwise_f1_micro,
        "complete_track_f1_macro": complete_track_f1_macro,
        "complete_track_f1_sd": complete_track_f1_sd,
        "complete_track_f1_micro": complete_track_f1_micro,
    }


def _subject_result_row(
    approach: str,
    subject: str,
    *,
    pairwise_f1: str,
    complete_track_f1: str,
    pairwise_counts: tuple[int, int, int],
    complete_track_counts: tuple[int, int, int],
) -> dict[str, str]:
    pairwise_tp, pairwise_fp, pairwise_fn = pairwise_counts
    complete_tp, complete_fp, complete_fn = complete_track_counts
    return {
        "approach": approach,
        "subject": subject,
        "pairwise_f1": pairwise_f1,
        "complete_track_f1": complete_track_f1,
        "pairwise_true_positives": str(pairwise_tp),
        "pairwise_false_positives": str(pairwise_fp),
        "pairwise_false_negatives": str(pairwise_fn),
        "complete_track_true_positives": str(complete_tp),
        "complete_track_false_positives": str(complete_fp),
        "complete_track_false_negatives": str(complete_fn),
    }


def _subject_comparison_rows() -> list[dict[str, str]]:
    return [
        _subject_result_row(
            "Track2p",
            "jm039",
            pairwise_f1="0.90",
            complete_track_f1="0.80",
            pairwise_counts=(9, 1, 1),
            complete_track_counts=(8, 2, 2),
        ),
        _subject_result_row(
            "BayesCaTrack",
            "jm039",
            pairwise_f1="0.60",
            complete_track_f1="0.50",
            pairwise_counts=(6, 2, 6),
            complete_track_counts=(5, 3, 7),
        ),
        _subject_result_row(
            "Track2p",
            "jm046",
            pairwise_f1="0.70",
            complete_track_f1="0.40",
            pairwise_counts=(7, 3, 3),
            complete_track_counts=(4, 6, 6),
        ),
        _subject_result_row(
            "BayesCaTrack",
            "jm046",
            pairwise_f1="0.75",
            complete_track_f1="0.30",
            pairwise_counts=(8, 2, 3),
            complete_track_counts=(3, 7, 7),
        ),
    ]


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
    rows: list[dict[str, float | int | str]] = [
        _aggregate_summary_row(
            "Base",
            subjects=10,
            pairwise_f1_macro=0.60,
            pairwise_f1_sd=0.01,
            pairwise_f1_micro=0.50,
            complete_track_f1_macro=0.40,
            complete_track_f1_sd=0.03,
            complete_track_f1_micro=0.20,
        ),
        _aggregate_summary_row(
            "Bayes",
            subjects=10,
            pairwise_f1_macro=0.80,
            pairwise_f1_micro=0.90,
            complete_track_f1_macro=0.70,
            complete_track_f1_micro=0.88,
        ),
    ]

    table = format_markdown_table(rows, highlight_best=True)

    assert "**0.800**" in table
    assert "**0.900**" in table
    assert "**0.700**" in table
    assert "**0.880**" in table


def test_best_summary_names_best_approach_for_main_metrics():
    rows: list[dict[str, float | int | str]] = [
        _aggregate_summary_row(
            "Base",
            pairwise_f1_macro=0.60,
            pairwise_f1_sd=0.01,
            pairwise_f1_micro=0.50,
            complete_track_f1_macro=0.40,
            complete_track_f1_sd=0.03,
            complete_track_f1_micro=0.20,
        ),
        _aggregate_summary_row(
            "Tuned",
            pairwise_f1_macro=0.80,
            pairwise_f1_micro=0.90,
            complete_track_f1_macro=0.70,
            complete_track_f1_micro=0.88,
        ),
    ]

    summary = format_best_summary(rows)

    assert "### Best by Metric" in summary
    assert "| pairwise F1 mean | Tuned | 0.800 |" in summary
    assert "| complete-track F1 micro | Tuned | 0.880 |" in summary


def test_reference_gap_summary_reports_best_non_reference_gap():
    rows: list[dict[str, float | int | str]] = [
        _aggregate_summary_row(
            "Track2p",
            pairwise_f1_macro=0.95,
            pairwise_f1_sd=0.01,
            pairwise_f1_micro=0.96,
            complete_track_f1_macro=0.90,
            complete_track_f1_sd=0.03,
            complete_track_f1_micro=0.91,
        ),
        _aggregate_summary_row(
            "Global-IoU",
            pairwise_f1_macro=0.55,
            pairwise_f1_micro=0.56,
            complete_track_f1_macro=0.20,
            complete_track_f1_micro=0.22,
        ),
        _aggregate_summary_row(
            "Tuned",
            pairwise_f1_macro=0.70,
            pairwise_f1_micro=0.72,
            complete_track_f1_macro=0.60,
            complete_track_f1_micro=0.62,
        ),
    ]

    summary = format_reference_gap_summary(rows, reference_approach="Track2p")

    assert "### Gap to Track2p" in summary
    assert "| pairwise F1 mean | 0.950 | Tuned | 0.700 | -0.250 |" in summary
    assert "| complete-track F1 micro | 0.910 | Tuned | 0.620 | -0.290 |" in summary


def test_reference_gap_csv_is_machine_readable(tmp_path):
    rows: list[dict[str, float | int | str]] = [
        _aggregate_summary_row(
            "Track2p",
            pairwise_f1_macro=0.95,
            pairwise_f1_sd=0.01,
            pairwise_f1_micro=0.96,
            complete_track_f1_macro=0.90,
            complete_track_f1_sd=0.03,
            complete_track_f1_micro=0.91,
        ),
        _aggregate_summary_row(
            "Tuned",
            pairwise_f1_macro=0.70,
            pairwise_f1_micro=0.72,
            complete_track_f1_macro=0.60,
            complete_track_f1_micro=0.62,
        ),
    ]

    gap_rows = build_reference_gap_rows(rows, reference_approach="Track2p")
    output_path = tmp_path / "reference_gaps.csv"
    write_reference_gap_csv(rows, output_path, reference_approach="Track2p")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert gap_rows[0]["metric_column"] == "pairwise_f1_macro"
    assert gap_rows[0]["gap_to_reference"] == pytest.approx(-0.25)
    assert csv_rows[0]["metric"] == "pairwise F1 mean"
    assert csv_rows[0]["reference_approach"] == "Track2p"
    assert csv_rows[0]["best_non_reference_approach"] == "Tuned"
    assert float(csv_rows[0]["gap_to_reference"]) == pytest.approx(-0.25)


def test_metric_csv_reports_ranks_and_reference_gaps(tmp_path):
    rows: list[dict[str, float | int | str]] = [
        _aggregate_summary_row(
            "Track2p",
            pairwise_f1_macro=0.95,
            pairwise_f1_sd=0.01,
            pairwise_f1_micro=0.96,
            complete_track_f1_macro=0.90,
            complete_track_f1_sd=0.03,
            complete_track_f1_micro=0.91,
        ),
        _aggregate_summary_row(
            "Tuned-A",
            pairwise_f1_macro=0.80,
            pairwise_f1_micro=0.72,
            complete_track_f1_macro=0.60,
            complete_track_f1_micro=0.62,
        ),
        _aggregate_summary_row(
            "Tuned-B",
            pairwise_f1_macro=0.80,
            pairwise_f1_micro=0.70,
            complete_track_f1_macro=0.58,
            complete_track_f1_micro=0.60,
        ),
    ]

    metric_rows = build_metric_rows(rows, reference_approach="Track2p")
    output_path = tmp_path / "metrics.csv"
    write_metric_csv(rows, output_path, reference_approach="Track2p")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    pairwise_rows = [
        row for row in metric_rows if row["metric_column"] == "pairwise_f1_macro"
    ]
    tuned_rows = [
        row for row in pairwise_rows if str(row["approach"]).startswith("Tuned")
    ]
    assert len(metric_rows) == 12
    assert {row["rank"] for row in tuned_rows} == {2}
    assert all(row["gap_to_reference"] == pytest.approx(-0.15) for row in tuned_rows)
    assert csv_rows[0]["metric_column"] == "pairwise_f1_macro"
    assert csv_rows[0]["approach"] == "Track2p"
    assert csv_rows[0]["rank"] == "1"
    assert csv_rows[0]["is_best"] == "true"
    assert csv_rows[0]["is_reference"] == "true"


def test_subject_metric_csv_reports_per_subject_reference_gaps(tmp_path):
    rows = _subject_comparison_rows()

    subject_rows = build_subject_metric_rows(rows, reference_approach="Track2p")
    output_path = tmp_path / "subject_metrics.csv"
    write_subject_metric_csv(rows, output_path, reference_approach="Track2p")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    jm039_bayes_pairwise = next(
        row
        for row in subject_rows
        if row["subject"] == "jm039"
        and row["approach"] == "BayesCaTrack"
        and row["metric_column"] == "pairwise_f1"
    )
    jm046_bayes_pairwise = next(
        row
        for row in subject_rows
        if row["subject"] == "jm046"
        and row["approach"] == "BayesCaTrack"
        and row["metric_column"] == "pairwise_f1"
    )

    assert len(subject_rows) == 8
    assert jm039_bayes_pairwise["rank"] == 2
    assert jm039_bayes_pairwise["gap_to_reference"] == pytest.approx(-0.30)
    assert jm046_bayes_pairwise["rank"] == 1
    assert jm046_bayes_pairwise["gap_to_reference"] == pytest.approx(0.05)
    assert csv_rows[0]["subject"] == "jm039"
    assert csv_rows[0]["approach"] == "Track2p"
    assert csv_rows[0]["true_positives"] == "9"
    assert csv_rows[0]["is_reference"] == "true"


def test_subject_gap_summary_reports_worst_non_reference_rows(tmp_path):
    rows = _subject_comparison_rows()

    gap_rows = build_subject_gap_summary_rows(
        rows, reference_approach="Track2p", limit=12
    )
    summary = format_subject_gap_summary(rows, reference_approach="Track2p", limit=12)
    output_path = tmp_path / "subject_gaps.md"
    write_subject_gap_summary(
        rows,
        output_path,
        reference_approach="Track2p",
        limit=2,
    )

    assert [row["subject"] for row in gap_rows] == ["jm039", "jm039", "jm046"]
    assert [row["metric_column"] for row in gap_rows] == [
        "complete_track_f1",
        "pairwise_f1",
        "complete_track_f1",
    ]
    assert all(float(row["gap_to_reference"]) < 0.0 for row in gap_rows)
    assert "### Worst Subject Gaps to Track2p" in summary
    assert (
        "| jm039 | complete-track F1 | BayesCaTrack | 0.500 | 0.800 | -0.300 | 2 |"
        in summary
    )
    assert (
        "| jm046 | pairwise F1 | BayesCaTrack | 0.750 | 0.700 | +0.050 |" not in summary
    )
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_subject_gap_summary_reports_when_no_deficits():
    rows = [
        _subject_result_row(
            "Track2p",
            "jm039",
            pairwise_f1="0.60",
            complete_track_f1="0.50",
            pairwise_counts=(6, 2, 6),
            complete_track_counts=(5, 3, 7),
        ),
        _subject_result_row(
            "BayesCaTrack",
            "jm039",
            pairwise_f1="0.70",
            complete_track_f1="0.55",
            pairwise_counts=(7, 2, 5),
            complete_track_counts=(6, 3, 6),
        ),
    ]

    summary = format_subject_gap_summary(rows, reference_approach="Track2p")

    assert "no non-reference deficits" in summary
