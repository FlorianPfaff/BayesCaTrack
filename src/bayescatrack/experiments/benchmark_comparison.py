"""Compare benchmark result CSV files across tracking approaches."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class ComparisonInput:
    """One labeled benchmark result CSV."""

    label: str
    path: Path


def load_labeled_rows(inputs: Sequence[ComparisonInput]) -> list[dict[str, str]]:
    """Load benchmark rows and attach an ``approach`` label."""

    rows: list[dict[str, str]] = []
    for benchmark_input in inputs:
        with benchmark_input.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({"approach": benchmark_input.label, **row})
    if not rows:
        raise ValueError("No benchmark rows were loaded")
    return rows


def aggregate_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate subject-level benchmark rows by approach."""

    labels = tuple(dict.fromkeys(row["approach"] for row in rows))
    return [
        _aggregate_approach(label, [row for row in rows if row["approach"] == label])
        for label in labels
    ]


def write_comparison(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    output_format: str,
    *,
    highlight_best: bool = False,
    include_best_summary: bool = False,
    include_reference_gap_summary: bool = False,
    reference_approach: str | None = None,
) -> None:
    """Write aggregate comparison rows as Markdown or CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_aggregate_columns())
            writer.writeheader()
            writer.writerows(rows)
        return
    sections = []
    if include_best_summary:
        sections.append(format_best_summary(rows))
    if include_reference_gap_summary:
        sections.append(
            format_reference_gap_summary(rows, reference_approach=reference_approach)
        )
    sections.append(format_markdown_table(rows, highlight_best=highlight_best))
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def write_reference_gap_csv(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    *,
    reference_approach: str | None,
) -> None:
    """Write machine-readable best non-reference gaps to a CSV artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_reference_gap_columns())
        writer.writeheader()
        writer.writerows(
            build_reference_gap_rows(rows, reference_approach=reference_approach)
        )


def write_metric_csv(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    *,
    reference_approach: str | None,
) -> None:
    """Write long-format metric values, ranks, and reference gaps to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_metric_columns())
        writer.writeheader()
        writer.writerows(build_metric_rows(rows, reference_approach=reference_approach))


def write_subject_metric_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    reference_approach: str | None,
) -> None:
    """Write per-subject metric values, ranks, and reference gaps to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_subject_metric_columns())
        writer.writeheader()
        writer.writerows(
            build_subject_metric_rows(rows, reference_approach=reference_approach)
        )


def write_subject_gap_summary(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    reference_approach: str | None,
    limit: int,
) -> None:
    """Write a Markdown summary of the largest subject-level reference gaps."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_subject_gap_summary(
            rows, reference_approach=reference_approach, limit=limit
        )
        + "\n",
        encoding="utf-8",
    )


def write_subject_deficit_summary(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    reference_approach: str | None,
    limit: int,
) -> None:
    """Write a Markdown summary of the largest grouped subject deficits."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_subject_deficit_summary(
            rows, reference_approach=reference_approach, limit=limit
        )
        + "\n",
        encoding="utf-8",
    )


def format_markdown_table(
    rows: Sequence[dict[str, float | int | str]],
    *,
    highlight_best: bool = False,
) -> str:
    """Format aggregate rows as a compact Markdown comparison table."""

    columns = _aggregate_columns()
    best_by_metric = _compute_best_values(rows) if highlight_best else {}
    headers = {
        "approach": "approach",
        "subjects": "n",
        "pairwise_f1_macro": "pairwise F1 mean",
        "pairwise_f1_sd": "pairwise F1 sd",
        "pairwise_f1_micro": "pairwise F1 micro",
        "complete_track_f1_macro": "complete-track F1 mean",
        "complete_track_f1_sd": "complete-track F1 sd",
        "complete_track_f1_micro": "complete-track F1 micro",
    }
    header = "| " + " | ".join(headers[column] for column in columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                _format_value(
                    row[column],
                    bold=highlight_best
                    and column in best_by_metric
                    and _value_is_best(_as_float(row[column]), best_by_metric[column]),
                )
                for column in columns
            )
            + " |"
        )
    return "\n".join(body)


def format_best_summary(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Format a short Markdown summary naming the best approach per metric."""

    best_rows = _compute_best_rows(rows)
    body = [
        "### Best by Metric",
        "",
        "| metric | approach | value |",
        "| --- | --- | ---: |",
    ]
    for column in _best_metric_columns():
        approaches, value = best_rows[column]
        body.append(
            f"| {_best_metric_headers()[column]} | {', '.join(approaches)} | {_format_value(value)} |"
        )
    return "\n".join(body)


def format_reference_gap_summary(
    rows: Sequence[dict[str, float | int | str]],
    *,
    reference_approach: str | None,
) -> str:
    """Format a Markdown table comparing the best non-reference row to a baseline."""

    gap_rows = build_reference_gap_rows(rows, reference_approach=reference_approach)
    reference_name = str(gap_rows[0]["reference_approach"])
    body = [
        f"### Gap to {reference_name}",
        "",
        (
            "| metric | "
            f"{reference_name} | best non-reference approach | best non-reference value | gap |"
        ),
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in gap_rows:
        body.append(
            "| "
            f"{row['metric']} | "
            f"{_format_value(_as_float(row['reference_value']))} | "
            f"{row['best_non_reference_approach']} | "
            f"{_format_value(_as_float(row['best_non_reference_value']))} | "
            f"{_format_delta(_as_float(row['gap_to_reference']))} |"
        )
    return "\n".join(body)


def format_subject_gap_summary(
    rows: Sequence[dict[str, str]],
    *,
    reference_approach: str | None,
    limit: int = 12,
) -> str:
    """Format a Markdown table of the largest subject-level reference gaps."""

    reference_name = _reference_approach_name(rows, reference_approach)
    gap_rows = build_subject_gap_summary_rows(
        rows, reference_approach=reference_approach, limit=limit
    )
    body = [
        f"### Worst Subject Gaps to {reference_name}",
        "",
        "| subject | metric | approach | value | reference | gap | rank |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if not gap_rows:
        body.append("| n/a | no non-reference deficits | n/a | n/a | n/a | n/a | n/a |")
        return "\n".join(body)

    for row in gap_rows:
        body.append(
            "| "
            f"{row['subject']} | "
            f"{row['metric']} | "
            f"{row['approach']} | "
            f"{_format_value(_as_float(row['value']))} | "
            f"{_format_value(_as_float(row['reference_value']))} | "
            f"{_format_delta(_as_float(row['gap_to_reference']))} | "
            f"{row['rank']} |"
        )
    return "\n".join(body)


def format_subject_deficit_summary(
    rows: Sequence[dict[str, str]],
    *,
    reference_approach: str | None,
    limit: int = 12,
) -> str:
    """Format grouped subject-level deficits against the reference approach."""

    reference_name = _reference_approach_name(rows, reference_approach)
    deficit_rows = build_subject_deficit_summary_rows(
        rows, reference_approach=reference_approach, limit=limit
    )
    body = [
        f"### Worst Subjects by Total Deficit to {reference_name}",
        "",
        (
            "| subject | approach | deficit metrics | total deficit | "
            "mean deficit | worst metric | worst gap |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    if not deficit_rows:
        body.append(
            "| n/a | no non-reference deficits | 0 | 0.000 | 0.000 | n/a | 0.000 |"
        )
        return "\n".join(body)

    for row in deficit_rows:
        body.append(
            "| "
            f"{row['subject']} | "
            f"{row['approach']} | "
            f"{row['deficit_metrics']} / {row['metrics_compared']} | "
            f"{_format_delta(_as_float(row['total_deficit']))} | "
            f"{_format_delta(_as_float(row['mean_deficit']))} | "
            f"{row['worst_metric']} | "
            f"{_format_delta(_as_float(row['worst_gap']))} |"
        )
    return "\n".join(body)


def build_reference_gap_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    reference_approach: str | None,
) -> list[dict[str, float | str]]:
    """Build structured rows comparing best competitors to a reference approach."""

    reference = _reference_row(rows, reference_approach=reference_approach)
    competitors = tuple(row for row in rows if row is not reference)
    if not competitors:
        raise ValueError("At least one non-reference approach is required")

    reference_name = str(reference["approach"])
    gap_rows: list[dict[str, float | str]] = []
    for column in _best_metric_columns():
        reference_value = _as_float(reference[column])
        competitor_names, competitor_value = _best_competitor_rows(competitors, column)
        gap_rows.append(
            {
                "metric": _best_metric_headers()[column],
                "metric_column": column,
                "reference_approach": reference_name,
                "reference_value": reference_value,
                "best_non_reference_approach": ", ".join(competitor_names),
                "best_non_reference_value": competitor_value,
                "gap_to_reference": competitor_value - reference_value,
            }
        )
    return gap_rows


def build_metric_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    reference_approach: str | None,
) -> list[dict[str, float | int | str]]:
    """Build long-format metric rows with ranks and reference deltas."""

    if not rows:
        raise ValueError("At least one aggregate row is required")

    reference = _reference_row(rows, reference_approach=reference_approach)
    reference_name = str(reference["approach"])
    metric_rows: list[dict[str, float | int | str]] = []
    for column in _best_metric_columns():
        reference_value = _as_float(reference[column])
        best_value = max(_as_float(row[column]) for row in rows)
        ranks = _descending_competition_ranks([_as_float(row[column]) for row in rows])
        for row in rows:
            value = _as_float(row[column])
            approach = str(row["approach"])
            metric_rows.append(
                {
                    "metric": _best_metric_headers()[column],
                    "metric_column": column,
                    "approach": approach,
                    "subjects": int(row["subjects"]),
                    "value": value,
                    "rank": ranks[value],
                    "is_best": _format_bool(_value_is_best(value, best_value)),
                    "reference_approach": reference_name,
                    "reference_value": reference_value,
                    "gap_to_reference": value - reference_value,
                    "is_reference": _format_bool(approach == reference_name),
                }
            )
    return metric_rows


def build_subject_metric_rows(
    rows: Sequence[dict[str, str]],
    *,
    reference_approach: str | None,
) -> list[dict[str, float | int | str]]:
    """Build long-format subject-level metric rows with ranks and reference deltas."""

    if not rows:
        raise ValueError("At least one subject-level row is required")

    reference_name = _reference_approach_name(rows, reference_approach)
    result_rows: list[dict[str, float | int | str]] = []
    for subject in dict.fromkeys(row.get("subject", "") for row in rows):
        subject_rows = [row for row in rows if row.get("subject", "") == subject]
        for metric_column, metric_name, count_prefix in _subject_metric_specs():
            values_by_row = [
                (row, _as_float(row[metric_column]))
                for row in subject_rows
                if row.get(metric_column, "") != ""
            ]
            if not values_by_row:
                continue
            ranks = _descending_competition_ranks([value for _, value in values_by_row])
            reference_value = _subject_reference_value(
                values_by_row,
                reference_name=reference_name,
            )
            for row, value in values_by_row:
                result_rows.append(
                    {
                        "subject": subject,
                        "metric": metric_name,
                        "metric_column": metric_column,
                        "approach": row["approach"],
                        "value": value,
                        "rank": ranks[value],
                        "true_positives": _int_value(
                            row, f"{count_prefix}_true_positives"
                        ),
                        "false_positives": _int_value(
                            row, f"{count_prefix}_false_positives"
                        ),
                        "false_negatives": _int_value(
                            row, f"{count_prefix}_false_negatives"
                        ),
                        "reference_approach": reference_name,
                        "reference_value": (
                            reference_value if reference_value is not None else ""
                        ),
                        "gap_to_reference": (
                            value - reference_value
                            if reference_value is not None
                            else ""
                        ),
                        "is_reference": _format_bool(row["approach"] == reference_name),
                    }
                )
    return result_rows


def build_subject_gap_summary_rows(
    rows: Sequence[dict[str, str]],
    *,
    reference_approach: str | None,
    limit: int = 12,
) -> list[dict[str, float | int | str]]:
    """Build subject-level non-reference deficits sorted by worst reference gap."""

    if limit < 1:
        raise ValueError("Subject gap summary limit must be at least 1")

    subject_metric_rows = build_subject_metric_rows(
        rows, reference_approach=reference_approach
    )
    gap_rows = [
        row
        for row in subject_metric_rows
        if row["is_reference"] != "true"
        and row["gap_to_reference"] != ""
        and _as_float(row["gap_to_reference"]) < 0.0
    ]
    return sorted(
        gap_rows,
        key=lambda row: (
            _as_float(row["gap_to_reference"]),
            str(row["subject"]),
            str(row["metric"]),
            int(row["rank"]),
            str(row["approach"]),
        ),
    )[:limit]


def build_subject_deficit_summary_rows(
    rows: Sequence[dict[str, str]],
    *,
    reference_approach: str | None,
    limit: int = 12,
) -> list[dict[str, float | int | str]]:
    """Build per-subject/per-approach deficits grouped across subject metrics."""

    if limit < 1:
        raise ValueError("Subject deficit summary limit must be at least 1")

    summaries: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for row in build_subject_metric_rows(rows, reference_approach=reference_approach):
        if row["is_reference"] == "true" or row["gap_to_reference"] == "":
            continue
        key = (str(row["subject"]), str(row["approach"]))
        summary = summaries.setdefault(
            key,
            {
                "subject": str(row["subject"]),
                "approach": str(row["approach"]),
                "metrics_compared": 0,
                "deficit_metrics": 0,
                "total_deficit": 0.0,
                "worst_metric": "",
                "worst_gap": 0.0,
            },
        )
        summary["metrics_compared"] = int(summary["metrics_compared"]) + 1
        gap = _as_float(row["gap_to_reference"])
        if gap < 0.0:
            summary["deficit_metrics"] = int(summary["deficit_metrics"]) + 1
            summary["total_deficit"] = _as_float(summary["total_deficit"]) + gap
            if summary["worst_metric"] == "" or gap < _as_float(summary["worst_gap"]):
                summary["worst_metric"] = str(row["metric"])
                summary["worst_gap"] = gap

    deficit_rows = [
        {
            **summary,
            "mean_deficit": _as_float(summary["total_deficit"])
            / int(summary["deficit_metrics"]),
        }
        for summary in summaries.values()
        if int(summary["deficit_metrics"]) > 0
    ]
    return sorted(
        deficit_rows,
        key=lambda row: (
            _as_float(row["total_deficit"]),
            _as_float(row["worst_gap"]),
            str(row["subject"]),
            str(row["approach"]),
        ),
    )[:limit]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the comparison-table CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark compare",
        description="Aggregate Track2p benchmark CSV files into a comparison table.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Labeled benchmark CSV to include; repeat for each approach",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output table path",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format",
    )
    parser.add_argument(
        "--highlight-best",
        action="store_true",
        help="Highlight best metric values in Markdown output",
    )
    parser.add_argument(
        "--include-best-summary",
        action="store_true",
        help="Prepend a Markdown summary naming the best approach for each metric",
    )
    parser.add_argument(
        "--include-reference-gap-summary",
        action="store_true",
        help="Prepend a Markdown summary of the best non-reference approach gap",
    )
    parser.add_argument(
        "--reference-approach",
        default=None,
        help="Approach label used as the reference baseline in gap summaries",
    )
    parser.add_argument(
        "--reference-gap-output",
        type=Path,
        default=None,
        help="Optional CSV path for best non-reference gaps to the reference approach",
    )
    parser.add_argument(
        "--metric-output",
        type=Path,
        default=None,
        help="Optional long-format CSV path for metric ranks and reference gaps",
    )
    parser.add_argument(
        "--subject-metric-output",
        type=Path,
        default=None,
        help="Optional long-format CSV path for subject-level reference gaps",
    )
    parser.add_argument(
        "--subject-gap-summary-output",
        type=Path,
        default=None,
        help="Optional Markdown path for the worst subject-level reference gaps",
    )
    parser.add_argument(
        "--subject-gap-summary-limit",
        type=int,
        default=12,
        help="Maximum number of rows to include in the subject gap summary",
    )
    parser.add_argument(
        "--subject-deficit-summary-output",
        type=Path,
        default=None,
        help="Optional Markdown path for grouped subject-level deficits",
    )
    parser.add_argument(
        "--subject-deficit-summary-limit",
        type=int,
        default=12,
        help="Maximum number of rows to include in the subject deficit summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison-table CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    inputs = [_parse_input_spec(spec) for spec in args.input]
    subject_rows = load_labeled_rows(inputs)
    rows = aggregate_rows(subject_rows)
    if args.reference_gap_output is not None:
        write_reference_gap_csv(
            rows,
            args.reference_gap_output,
            reference_approach=args.reference_approach,
        )
    if args.metric_output is not None:
        write_metric_csv(
            rows,
            args.metric_output,
            reference_approach=args.reference_approach,
        )
    if args.subject_metric_output is not None:
        write_subject_metric_csv(
            subject_rows,
            args.subject_metric_output,
            reference_approach=args.reference_approach,
        )
    if args.subject_gap_summary_output is not None:
        write_subject_gap_summary(
            subject_rows,
            args.subject_gap_summary_output,
            reference_approach=args.reference_approach,
            limit=args.subject_gap_summary_limit,
        )
    if args.subject_deficit_summary_output is not None:
        write_subject_deficit_summary(
            subject_rows,
            args.subject_deficit_summary_output,
            reference_approach=args.reference_approach,
            limit=args.subject_deficit_summary_limit,
        )
    if args.output is not None:
        write_comparison(
            rows,
            args.output,
            args.format,
            highlight_best=args.highlight_best,
            include_best_summary=args.include_best_summary,
            include_reference_gap_summary=args.include_reference_gap_summary,
            reference_approach=args.reference_approach,
        )
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_aggregate_columns())
        writer.writeheader()
        writer.writerows(rows)
    else:
        sections = []
        if args.include_best_summary:
            sections.append(format_best_summary(rows))
        if args.include_reference_gap_summary:
            sections.append(
                format_reference_gap_summary(
                    rows, reference_approach=args.reference_approach
                )
            )
        sections.append(format_markdown_table(rows, highlight_best=args.highlight_best))
        print("\n\n".join(sections))
    return 0


def _parse_input_spec(spec: str) -> ComparisonInput:
    if "=" not in spec:
        path = Path(spec)
        return ComparisonInput(label=path.stem, path=path)
    label, path_text = spec.split("=", 1)
    if not label:
        raise ValueError("--input labels must not be empty")
    return ComparisonInput(label=label, path=Path(path_text))


def _aggregate_approach(
    label: str, rows: Sequence[dict[str, str]]
) -> dict[str, float | int | str]:
    return {
        "approach": label,
        "subjects": len(rows),
        "pairwise_f1_macro": _mean(rows, "pairwise_f1"),
        "pairwise_f1_sd": _stdev(rows, "pairwise_f1"),
        "pairwise_f1_micro": _micro_f1(rows, "pairwise"),
        "complete_track_f1_macro": _mean(rows, "complete_track_f1"),
        "complete_track_f1_sd": _stdev(rows, "complete_track_f1"),
        "complete_track_f1_micro": _micro_f1(rows, "complete_track"),
    }


def _mean(rows: Sequence[dict[str, str]], key: str) -> float:
    return float(mean(_float_values(rows, key)))


def _stdev(rows: Sequence[dict[str, str]], key: str) -> float:
    values = _float_values(rows, key)
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def _micro_f1(rows: Sequence[dict[str, str]], prefix: str) -> float:
    true_positives = sum(_int_value(row, f"{prefix}_true_positives") for row in rows)
    false_positives = sum(_int_value(row, f"{prefix}_false_positives") for row in rows)
    false_negatives = sum(_int_value(row, f"{prefix}_false_negatives") for row in rows)
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return float(2 * true_positives / denominator)


def _float_values(rows: Sequence[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _int_value(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        return 0
    return int(float(value))


def _aggregate_columns() -> list[str]:
    return [
        "approach",
        "subjects",
        "pairwise_f1_macro",
        "pairwise_f1_sd",
        "pairwise_f1_micro",
        "complete_track_f1_macro",
        "complete_track_f1_sd",
        "complete_track_f1_micro",
    ]


def _reference_gap_columns() -> list[str]:
    return [
        "metric",
        "metric_column",
        "reference_approach",
        "reference_value",
        "best_non_reference_approach",
        "best_non_reference_value",
        "gap_to_reference",
    ]


def _metric_columns() -> list[str]:
    return [
        "metric",
        "metric_column",
        "approach",
        "subjects",
        "value",
        "rank",
        "is_best",
        "reference_approach",
        "reference_value",
        "gap_to_reference",
        "is_reference",
    ]


def _subject_metric_columns() -> list[str]:
    return [
        "subject",
        "metric",
        "metric_column",
        "approach",
        "value",
        "rank",
        "true_positives",
        "false_positives",
        "false_negatives",
        "reference_approach",
        "reference_value",
        "gap_to_reference",
        "is_reference",
    ]


def _best_metric_columns() -> tuple[str, ...]:
    return (
        "pairwise_f1_macro",
        "pairwise_f1_micro",
        "complete_track_f1_macro",
        "complete_track_f1_micro",
    )


def _best_metric_headers() -> dict[str, str]:
    return {
        "pairwise_f1_macro": "pairwise F1 mean",
        "pairwise_f1_micro": "pairwise F1 micro",
        "complete_track_f1_macro": "complete-track F1 mean",
        "complete_track_f1_micro": "complete-track F1 micro",
    }


def _subject_metric_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("pairwise_f1", "pairwise F1", "pairwise"),
        ("complete_track_f1", "complete-track F1", "complete_track"),
    )


def _compute_best_values(
    rows: Sequence[dict[str, float | int | str]],
) -> dict[str, float]:
    if not rows:
        return {}
    return {
        column: max(_as_float(row[column]) for row in rows)
        for column in _best_metric_columns()
    }


def _compute_best_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> dict[str, tuple[tuple[str, ...], float]]:
    if not rows:
        raise ValueError("At least one aggregate row is required")
    best_values = _compute_best_values(rows)
    return {
        column: (
            tuple(
                str(row["approach"])
                for row in rows
                if _value_is_best(_as_float(row[column]), best_values[column])
            ),
            best_values[column],
        )
        for column in _best_metric_columns()
    }


def _reference_row(
    rows: Sequence[dict[str, float | int | str]],
    *,
    reference_approach: str | None,
) -> dict[str, float | int | str]:
    if not rows:
        raise ValueError("At least one aggregate row is required")
    if reference_approach is None:
        return rows[0]
    matches = [row for row in rows if str(row["approach"]) == reference_approach]
    if not matches:
        available = ", ".join(str(row["approach"]) for row in rows)
        raise ValueError(
            f"Reference approach {reference_approach!r} not found; available: {available}"
        )
    return matches[0]


def _reference_approach_name(
    rows: Sequence[dict[str, str]],
    reference_approach: str | None,
) -> str:
    if reference_approach is None:
        return rows[0]["approach"]
    if not any(row["approach"] == reference_approach for row in rows):
        available = ", ".join(dict.fromkeys(row["approach"] for row in rows))
        raise ValueError(
            f"Reference approach {reference_approach!r} not found; available: {available}"
        )
    return reference_approach


def _subject_reference_value(
    values_by_row: Sequence[tuple[dict[str, str], float]],
    *,
    reference_name: str,
) -> float | None:
    for row, value in values_by_row:
        if row["approach"] == reference_name:
            return value
    return None


def _best_competitor_rows(
    rows: Sequence[dict[str, float | int | str]], column: str
) -> tuple[tuple[str, ...], float]:
    value = max(_as_float(row[column]) for row in rows)
    return (
        tuple(
            str(row["approach"])
            for row in rows
            if _value_is_best(_as_float(row[column]), value)
        ),
        value,
    )


def _value_is_best(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-12


def _descending_competition_ranks(values: Sequence[float]) -> dict[float, int]:
    unique_values = sorted(set(values), reverse=True)
    return {
        value: sum(1 for candidate in values if candidate > value) + 1
        for value in unique_values
    }


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_value(value: float | int | str, *, bold: bool = False) -> str:
    if isinstance(value, float):
        formatted = f"{value:.3f}"
    elif isinstance(value, int):
        formatted = str(value)
    else:
        formatted = str(value)
    return f"**{formatted}**" if bold else formatted


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


def _as_float(value: float | int | str) -> float:
    if isinstance(value, bool):
        raise ValueError("Benchmark metric values must be numeric")
    if isinstance(value, (float, int)):
        return float(value)
    return float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
