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


def aggregate_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, float | int | str]]:
    """Aggregate subject-level benchmark rows by approach."""

    labels = tuple(dict.fromkeys(row["approach"] for row in rows))
    return [_aggregate_approach(label, [row for row in rows if row["approach"] == label]) for label in labels]


def write_comparison(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    output_format: str,
    *,
    highlight_best: bool = False,
) -> None:
    """Write aggregate comparison rows as Markdown or CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_aggregate_columns())
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(
        format_markdown_table(rows, highlight_best=highlight_best) + "\n",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison-table CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    inputs = [_parse_input_spec(spec) for spec in args.input]
    rows = aggregate_rows(load_labeled_rows(inputs))
    if args.output is not None:
        write_comparison(rows, args.output, args.format, highlight_best=args.highlight_best)
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_aggregate_columns())
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_markdown_table(rows, highlight_best=args.highlight_best))
    return 0


def _parse_input_spec(spec: str) -> ComparisonInput:
    if "=" not in spec:
        path = Path(spec)
        return ComparisonInput(label=path.stem, path=path)
    label, path_text = spec.split("=", 1)
    if not label:
        raise ValueError("--input labels must not be empty")
    return ComparisonInput(label=label, path=Path(path_text))


def _aggregate_approach(label: str, rows: Sequence[dict[str, str]]) -> dict[str, float | int | str]:
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


def _compute_best_values(
    rows: Sequence[dict[str, float | int | str]],
) -> dict[str, float]:
    columns = {
        "pairwise_f1_macro",
        "pairwise_f1_micro",
        "complete_track_f1_macro",
        "complete_track_f1_micro",
    }
    if not rows:
        return {}
    return {column: max(_as_float(row[column]) for row in rows) for column in columns}


def _value_is_best(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-12


def _format_value(value: float | int | str, *, bold: bool = False) -> str:
    if isinstance(value, float):
        formatted = f"{value:.3f}"
    elif isinstance(value, int):
        formatted = str(value)
    else:
        formatted = str(value)
    return f"**{formatted}**" if bold else formatted


def _as_float(value: float | int | str) -> float:
    if isinstance(value, bool):
        raise ValueError("Benchmark metric values must be numeric")
    if isinstance(value, (float, int)):
        return float(value)
    return float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
