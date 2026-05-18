"""Command-line entry point for Track2p teacher/debug-oracle edge audits.

This script operates on exported track matrices. Each matrix must have one track
per row and one session per column. Missing detections can be encoded as an
empty field, ``NaN``, ``None``, or a negative integer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from bayescatrack.evaluation.track2p_teacher_audit import (
    audit_track2p_teacher_edges,
    write_teacher_audit_rows_csv,
    write_teacher_audit_summary_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-gt", required=True, type=Path, help="Manual-GT track matrix CSV.")
    parser.add_argument("--track2p", required=True, type=Path, help="Track2p teacher track matrix CSV.")
    parser.add_argument("--bayes", required=True, type=Path, help="BayesCaTrack track matrix CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output edge-level audit CSV.")
    parser.add_argument("--summary-output", type=Path, help="Optional output summary CSV.")
    parser.add_argument("--subject", default="", help="Subject identifier copied into output rows.")
    parser.add_argument("--max-gap", type=int, help="Maximum audited session gap. Defaults to all gaps.")
    parser.add_argument(
        "--session-names",
        help="Comma-separated session names. Defaults to numeric session indices.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Emit only manual-GT edges, suppressing teacher/Bayes false-positive-only rows.",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Print the summary as JSON on stdout in addition to writing CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p teacher/debug-oracle audit."""

    args = build_arg_parser().parse_args(argv)
    session_names = _parse_session_names(args.session_names)
    result = audit_track2p_teacher_edges(
        _read_track_matrix(args.manual_gt),
        _read_track_matrix(args.track2p),
        _read_track_matrix(args.bayes),
        subject=args.subject,
        session_names=session_names,
        max_gap=args.max_gap,
        include_non_gt_edges=not args.gt_only,
    )
    write_teacher_audit_rows_csv(result.rows, args.output)
    if args.summary_output is not None:
        write_teacher_audit_summary_csv([result.summary], args.summary_output)
    if args.summary_json:
        print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def _read_track_matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError(f"empty track matrix: {path}")
    return np.asarray(rows, dtype=object)


def _parse_session_names(raw_names: str | None) -> tuple[str, ...] | None:
    if raw_names is None:
        return None
    names = tuple(name.strip() for name in raw_names.split(","))
    if any(name == "" for name in names):
        raise ValueError("--session-names must not contain empty names")
    return names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
