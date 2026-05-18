"""Audit manual-GT ROI index spaces for Track2p benchmarks.

The Track2p benchmark expects manual reference ROI IDs to be compatible with
loaded Suite2p ROI identifiers.  Public or manually curated data may instead
use raw ``stat.npy`` rows, filtered-cell ordinals, or another reindexed subset.
This module makes those cases explicit before they become opaque benchmark
failures, especially for subjects such as jm046.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    discover_subject_dirs,
)
from bayescatrack.reference import Track2pReference

OutputFormat = Literal["csv", "json", "markdown", "table"]


@dataclass(frozen=True)
class RoiIndexAuditRow:
    """One subject/session manual-GT ROI index-space diagnostic."""

    subject: str
    session: str
    session_index: int
    reference_source: str
    n_stat_rows: int | None
    n_loaded_rois: int
    n_loaded_cells: int
    n_loaded_rois_with_non_cells: int
    n_gt_rois: int
    max_gt_roi_index: int | None
    n_gt_rois_missing_from_loaded_indices: int
    n_gt_rois_missing_with_include_non_cells: int
    missing_gt_roi_examples: tuple[int, ...]
    include_non_cells_resolves_mismatch: bool
    gt_fits_loaded_indices: bool
    gt_fits_loaded_cell_indices: bool
    gt_fits_include_non_cells_indices: bool
    gt_fits_raw_stat_row_space: bool | None
    gt_fits_filtered_cell_ordinal_space: bool
    gt_index_space: str

    @property
    def compatible(self) -> bool:
        """Return whether all manual-GT IDs resolve in the current loaded ROI set."""

        return self.n_gt_rois_missing_from_loaded_indices == 0

    def to_dict(self) -> dict[str, int | str | bool | None]:
        """Return a stable CSV/JSON row."""

        return {
            "subject": self.subject,
            "session": self.session,
            "session_index": int(self.session_index),
            "reference_source": self.reference_source,
            "n_stat_rows": self.n_stat_rows,
            "n_loaded_rois": int(self.n_loaded_rois),
            "n_loaded_cells": int(self.n_loaded_cells),
            "n_loaded_rois_with_non_cells": int(self.n_loaded_rois_with_non_cells),
            "n_gt_rois": int(self.n_gt_rois),
            "max_gt_roi_index": self.max_gt_roi_index,
            "n_gt_rois_missing_from_loaded_indices": int(
                self.n_gt_rois_missing_from_loaded_indices
            ),
            "n_gt_rois_missing_with_include_non_cells": int(
                self.n_gt_rois_missing_with_include_non_cells
            ),
            "missing_gt_roi_examples": " ".join(
                str(value) for value in self.missing_gt_roi_examples
            ),
            "include_non_cells_resolves_mismatch": bool(
                self.include_non_cells_resolves_mismatch
            ),
            "gt_fits_loaded_indices": bool(self.gt_fits_loaded_indices),
            "gt_fits_loaded_cell_indices": bool(self.gt_fits_loaded_cell_indices),
            "gt_fits_include_non_cells_indices": bool(
                self.gt_fits_include_non_cells_indices
            ),
            "gt_fits_raw_stat_row_space": self.gt_fits_raw_stat_row_space,
            "gt_fits_filtered_cell_ordinal_space": bool(
                self.gt_fits_filtered_cell_ordinal_space
            ),
            "gt_index_space": self.gt_index_space,
            "compatible": bool(self.compatible),
        }


@dataclass(frozen=True)
class ManualGtRoiIndexAuditConfig:
    """Configuration for manual-GT ROI index-space auditing."""

    data: Path
    reference: Path | None = None
    reference_kind: ReferenceKind = "manual-gt"
    plane_name: str = "plane0"
    input_format: str = "auto"
    include_behavior: bool = False
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    missing_preview_limit: int = 20


@dataclass(frozen=True)
class ManualGtRoiIndexAuditResult:
    """All manual-GT ROI index-space diagnostics for one input root."""

    rows: tuple[RoiIndexAuditRow, ...]

    @property
    def compatible(self) -> bool:
        """Return whether every manual-GT ID resolves in the current loaded ROI set."""

        return bool(self.rows) and all(row.compatible for row in self.rows)

    @property
    def subjects(self) -> tuple[str, ...]:
        """Return subject names covered by the audit."""

        return tuple(sorted({row.subject for row in self.rows}))

    @property
    def incompatible_subjects(self) -> tuple[str, ...]:
        """Return subjects with at least one unresolved manual-GT ROI ID."""

        return tuple(sorted({row.subject for row in self.rows if not row.compatible}))

    def to_rows(self) -> list[dict[str, int | str | bool | None]]:
        """Return JSON/CSV-ready rows."""

        return [row.to_dict() for row in self.rows]


def run_manual_gt_roi_index_audit(
    config: ManualGtRoiIndexAuditConfig,
) -> ManualGtRoiIndexAuditResult:
    """Audit whether manual-GT ROI IDs match loaded Suite2p/Track2p index spaces."""

    benchmark_config = Track2pBenchmarkConfig(
        data=config.data,
        method="track2p-baseline",
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=config.reference,
        reference_kind=config.reference_kind,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
    )
    loaded_cell_config = replace(benchmark_config, include_non_cells=False)
    include_non_cell_config = replace(benchmark_config, include_non_cells=True)

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    rows: list[RoiIndexAuditRow] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=benchmark_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                f"Subject {subject_dir.name!r} resolved reference source {reference.source!r}, not independent manual ground truth. "
                "Pass --reference-kind manual-gt with a ground_truth.csv file/root."
            )

        loaded_sessions = tuple(_load_subject_sessions(subject_dir, benchmark_config))
        loaded_cell_sessions = tuple(
            _load_subject_sessions(subject_dir, loaded_cell_config)
        )
        include_non_cell_sessions = tuple(
            _load_subject_sessions(subject_dir, include_non_cell_config)
        )
        _validate_session_alignment(
            subject_dir.name,
            reference,
            loaded_sessions,
            loaded_cell_sessions,
            include_non_cell_sessions,
        )

        for session_index, session in enumerate(loaded_sessions):
            rows.append(
                _audit_row_for_session(
                    subject=subject_dir.name,
                    session=session,
                    loaded_cell_session=loaded_cell_sessions[session_index],
                    include_non_cell_session=include_non_cell_sessions[session_index],
                    session_index=session_index,
                    reference=reference,
                    missing_preview_limit=config.missing_preview_limit,
                )
            )

    return ManualGtRoiIndexAuditResult(rows=tuple(rows))


def write_audit_result(
    result: ManualGtRoiIndexAuditResult, output_path: Path, output_format: OutputFormat
) -> None:
    """Write audit diagnostics to ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(result.to_rows(), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(result.to_rows()))
            writer.writeheader()
            writer.writerows(result.to_rows())
        return
    output_path.write_text(format_audit_markdown(result) + "\n", encoding="utf-8")


def format_audit_markdown(result: ManualGtRoiIndexAuditResult) -> str:
    """Format audit diagnostics as Markdown."""

    lines = [
        "## Track2p manual-GT ROI index-space audit",
        "",
        f"- compatible: `{str(result.compatible).lower()}`",
        f"- subjects: `{','.join(result.subjects)}`",
        f"- incompatible_subjects: `{','.join(result.incompatible_subjects)}`",
        "",
        "| subject | session | stat rows | loaded ROIs | loaded cells | "
        "loaded with non-cells | GT ROIs | max GT index | missing | "
        "missing with non-cells | index space | compatible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.subject,
                    row.session,
                    _format_optional_int(row.n_stat_rows),
                    str(row.n_loaded_rois),
                    str(row.n_loaded_cells),
                    str(row.n_loaded_rois_with_non_cells),
                    str(row.n_gt_rois),
                    _format_optional_int(row.max_gt_roi_index),
                    str(row.n_gt_rois_missing_from_loaded_indices),
                    str(row.n_gt_rois_missing_with_include_non_cells),
                    row.gt_index_space,
                    str(row.compatible).lower(),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark audit-manual-gt-rois",
        description="Audit manual-GT ROI index spaces before Track2p benchmark runs.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p data root or one subject directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground_truth.csv file/root",
    )
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        help="Reference kind to resolve; manual-gt is recommended for paper-facing audits",
    )
    parser.add_argument(
        "--plane", dest="plane_name", default="plane0", help="Plane name such as plane0"
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format for loading sessions",
    )
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load behavior arrays when present",
    )
    parser.add_argument(
        "--include-non-cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use this ROI filtering policy for the compatibility column",
    )
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--missing-preview-limit", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output file"
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "table", "json", "csv"),
        default="markdown",
        help="Stdout/output format",
    )
    parser.add_argument(
        "--fail-on-incompatible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return exit code 1 when any manual-GT ROI IDs are missing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI."""

    args = build_arg_parser().parse_args(argv)
    result = run_manual_gt_roi_index_audit(
        ManualGtRoiIndexAuditConfig(
            data=args.data,
            reference=args.reference,
            reference_kind=args.reference_kind,
            plane_name=args.plane_name,
            input_format=args.input_format,
            include_behavior=args.include_behavior,
            include_non_cells=args.include_non_cells,
            cell_probability_threshold=args.cell_probability_threshold,
            weighted_masks=args.weighted_masks,
            exclude_overlapping_pixels=args.exclude_overlapping_pixels,
            missing_preview_limit=args.missing_preview_limit,
        )
    )
    if args.output is not None:
        write_audit_result(result, args.output, args.format)
    else:
        _write_stdout(result, args.format)
    return 1 if args.fail_on_incompatible and not result.compatible else 0


def _validate_session_alignment(
    subject: str,
    reference: Track2pReference,
    loaded_sessions: Sequence[Track2pSession],
    loaded_cell_sessions: Sequence[Track2pSession],
    include_non_cell_sessions: Sequence[Track2pSession],
) -> None:
    expected = reference.n_sessions
    for label, sessions in {
        "loaded": loaded_sessions,
        "loaded-cell": loaded_cell_sessions,
        "include-non-cell": include_non_cell_sessions,
    }.items():
        if len(sessions) != expected:
            raise ValueError(
                f"Subject {subject!r} has {len(sessions)} {label} sessions, but the reference has {expected} sessions"
            )
        session_names = tuple(session.session_name for session in sessions)
        if session_names != reference.session_names:
            raise ValueError(
                f"Subject {subject!r} {label} session order {session_names!r} does not match reference order {reference.session_names!r}"
            )


def _audit_row_for_session(
    *,
    subject: str,
    session: Track2pSession,
    loaded_cell_session: Track2pSession,
    include_non_cell_session: Track2pSession,
    session_index: int,
    reference: Track2pReference,
    missing_preview_limit: int,
) -> RoiIndexAuditRow:
    loaded_indices = _roi_index_set(session)
    loaded_cell_indices = _roi_index_set(loaded_cell_session)
    include_non_cell_indices = _roi_index_set(include_non_cell_session)
    gt_indices = _valid_reference_roi_values(
        reference.suite2p_indices[:, session_index]
    )
    missing_loaded = sorted(gt_indices - loaded_indices)
    missing_include_non_cells = sorted(gt_indices - include_non_cell_indices)
    n_stat_rows = _count_stat_rows(session)
    max_gt = max(gt_indices) if gt_indices else None
    gt_fits_loaded = gt_indices.issubset(loaded_indices)
    gt_fits_loaded_cells = gt_indices.issubset(loaded_cell_indices)
    gt_fits_include_non_cells = gt_indices.issubset(include_non_cell_indices)
    gt_fits_raw_stat = (
        None if n_stat_rows is None else _fits_ordinal_space(gt_indices, n_stat_rows)
    )
    gt_fits_filtered_cell_ordinals = _fits_ordinal_space(
        gt_indices, len(loaded_cell_indices)
    )
    gt_index_space = _classify_index_space(
        gt_indices=gt_indices,
        n_stat_rows=n_stat_rows,
        loaded_indices=loaded_indices,
        loaded_cell_indices=loaded_cell_indices,
        include_non_cell_indices=include_non_cell_indices,
    )

    return RoiIndexAuditRow(
        subject=subject,
        session=session.session_name,
        session_index=int(session_index),
        reference_source=reference.source,
        n_stat_rows=n_stat_rows,
        n_loaded_rois=len(loaded_indices),
        n_loaded_cells=len(loaded_cell_indices),
        n_loaded_rois_with_non_cells=len(include_non_cell_indices),
        n_gt_rois=len(gt_indices),
        max_gt_roi_index=max_gt,
        n_gt_rois_missing_from_loaded_indices=len(missing_loaded),
        n_gt_rois_missing_with_include_non_cells=len(missing_include_non_cells),
        missing_gt_roi_examples=tuple(missing_loaded[:missing_preview_limit]),
        include_non_cells_resolves_mismatch=bool(
            missing_loaded and not missing_include_non_cells
        ),
        gt_fits_loaded_indices=gt_fits_loaded,
        gt_fits_loaded_cell_indices=gt_fits_loaded_cells,
        gt_fits_include_non_cells_indices=gt_fits_include_non_cells,
        gt_fits_raw_stat_row_space=gt_fits_raw_stat,
        gt_fits_filtered_cell_ordinal_space=gt_fits_filtered_cell_ordinals,
        gt_index_space=gt_index_space,
    )


def _roi_index_set(session: Track2pSession) -> set[int]:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return set(range(session.plane_data.n_rois))
    return {int(value) for value in np.asarray(roi_indices, dtype=int).reshape(-1)}


def _valid_reference_roi_values(values: np.ndarray) -> set[int]:
    rois: set[int] = set()
    for value in np.asarray(values, dtype=object).reshape(-1):
        if value is None:
            continue
        try:
            roi = int(value)
        except (TypeError, ValueError):
            continue
        if roi >= 0:
            rois.add(roi)
    return rois


def _count_stat_rows(session: Track2pSession) -> int | None:
    plane_name = session.plane_data.plane_name or "plane0"
    stat_path = session.session_dir / "suite2p" / plane_name / "stat.npy"
    if not stat_path.exists():
        return None
    return int(np.load(stat_path, allow_pickle=True).shape[0])


def _fits_ordinal_space(indices: set[int], size: int | None) -> bool:
    if not indices:
        return True
    if size is None or size <= 0:
        return False
    return min(indices) >= 0 and max(indices) < size


def _classify_index_space(
    *,
    gt_indices: set[int],
    n_stat_rows: int | None,
    loaded_indices: set[int],
    loaded_cell_indices: set[int],
    include_non_cell_indices: set[int],
) -> str:
    if not gt_indices:
        return "no_gt_rois"
    gt_fits_loaded = gt_indices.issubset(loaded_indices)
    gt_fits_loaded_cells = gt_indices.issubset(loaded_cell_indices)
    gt_fits_include_non_cells = gt_indices.issubset(include_non_cell_indices)
    gt_fits_raw_stat_space = (
        False if n_stat_rows is None else _fits_ordinal_space(gt_indices, n_stat_rows)
    )
    gt_fits_filtered_cell_ordinals = _fits_ordinal_space(
        gt_indices, len(loaded_cell_indices)
    )

    if gt_fits_loaded and gt_fits_loaded_cells:
        return (
            "raw_stat_rows_loaded" if n_stat_rows is not None else "loaded_roi_indices"
        )
    if gt_fits_loaded and gt_fits_include_non_cells:
        if gt_fits_filtered_cell_ordinals:
            return "raw_stat_rows_or_filtered_cell_ordinals"
        return (
            "raw_stat_rows_loaded" if n_stat_rows is not None else "loaded_roi_indices"
        )
    if gt_fits_include_non_cells:
        if gt_fits_filtered_cell_ordinals:
            return "raw_stat_rows_or_filtered_cell_ordinals"
        return "raw_stat_rows_requires_include_non_cells"
    if gt_fits_filtered_cell_ordinals:
        return "filtered_cell_ordinal_indices_suspected"
    if gt_fits_raw_stat_space:
        return "raw_stat_row_space_but_not_loaded"
    if n_stat_rows is not None and max(gt_indices) >= n_stat_rows:
        return "outside_stat_row_space"
    return "unknown_or_reindexed"


def _fieldnames(rows: Sequence[dict[str, int | str | bool | None]]) -> list[str]:
    preferred = [
        "subject",
        "session",
        "session_index",
        "reference_source",
        "n_stat_rows",
        "n_loaded_rois",
        "n_loaded_cells",
        "n_loaded_rois_with_non_cells",
        "n_gt_rois",
        "max_gt_roi_index",
        "n_gt_rois_missing_from_loaded_indices",
        "n_gt_rois_missing_with_include_non_cells",
        "missing_gt_roi_examples",
        "include_non_cells_resolves_mismatch",
        "gt_fits_loaded_indices",
        "gt_fits_loaded_cell_indices",
        "gt_fits_include_non_cells_indices",
        "gt_fits_raw_stat_row_space",
        "gt_fits_filtered_cell_ordinal_space",
        "gt_index_space",
        "compatible",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + extra


def _write_stdout(
    result: ManualGtRoiIndexAuditResult, output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(result.to_rows(), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(result.to_rows()))
        writer.writeheader()
        writer.writerows(result.to_rows())
        return
    print(format_audit_markdown(result))


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
