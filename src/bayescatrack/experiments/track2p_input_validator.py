"""Validate Track2p manual-GT inputs before running paper-facing benchmarks.

The public Track2p data can contain a reduced or reindexed Suite2p ROI subset
while ``ground_truth.csv`` refers to original Suite2p ``stat.npy`` row indices.
This module turns that failure mode into an explicit preflight check that can be
run as soon as full pre-Track2p Suite2p folders arrive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _loaded_suite2p_index_set,
    discover_subject_dirs,
)

OutputFormat = Literal["csv", "json", "markdown", "table"]


@dataclass(frozen=True)
class RoiCoverageRow:
    """One subject/session ROI-index compatibility diagnostic."""

    subject: str
    session: str
    session_index: int
    reference_source: str
    referenced_rois: int
    loaded_rois: int
    missing_rois: int
    compatible: bool
    referenced_min: int | None
    referenced_max: int | None
    loaded_min: int | None
    loaded_max: int | None
    missing_preview: tuple[int, ...]
    missing_minus_one_present: int
    missing_plus_one_present: int
    index_space_hint: str

    def to_dict(self) -> dict[str, int | str | bool | None]:
        """Return a stable CSV/JSON row."""

        return {
            "subject": self.subject,
            "session": self.session,
            "session_index": int(self.session_index),
            "reference_source": self.reference_source,
            "referenced_rois": int(self.referenced_rois),
            "loaded_rois": int(self.loaded_rois),
            "missing_rois": int(self.missing_rois),
            "compatible": bool(self.compatible),
            "referenced_min": self.referenced_min,
            "referenced_max": self.referenced_max,
            "loaded_min": self.loaded_min,
            "loaded_max": self.loaded_max,
            "missing_preview": " ".join(str(value) for value in self.missing_preview),
            "missing_minus_one_present": int(self.missing_minus_one_present),
            "missing_plus_one_present": int(self.missing_plus_one_present),
            "index_space_hint": self.index_space_hint,
        }


@dataclass(frozen=True)
class Track2pInputValidationConfig:
    """Configuration for manual-GT ROI-coverage validation."""

    data: Path
    reference: Path | None = None
    reference_kind: ReferenceKind = "manual-gt"
    plane_name: str = "plane0"
    input_format: str = "auto"
    include_behavior: bool = False
    include_non_cells: bool = True
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    missing_preview_limit: int = 20


@dataclass(frozen=True)
class Track2pInputValidationResult:
    """All ROI-coverage diagnostics for a Track2p input root."""

    rows: tuple[RoiCoverageRow, ...]

    @property
    def compatible(self) -> bool:
        """Return whether all loaded subject/session pairs are compatible."""

        return bool(self.rows) and all(row.compatible for row in self.rows)

    @property
    def subjects(self) -> tuple[str, ...]:
        """Return subject names covered by the validation run."""

        return tuple(sorted({row.subject for row in self.rows}))

    @property
    def incompatible_subjects(self) -> tuple[str, ...]:
        """Return subjects with at least one incompatible session."""

        return tuple(sorted({row.subject for row in self.rows if not row.compatible}))

    def to_rows(self) -> list[dict[str, int | str | bool | None]]:
        """Return JSON/CSV-ready rows."""

        return [row.to_dict() for row in self.rows]


def run_track2p_input_validation(
    config: Track2pInputValidationConfig,
) -> Track2pInputValidationResult:
    """Validate that manual-GT ROI IDs resolve in loaded Suite2p sessions."""

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
    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {config.data}")

    rows: list[RoiCoverageRow] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=benchmark_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                f"Subject {subject_dir.name!r} resolved reference source "
                f"{reference.source!r}, not independent manual ground truth. "
                "Pass --reference-kind manual-gt with a ground_truth.csv file/root."
            )
        sessions = tuple(_load_subject_sessions(subject_dir, benchmark_config))
        if len(sessions) != reference.n_sessions:
            raise ValueError(
                f"Subject {subject_dir.name!r} has {len(sessions)} loaded sessions, "
                f"but the reference has {reference.n_sessions} sessions"
            )
        session_names = tuple(session.session_name for session in sessions)
        if session_names != reference.session_names:
            raise ValueError(
                f"Subject {subject_dir.name!r} loaded session order {session_names!r} "
                f"does not match reference order {reference.session_names!r}"
            )

        for session_index, session in enumerate(sessions):
            rows.append(
                _coverage_row_for_session(
                    subject=subject_dir.name,
                    session=session,
                    session_index=session_index,
                    reference_source=reference.source,
                    reference_values=reference.suite2p_indices[:, session_index],
                    missing_preview_limit=config.missing_preview_limit,
                )
            )

    return Track2pInputValidationResult(rows=tuple(rows))


def write_validation_result(
    result: Track2pInputValidationResult,
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write validation diagnostics to ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(result.to_rows()))
            writer.writeheader()
            writer.writerows(result.to_rows())
        return
    if output_format == "json":
        output_path.write_text(
            json.dumps(result.to_rows(), indent=2) + "\n", encoding="utf-8"
        )
        return
    output_path.write_text(format_validation_markdown(result) + "\n", encoding="utf-8")


def format_validation_markdown(result: Track2pInputValidationResult) -> str:
    """Format validation diagnostics as Markdown."""

    lines = [
        "## Track2p manual-GT ROI coverage validation",
        "",
        f"- compatible: `{str(result.compatible).lower()}`",
        f"- subjects: `{','.join(result.subjects)}`",
        f"- incompatible_subjects: `{','.join(result.incompatible_subjects)}`",
        "",
        "| subject | session | referenced ROIs | loaded ROIs | missing ROIs | compatible | hint |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.subject,
                    row.session,
                    str(row.referenced_rois),
                    str(row.loaded_rois),
                    str(row.missing_rois),
                    str(row.compatible).lower(),
                    row.index_space_hint,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark validate-track2p-inputs",
        description=(
            "Validate that Track2p manual ground-truth ROI IDs resolve in the "
            "loaded Suite2p/Track2p input sessions."
        ),
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="Track2p data root or one subject directory"
    )
    parser.add_argument("--reference", type=Path, default=None, help="Optional ground_truth.csv file/root")
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        help="Reference kind to resolve; manual-gt is recommended for paper-facing validation",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0", help="Plane name such as plane0")
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
        default=True,
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--missing-preview-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Optional output file")
    parser.add_argument(
        "--format",
        choices=("markdown", "table", "json", "csv"),
        default="markdown",
        help="Stdout/output format",
    )
    parser.add_argument(
        "--fail-on-incompatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return exit code 1 when any manual-GT ROI IDs are missing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator CLI."""

    args = build_arg_parser().parse_args(argv)
    result = run_track2p_input_validation(
        Track2pInputValidationConfig(
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
        write_validation_result(result, args.output, args.format)
    else:
        _write_stdout(result, args.format)
    return 1 if args.fail_on_incompatible and not result.compatible else 0


def _coverage_row_for_session(
    *,
    subject: str,
    session: object,
    session_index: int,
    reference_source: str,
    reference_values: np.ndarray,
    missing_preview_limit: int,
) -> RoiCoverageRow:
    available = sorted(_loaded_suite2p_index_set(session))
    available_set = set(available)
    referenced = sorted(_valid_reference_roi_values(reference_values))
    missing = sorted(set(referenced) - available_set)
    minus_one_matches = sum(1 for value in missing if value - 1 in available_set)
    plus_one_matches = sum(1 for value in missing if value + 1 in available_set)
    hint = _index_space_hint(
        referenced=referenced,
        available=available,
        missing=missing,
        minus_one_matches=minus_one_matches,
        plus_one_matches=plus_one_matches,
    )
    return RoiCoverageRow(
        subject=subject,
        session=session.session_name,
        session_index=int(session_index),
        reference_source=reference_source,
        referenced_rois=len(referenced),
        loaded_rois=len(available),
        missing_rois=len(missing),
        compatible=not missing,
        referenced_min=min(referenced) if referenced else None,
        referenced_max=max(referenced) if referenced else None,
        loaded_min=min(available) if available else None,
        loaded_max=max(available) if available else None,
        missing_preview=tuple(missing[:missing_preview_limit]),
        missing_minus_one_present=minus_one_matches,
        missing_plus_one_present=plus_one_matches,
        index_space_hint=hint,
    )


def _valid_reference_roi_values(values: np.ndarray) -> set[int]:
    """Return non-negative integer ROI IDs from a reference column."""

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


def _index_space_hint(
    *,
    referenced: Sequence[int],
    available: Sequence[int],
    missing: Sequence[int],
    minus_one_matches: int,
    plus_one_matches: int,
) -> str:
    if not missing:
        return "compatible"
    if missing and minus_one_matches == len(missing):
        return "possible_one_based_reference"
    if missing and plus_one_matches == len(missing):
        return "possible_minus_one_reference"
    if available and referenced and max(referenced) > max(available):
        return "loaded_roi_subset_or_reindexed_public_data"
    return "missing_reference_rois"


def _fieldnames(rows: Sequence[dict[str, int | str | bool | None]]) -> list[str]:
    preferred = [
        "subject",
        "session",
        "session_index",
        "reference_source",
        "referenced_rois",
        "loaded_rois",
        "missing_rois",
        "compatible",
        "referenced_min",
        "referenced_max",
        "loaded_min",
        "loaded_max",
        "missing_preview",
        "missing_minus_one_present",
        "missing_plus_one_present",
        "index_space_hint",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + extra


def _write_stdout(result: Track2pInputValidationResult, output_format: OutputFormat) -> None:
    if output_format == "json":
        print(json.dumps(result.to_rows(), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(result.to_rows()))
        writer.writeheader()
        writer.writerows(result.to_rows())
        return
    print(format_validation_markdown(result))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
