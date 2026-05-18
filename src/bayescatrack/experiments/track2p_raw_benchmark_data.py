"""Prepare raw Suite2p data for Track2p manual-ground-truth benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bayescatrack.core.bridge import find_track2p_session_dirs
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_CSV_NAME,
    Track2pBenchmarkConfig,
    _load_ground_truth_csv_reference,
    _load_subject_sessions,
    _loaded_suite2p_index_set,
)
from bayescatrack.reference import Track2pReference, load_track2p_reference


@dataclass(frozen=True)
class RawBenchmarkDiagnostic:
    """ROI index coverage diagnostic for one subject/session/reference source."""

    subject: str
    source: str
    session: str
    session_index: int
    referenced_rois: int
    loaded_rois: int
    missing_rois: int
    compatible: bool
    referenced_max: int | None = None
    loaded_max: int | None = None
    missing_preview: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "subject": self.subject,
            "source": self.source,
            "session": self.session,
            "session_index": self.session_index,
            "referenced_rois": self.referenced_rois,
            "loaded_rois": self.loaded_rois,
            "missing_rois": self.missing_rois,
            "compatible": str(self.compatible).lower(),
            "referenced_max": (
                "" if self.referenced_max is None else self.referenced_max
            ),
            "loaded_max": "" if self.loaded_max is None else self.loaded_max,
            "missing_preview": " ".join(str(value) for value in self.missing_preview),
        }


@dataclass(frozen=True)
class RawBenchmarkPreparation:
    """Summary of a prepared raw Suite2p benchmark tree."""

    output_root: Path
    plane_name: str = "plane0"
    included: tuple[str, ...] = ()
    excluded_by_user: tuple[str, ...] = ()
    excluded_no_raw_suite2p: tuple[str, ...] = ()
    excluded_no_ground_truth: tuple[str, ...] = ()
    excluded_no_track2p_suite2p_indices: tuple[str, ...] = ()
    excluded_incompatible: tuple[str, ...] = ()
    filtered_manual_gt_tracks: tuple[str, ...] = ()
    diagnostics: tuple[RawBenchmarkDiagnostic, ...] = ()

    @property
    def has_usable_subjects(self) -> bool:
        return bool(self.included)

    def to_outputs(self) -> dict[str, str]:
        return {
            "data_root": str(self.output_root),
            "reference_root": str(self.output_root),
            "reference_kind": "manual-gt",
            "plane": self.plane_name,
            "included": ",".join(self.included),
            "excluded_by_user": ",".join(self.excluded_by_user),
            "excluded_no_raw_suite2p": ",".join(self.excluded_no_raw_suite2p),
            "excluded_no_ground_truth": ",".join(self.excluded_no_ground_truth),
            "excluded_no_track2p_suite2p_indices": ",".join(
                self.excluded_no_track2p_suite2p_indices
            ),
            "excluded_incompatible": ",".join(self.excluded_incompatible),
            "filtered_manual_gt_tracks": ",".join(self.filtered_manual_gt_tracks),
            "has_usable_subjects": str(self.has_usable_subjects).lower(),
        }


@dataclass(frozen=True)
class _CandidateSubject:
    name: str
    path: Path
    has_raw_suite2p: bool
    has_ground_truth: bool
    has_track2p_suite2p_indices: bool


def prepare_raw_suite2p_benchmark_data(
    *,
    raw_root: Path,
    output_root: Path,
    metadata_root: Path | None = None,
    plane_name: str = "plane0",
    exclude_subjects: Iterable[str] = (),
    min_subjects: int = 1,
    diagnostics_dir: Path | None = None,
    require_track2p_suite2p_indices: bool = True,
    require_suite2p_ops_meanimg: bool = True,
    filter_missing_manual_rois: bool = False,
) -> RawBenchmarkPreparation:
    """Build a benchmark tree using raw Suite2p sessions and Track2p/GT metadata."""

    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    metadata_root = raw_root if metadata_root is None else Path(metadata_root).resolve()
    excluded_by_user = frozenset(
        name.strip() for name in exclude_subjects if name.strip()
    )

    raw_subjects = _discover_candidate_subjects(raw_root, plane_name=plane_name)
    metadata_subjects = _discover_candidate_subjects(
        metadata_root, plane_name=plane_name
    )
    subject_names = sorted(set(raw_subjects) | set(metadata_subjects))

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    included: list[str] = []
    excluded_user: list[str] = []
    excluded_no_raw: list[str] = []
    excluded_no_gt: list[str] = []
    excluded_no_suite2p_indices: list[str] = []
    excluded_incompatible: list[str] = []
    filtered_manual_gt_tracks: list[str] = []
    diagnostics: list[RawBenchmarkDiagnostic] = []

    for subject_name in subject_names:
        if subject_name in excluded_by_user:
            excluded_user.append(subject_name)
            continue

        raw_subject = raw_subjects.get(subject_name)
        metadata_subject = metadata_subjects.get(subject_name)
        if raw_subject is None or not raw_subject.has_raw_suite2p:
            excluded_no_raw.append(subject_name)
            continue
        if metadata_subject is None or not metadata_subject.has_ground_truth:
            excluded_no_gt.append(subject_name)
            continue
        track2p_subject = (
            metadata_subject
            if metadata_subject.has_track2p_suite2p_indices
            else raw_subject if raw_subject.has_track2p_suite2p_indices else None
        )
        if require_track2p_suite2p_indices and track2p_subject is None:
            excluded_no_suite2p_indices.append(subject_name)
            continue

        track2p_session_names = (
            None
            if track2p_subject is None
            else _track2p_bridge_session_names(
                track2p_subject.path,
                plane_name=plane_name,
            )
        )
        prepared_subject = output_root / subject_name
        prepared_subject.mkdir()
        _link_raw_suite2p_sessions(
            raw_subject.path,
            prepared_subject,
            plane_name=plane_name,
            session_names=track2p_session_names,
        )
        _link_path(
            metadata_subject.path / GROUND_TRUTH_CSV_NAME,
            prepared_subject / GROUND_TRUTH_CSV_NAME,
        )
        if track2p_subject is not None:
            _link_path(track2p_subject.path / "track2p", prepared_subject / "track2p")

        incompatibilities, subject_diagnostics, removed_manual_gt_tracks = (
            _validate_prepared_subject(
                prepared_subject,
                plane_name=plane_name,
                require_track2p_suite2p_indices=require_track2p_suite2p_indices,
                require_suite2p_ops_meanimg=require_suite2p_ops_meanimg,
                filter_missing_manual_rois=filter_missing_manual_rois,
            )
        )
        diagnostics.extend(subject_diagnostics)
        if removed_manual_gt_tracks:
            filtered_manual_gt_tracks.append(
                f"{subject_name}:{removed_manual_gt_tracks}"
            )
        if incompatibilities:
            shutil.rmtree(prepared_subject)
            excluded_incompatible.append(
                f"{subject_name} ({'; '.join(incompatibilities)})"
            )
            continue
        included.append(subject_name)

    summary = RawBenchmarkPreparation(
        output_root=output_root,
        plane_name=plane_name,
        included=tuple(included),
        excluded_by_user=tuple(sorted(excluded_user)),
        excluded_no_raw_suite2p=tuple(sorted(excluded_no_raw)),
        excluded_no_ground_truth=tuple(sorted(excluded_no_gt)),
        excluded_no_track2p_suite2p_indices=tuple(sorted(excluded_no_suite2p_indices)),
        excluded_incompatible=tuple(sorted(excluded_incompatible)),
        filtered_manual_gt_tracks=tuple(sorted(filtered_manual_gt_tracks)),
        diagnostics=tuple(diagnostics),
    )
    if diagnostics_dir is not None:
        write_raw_benchmark_diagnostics(summary, diagnostics_dir)
    if len(summary.included) < min_subjects:
        raise ValueError(
            f"Need at least {min_subjects} raw Suite2p manual-GT subject(s); "
            f"included={list(summary.included)}, excluded_incompatible={list(summary.excluded_incompatible)}"
        )
    return summary


def write_raw_benchmark_diagnostics(
    preparation: RawBenchmarkPreparation, diagnostics_dir: Path
) -> None:
    """Write machine- and human-readable raw benchmark preparation diagnostics."""

    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = diagnostics_dir / "raw_suite2p_benchmark_subjects.json"
    summary_path.write_text(
        json.dumps(preparation.to_outputs(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = diagnostics_dir / "raw_suite2p_roi_diagnostics.csv"
    fieldnames = [
        "subject",
        "source",
        "session",
        "session_index",
        "referenced_rois",
        "loaded_rois",
        "missing_rois",
        "compatible",
        "referenced_max",
        "loaded_max",
        "missing_preview",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diagnostic in preparation.diagnostics:
            writer.writerow(diagnostic.to_dict())

    lines = [
        "## Raw Suite2p benchmark data",
        "",
        f"Included usable manual-GT subjects: {', '.join(preparation.included) or '(none)'}",
        f"Excluded by user: {', '.join(preparation.excluded_by_user) or '(none)'}",
        f"Excluded without raw Suite2p sessions: {', '.join(preparation.excluded_no_raw_suite2p) or '(none)'}",
        f"Excluded without ground_truth.csv: {', '.join(preparation.excluded_no_ground_truth) or '(none)'}",
        f"Excluded without track2p/{preparation.plane_name}_suite2p_indices.npy: {', '.join(preparation.excluded_no_track2p_suite2p_indices) or '(none)'}",
        f"Excluded with incompatible ROI indices: {', '.join(preparation.excluded_incompatible) or '(none)'}",
        f"Filtered manual-GT tracks with absent raw Suite2p ROIs: {', '.join(preparation.filtered_manual_gt_tracks) or '(none)'}",
        "",
        "| subject | source | session | referenced ROIs | loaded ROIs | missing ROIs | compatible |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for diagnostic in preparation.diagnostics:
        lines.append(
            f"| {diagnostic.subject} | {diagnostic.source} | {diagnostic.session} | "
            f"{diagnostic.referenced_rois} | {diagnostic.loaded_rois} | {diagnostic.missing_rois} | {str(diagnostic.compatible).lower()} |"
        )
    (diagnostics_dir / "raw_suite2p_roi_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _discover_candidate_subjects(
    root: Path, *, plane_name: str
) -> dict[str, _CandidateSubject]:
    candidates: dict[str, _CandidateSubject] = {}
    if not root.exists():
        return candidates
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name
        if not _looks_like_subject_name(name):
            continue
        candidate = _CandidateSubject(
            name=name,
            path=path,
            has_raw_suite2p=_has_raw_suite2p_sessions(path, plane_name=plane_name),
            has_ground_truth=(path / GROUND_TRUTH_CSV_NAME).is_file(),
            has_track2p_suite2p_indices=(
                path / "track2p" / f"{plane_name}_suite2p_indices.npy"
            ).is_file()
            and (path / "track2p" / "track_ops.npy").is_file(),
        )
        if (
            candidate.has_raw_suite2p
            or candidate.has_ground_truth
            or (path / "track2p").exists()
        ):
            current = candidates.get(name)
            if current is None or _candidate_score(candidate) > _candidate_score(
                current
            ):
                candidates[name] = candidate
    return candidates


def _candidate_score(candidate: _CandidateSubject) -> tuple[int, int, int, int]:
    return (
        int(candidate.has_raw_suite2p),
        int(candidate.has_ground_truth),
        int(candidate.has_track2p_suite2p_indices),
        -len(candidate.path.parts),
    )


def _looks_like_subject_name(name: str) -> bool:
    return len(name) == 5 and name.startswith("jm") and name[2:].isdigit()


def _has_raw_suite2p_sessions(subject_dir: Path, *, plane_name: str) -> bool:
    return any(
        (session_dir / "suite2p" / plane_name / "stat.npy").is_file()
        for session_dir in find_track2p_session_dirs(subject_dir)
    )


def _link_raw_suite2p_sessions(
    raw_subject: Path,
    prepared_subject: Path,
    *,
    plane_name: str,
    session_names: Sequence[str] | None = None,
) -> None:
    linked = False
    raw_session_dirs = find_track2p_session_dirs(raw_subject)
    if session_names is None:
        session_dirs = raw_session_dirs
    else:
        raw_session_by_name = {
            session_dir.name: session_dir for session_dir in raw_session_dirs
        }
        session_dirs = [
            raw_session_by_name[session_name]
            for session_name in session_names
            if session_name in raw_session_by_name
        ]
    for session_dir in session_dirs:
        if not (session_dir / "suite2p" / plane_name / "stat.npy").is_file():
            continue
        _link_path(session_dir, prepared_subject / session_dir.name)
        linked = True
    if not linked:
        raise ValueError(
            f"No raw Suite2p {plane_name} sessions found for {raw_subject}"
        )


def _track2p_bridge_session_names(
    subject_dir: Path,
    *,
    plane_name: str,
) -> tuple[str, ...] | None:
    track2p_dir = subject_dir / "track2p"
    if not track2p_dir.exists():
        return None
    return load_track2p_reference(track2p_dir, plane_name=plane_name).session_names


def _validate_prepared_subject(
    subject_dir: Path,
    *,
    plane_name: str,
    require_track2p_suite2p_indices: bool,
    require_suite2p_ops_meanimg: bool,
    filter_missing_manual_rois: bool,
) -> tuple[list[str], list[RawBenchmarkDiagnostic], int]:
    config = Track2pBenchmarkConfig(
        data=subject_dir.parent,
        method="track2p-baseline",
        reference=subject_dir.parent,
        reference_kind="manual-gt",
        include_behavior=False,
        include_non_cells=True,
        plane_name=plane_name,
    )
    sessions = _load_subject_sessions(subject_dir, config)
    ground_truth = _load_ground_truth_csv_reference(
        subject_dir / GROUND_TRUTH_CSV_NAME,
        subject_dir=subject_dir,
    )
    incompatibilities: list[str] = []
    diagnostics: list[RawBenchmarkDiagnostic] = []
    removed_manual_gt_tracks = 0
    if require_suite2p_ops_meanimg:
        incompatibilities.extend(
            _suite2p_mean_image_incompatibilities(
                subject_dir,
                sessions,
                plane_name=plane_name,
            )
        )

    if filter_missing_manual_rois:
        ground_truth, removed_manual_gt_tracks = _filter_reference_to_loaded_rois(
            ground_truth,
            sessions,
        )
        if removed_manual_gt_tracks:
            _write_ground_truth_reference_csv(
                ground_truth,
                subject_dir / GROUND_TRUTH_CSV_NAME,
            )

    references: list[tuple[str, Track2pReference]] = [("manual_gt", ground_truth)]
    track2p_dir = subject_dir / "track2p"
    if track2p_dir.exists():
        track2p_reference = load_track2p_reference(track2p_dir, plane_name=plane_name)
        references.append(("track2p_suite2p_indices", track2p_reference))
        if (
            require_track2p_suite2p_indices
            and track2p_reference.source != "track2p_output_suite2p_indices"
        ):
            incompatibilities.append(
                f"Track2p reference source is {track2p_reference.source}, not track2p_output_suite2p_indices"
            )
    elif require_track2p_suite2p_indices:
        incompatibilities.append(f"Missing track2p/{plane_name}_suite2p_indices.npy")

    for source, reference in references:
        source_incompatibilities, source_diagnostics = _reference_coverage_diagnostics(
            subject_dir.name,
            source,
            reference,
            sessions,
        )
        if source == "manual_gt":
            incompatibilities.extend(source_incompatibilities)
        elif require_track2p_suite2p_indices:
            incompatibilities.extend(
                incompatibility
                for incompatibility in source_incompatibilities
                if "missing ROI indices" not in incompatibility
            )
        diagnostics.extend(source_diagnostics)
    return incompatibilities, diagnostics, removed_manual_gt_tracks


def _suite2p_mean_image_incompatibilities(
    subject_dir: Path,
    sessions: Sequence,
    *,
    plane_name: str,
) -> list[str]:
    incompatibilities: list[str] = []
    for session in sessions:
        plane_dir = subject_dir / session.session_name / "suite2p" / plane_name
        ops_path = plane_dir / "ops.npy"
        if not ops_path.is_file():
            incompatibilities.append(
                f"{session.session_name} missing suite2p/{plane_name}/ops.npy; "
                "FOV registration would use ROI-mask occupancy fallback"
            )
            continue
        ops = session.plane_data.ops
        mean_image = None if ops is None else ops.get("meanImg")
        if mean_image is None:
            incompatibilities.append(
                f"{session.session_name} ops.npy lacks meanImg; "
                "FOV registration would use ROI-mask occupancy fallback"
            )
            continue
        mean_image_array = np.asarray(mean_image)
        if mean_image_array.ndim != 2:
            incompatibilities.append(
                f"{session.session_name} ops.npy meanImg has shape "
                f"{mean_image_array.shape!r}, not a 2-D FOV image"
            )
            continue
        if mean_image_array.shape != session.plane_data.image_shape:
            incompatibilities.append(
                f"{session.session_name} ops.npy meanImg shape "
                f"{mean_image_array.shape!r} does not match ROI mask shape "
                f"{session.plane_data.image_shape!r}"
            )
    return incompatibilities


def _filter_reference_to_loaded_rois(
    reference: Track2pReference,
    sessions: Sequence,
) -> tuple[Track2pReference, int]:
    if len(sessions) != reference.n_sessions:
        return reference, 0
    session_names = tuple(session.session_name for session in sessions)
    if session_names != reference.session_names:
        return reference, 0

    available_by_session = [
        set(_loaded_suite2p_index_set(session)) for session in sessions
    ]
    keep = np.ones((reference.n_tracks,), dtype=bool)
    for row_index, row in enumerate(reference.suite2p_indices):
        for session_index, value in enumerate(row):
            if (
                value is not None
                and int(value) not in available_by_session[session_index]
            ):
                keep[row_index] = False
                break

    removed = int(np.count_nonzero(~keep))
    if removed == 0:
        return reference, 0
    curated_mask = (
        None if reference.curated_mask is None else reference.curated_mask[keep]
    )
    return (
        Track2pReference(
            session_names=reference.session_names,
            suite2p_indices=reference.suite2p_indices[keep],
            session_dates=reference.session_dates,
            curated_mask=curated_mask,
            source=reference.source,
        ),
        removed,
    )


def _write_ground_truth_reference_csv(
    reference: Track2pReference,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track_id", *reference.session_names))
        for track_id, row in enumerate(reference.suite2p_indices):
            writer.writerow(
                (
                    track_id,
                    *("" if value is None else int(value) for value in row),
                )
            )


def _reference_coverage_diagnostics(
    subject_name: str,
    source: str,
    reference: Track2pReference,
    sessions: Sequence,
) -> tuple[list[str], list[RawBenchmarkDiagnostic]]:
    diagnostics: list[RawBenchmarkDiagnostic] = []
    incompatibilities: list[str] = []
    if len(sessions) != reference.n_sessions:
        return (
            [
                f"{source} has {reference.n_sessions} sessions but raw data has {len(sessions)}"
            ],
            diagnostics,
        )
    session_names = tuple(session.session_name for session in sessions)
    if session_names != reference.session_names:
        return (
            [
                f"{source} session order {reference.session_names!r} does not match raw sessions {session_names!r}"
            ],
            diagnostics,
        )
    for session_index, session in enumerate(sessions):
        available = sorted(_loaded_suite2p_index_set(session))
        referenced = sorted(
            {
                int(value)
                for value in reference.suite2p_indices[:, session_index]
                if value is not None
            }
        )
        missing = sorted(set(referenced) - set(available))
        if missing:
            incompatibilities.append(
                f"{source} has {len(missing)} missing ROI indices in {session.session_name}: {' '.join(str(value) for value in missing[:5])}"
            )
        diagnostics.append(
            RawBenchmarkDiagnostic(
                subject=subject_name,
                source=source,
                session=session.session_name,
                session_index=session_index,
                referenced_rois=len(referenced),
                loaded_rois=len(available),
                missing_rois=len(missing),
                compatible=not missing,
                referenced_max=max(referenced) if referenced else None,
                loaded_max=max(available) if available else None,
                missing_preview=tuple(missing[:20]),
            )
        )
    return incompatibilities, diagnostics


def _link_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _parse_excluded_subjects(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _write_github_outputs(outputs: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            print(f"{key}={value}", file=handle)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--plane", default="plane0")
    parser.add_argument("--exclude-subjects", default="")
    parser.add_argument("--min-subjects", type=int, default=1)
    parser.add_argument(
        "--require-track2p-suite2p-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a compatible track2p/<plane>_suite2p_indices.npy bridge. "
            "Disable for manual-GT-only diagnostics that do not run Track2p baseline rows."
        ),
    )
    parser.add_argument(
        "--require-suite2p-ops-meanimg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require suite2p/<plane>/ops.npy with a meanImg FOV for every raw "
            "session. This prevents FOV registration from silently using ROI-mask "
            "occupancy as the image."
        ),
    )
    parser.add_argument(
        "--filter-missing-manual-rois",
        action="store_true",
        help=(
            "Drop manual-GT tracks that reference Suite2p ROI indices absent from "
            "the loaded raw sessions. This is diagnostic-only and is reported in outputs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preparation = prepare_raw_suite2p_benchmark_data(
        raw_root=args.raw_root,
        metadata_root=args.metadata_root,
        output_root=args.output,
        plane_name=args.plane,
        exclude_subjects=_parse_excluded_subjects(args.exclude_subjects),
        min_subjects=args.min_subjects,
        diagnostics_dir=args.diagnostics_dir,
        require_track2p_suite2p_indices=args.require_track2p_suite2p_indices,
        require_suite2p_ops_meanimg=args.require_suite2p_ops_meanimg,
        filter_missing_manual_rois=args.filter_missing_manual_rois,
    )
    outputs = preparation.to_outputs()
    _write_github_outputs(outputs)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
