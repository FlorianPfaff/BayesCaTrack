"""Track2p-as-teacher edge-disagreement audit utilities.

The audit compares three longitudinal ROI identity tables:

* independent manual ground truth,
* Track2p output used as a teacher/debug oracle, and
* a BayesCaTrack prediction.

It emits one row per pairwise edge in the union of those three edge sets.  The
most important category is ``GT+Track2p+Bayes-``: manually correct edges found
by Track2p but missed by BayesCaTrack.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class EdgeKey:
    """A directed longitudinal ROI edge from one session to a later session."""

    session_a: int
    session_b: int
    roi_a: int
    roi_b: int

    @property
    def gap(self) -> int:
        """Return the number of session steps crossed by this edge."""

        return self.session_b - self.session_a


@dataclass(frozen=True)
class TrackEdgeIndex:
    """Indexed pairwise edges derived from a longitudinal track matrix."""

    edges: frozenset[EdgeKey]
    rows_by_edge: Mapping[EdgeKey, tuple[int, ...]]
    targets_by_source: Mapping[tuple[int, int, int], tuple[int, ...]]
    sources_by_target: Mapping[tuple[int, int, int], tuple[int, ...]]
    n_tracks: int
    n_sessions: int


@dataclass(frozen=True)
class TeacherAuditRow:
    """One edge-level disagreement row."""

    subject: str
    session_a: int
    session_b: int
    gap: int
    session_name_a: str
    session_name_b: str
    roi_a: int
    roi_b: int
    manual_gt_edge: bool
    track2p_edge: bool
    bayes_edge: bool
    category: str
    manual_gt_track_rows: tuple[int, ...]
    track2p_track_rows: tuple[int, ...]
    bayes_track_rows: tuple[int, ...]
    track2p_targets_for_source: tuple[int, ...]
    bayes_targets_for_source: tuple[int, ...]
    track2p_sources_for_target: tuple[int, ...]
    bayes_sources_for_target: tuple[int, ...]

    def to_dict(self) -> dict[str, int | str]:
        """Return a CSV/JSON-friendly dictionary representation."""

        return {
            "subject": self.subject,
            "session_a": self.session_a,
            "session_b": self.session_b,
            "gap": self.gap,
            "session_name_a": self.session_name_a,
            "session_name_b": self.session_name_b,
            "roi_a": self.roi_a,
            "roi_b": self.roi_b,
            "manual_gt_edge": int(self.manual_gt_edge),
            "track2p_edge": int(self.track2p_edge),
            "bayes_edge": int(self.bayes_edge),
            "category": self.category,
            "manual_gt_track_rows": _join_ints(self.manual_gt_track_rows),
            "track2p_track_rows": _join_ints(self.track2p_track_rows),
            "bayes_track_rows": _join_ints(self.bayes_track_rows),
            "track2p_targets_for_source": _join_ints(self.track2p_targets_for_source),
            "bayes_targets_for_source": _join_ints(self.bayes_targets_for_source),
            "track2p_sources_for_target": _join_ints(self.track2p_sources_for_target),
            "bayes_sources_for_target": _join_ints(self.bayes_sources_for_target),
        }


@dataclass(frozen=True)
class TeacherAuditResult:
    """A complete teacher-audit report for one subject."""

    subject: str
    session_names: tuple[str, ...]
    rows: tuple[TeacherAuditRow, ...]
    summary: Mapping[str, int | float | str]

    def row_dicts(self) -> list[dict[str, int | str]]:
        """Return all edge-level rows as dictionaries."""

        return [row.to_dict() for row in self.rows]


CATEGORY_ORDER = (
    "GT+Track2p+Bayes+",
    "GT+Track2p+Bayes-",
    "GT+Track2p-Bayes+",
    "GT+Track2p-Bayes-",
    "GT-Track2p+Bayes+",
    "GT-Track2p+Bayes-",
    "GT-Track2p-Bayes+",
)

ROW_FIELDNAMES = (
    "subject",
    "session_a",
    "session_b",
    "gap",
    "session_name_a",
    "session_name_b",
    "roi_a",
    "roi_b",
    "manual_gt_edge",
    "track2p_edge",
    "bayes_edge",
    "category",
    "manual_gt_track_rows",
    "track2p_track_rows",
    "bayes_track_rows",
    "track2p_targets_for_source",
    "bayes_targets_for_source",
    "track2p_sources_for_target",
    "bayes_sources_for_target",
)


def build_track_edge_index(track_matrix: Any, *, max_gap: int | None = None) -> TrackEdgeIndex:
    """Index all pairwise ROI edges in a longitudinal track matrix.

    The expected layout is one track per row and one session per column. Missing
    detections may be encoded as ``None``, ``NaN``, an empty string, or a
    negative integer.
    """

    matrix = _as_object_matrix(track_matrix)
    if max_gap is not None and max_gap < 1:
        raise ValueError("max_gap must be at least 1 when provided")

    rows_by_edge: dict[EdgeKey, set[int]] = {}
    targets_by_source: dict[tuple[int, int, int], set[int]] = {}
    sources_by_target: dict[tuple[int, int, int], set[int]] = {}

    n_tracks, n_sessions = matrix.shape
    for row_index in range(n_tracks):
        row_rois = [_coerce_roi_index(value) for value in matrix[row_index, :]]
        for session_a in range(n_sessions):
            roi_a = row_rois[session_a]
            if roi_a is None:
                continue
            for session_b in range(session_a + 1, n_sessions):
                if max_gap is not None and session_b - session_a > max_gap:
                    continue
                roi_b = row_rois[session_b]
                if roi_b is None:
                    continue
                edge = EdgeKey(session_a=session_a, session_b=session_b, roi_a=roi_a, roi_b=roi_b)
                rows_by_edge.setdefault(edge, set()).add(row_index)
                targets_by_source.setdefault((session_a, session_b, roi_a), set()).add(roi_b)
                sources_by_target.setdefault((session_a, session_b, roi_b), set()).add(roi_a)

    return TrackEdgeIndex(
        edges=frozenset(rows_by_edge),
        rows_by_edge={edge: tuple(sorted(rows)) for edge, rows in rows_by_edge.items()},
        targets_by_source={key: tuple(sorted(values)) for key, values in targets_by_source.items()},
        sources_by_target={key: tuple(sorted(values)) for key, values in sources_by_target.items()},
        n_tracks=n_tracks,
        n_sessions=n_sessions,
    )


def audit_track2p_teacher_edges(
    manual_gt_tracks: Any,
    track2p_tracks: Any,
    bayes_tracks: Any,
    *,
    subject: str = "",
    session_names: Sequence[str] | None = None,
    max_gap: int | None = None,
    include_non_gt_edges: bool = True,
) -> TeacherAuditResult:
    """Compare manual-GT, Track2p-teacher, and BayesCaTrack pairwise edges."""

    gt_index = build_track_edge_index(manual_gt_tracks, max_gap=max_gap)
    teacher_index = build_track_edge_index(track2p_tracks, max_gap=max_gap)
    bayes_index = build_track_edge_index(bayes_tracks, max_gap=max_gap)
    _validate_compatible_session_count(gt_index, teacher_index, bayes_index)

    names = _normalize_session_names(session_names, gt_index.n_sessions)
    audited_edges = set(gt_index.edges)
    if include_non_gt_edges:
        audited_edges.update(teacher_index.edges)
        audited_edges.update(bayes_index.edges)

    rows = tuple(
        _audit_one_edge(
            edge,
            subject=subject,
            session_names=names,
            gt_index=gt_index,
            teacher_index=teacher_index,
            bayes_index=bayes_index,
        )
        for edge in sorted(audited_edges)
    )
    summary = summarize_teacher_audit(
        rows,
        subject=subject,
        session_names=names,
        gt_edges=gt_index.edges,
        teacher_edges=teacher_index.edges,
        bayes_edges=bayes_index.edges,
        n_manual_tracks=gt_index.n_tracks,
        n_track2p_tracks=teacher_index.n_tracks,
        n_bayes_tracks=bayes_index.n_tracks,
        max_gap=max_gap,
    )
    return TeacherAuditResult(subject=subject, session_names=names, rows=rows, summary=summary)


def summarize_teacher_audit(
    rows: Sequence[TeacherAuditRow],
    *,
    subject: str,
    session_names: Sequence[str],
    gt_edges: Iterable[EdgeKey],
    teacher_edges: Iterable[EdgeKey],
    bayes_edges: Iterable[EdgeKey],
    n_manual_tracks: int,
    n_track2p_tracks: int,
    n_bayes_tracks: int,
    max_gap: int | None,
) -> dict[str, int | float | str]:
    """Summarize an edge-level teacher audit."""

    gt_edge_set = frozenset(gt_edges)
    teacher_edge_set = frozenset(teacher_edges)
    bayes_edge_set = frozenset(bayes_edges)
    counts = Counter(row.category for row in rows)
    track2p_precision, track2p_recall, track2p_f1 = _precision_recall_f1(teacher_edge_set, gt_edge_set)
    bayes_precision, bayes_recall, bayes_f1 = _precision_recall_f1(bayes_edge_set, gt_edge_set)
    both = gt_edge_set & teacher_edge_set & bayes_edge_set
    teacher_only = (gt_edge_set & teacher_edge_set) - bayes_edge_set
    bayes_only = (gt_edge_set & bayes_edge_set) - teacher_edge_set
    missed_by_both = gt_edge_set - teacher_edge_set - bayes_edge_set

    summary: dict[str, int | float | str] = {
        "subject": subject,
        "n_sessions": len(session_names),
        "max_gap": "all" if max_gap is None else max_gap,
        "n_manual_tracks": n_manual_tracks,
        "n_track2p_tracks": n_track2p_tracks,
        "n_bayes_tracks": n_bayes_tracks,
        "manual_gt_edges": len(gt_edge_set),
        "track2p_edges": len(teacher_edge_set),
        "bayes_edges": len(bayes_edge_set),
        "track2p_precision": track2p_precision,
        "track2p_recall": track2p_recall,
        "track2p_f1": track2p_f1,
        "bayes_precision": bayes_precision,
        "bayes_recall": bayes_recall,
        "bayes_f1": bayes_f1,
        "gt_found_by_both": len(both),
        "gt_track2p_found_bayes_missed": len(teacher_only),
        "gt_track2p_missed_bayes_found": len(bayes_only),
        "gt_missed_by_both": len(missed_by_both),
    }
    for category in CATEGORY_ORDER:
        summary[f"category_{_slug(category)}"] = counts.get(category, 0)
    return summary


def write_teacher_audit_rows_csv(rows: Sequence[TeacherAuditRow], path: str | Path) -> None:
    """Write edge-level teacher-audit rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_teacher_audit_summary_csv(summaries: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """Write one or more teacher-audit summaries to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for summary in summaries for key in summary})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(summary) for summary in summaries)


def _audit_one_edge(
    edge: EdgeKey,
    *,
    subject: str,
    session_names: Sequence[str],
    gt_index: TrackEdgeIndex,
    teacher_index: TrackEdgeIndex,
    bayes_index: TrackEdgeIndex,
) -> TeacherAuditRow:
    gt_edge = edge in gt_index.edges
    teacher_edge = edge in teacher_index.edges
    bayes_edge = edge in bayes_index.edges
    source_key = (edge.session_a, edge.session_b, edge.roi_a)
    target_key = (edge.session_a, edge.session_b, edge.roi_b)
    return TeacherAuditRow(
        subject=subject,
        session_a=edge.session_a,
        session_b=edge.session_b,
        gap=edge.gap,
        session_name_a=session_names[edge.session_a],
        session_name_b=session_names[edge.session_b],
        roi_a=edge.roi_a,
        roi_b=edge.roi_b,
        manual_gt_edge=gt_edge,
        track2p_edge=teacher_edge,
        bayes_edge=bayes_edge,
        category=_category(gt_edge, teacher_edge, bayes_edge),
        manual_gt_track_rows=gt_index.rows_by_edge.get(edge, ()),
        track2p_track_rows=teacher_index.rows_by_edge.get(edge, ()),
        bayes_track_rows=bayes_index.rows_by_edge.get(edge, ()),
        track2p_targets_for_source=teacher_index.targets_by_source.get(source_key, ()),
        bayes_targets_for_source=bayes_index.targets_by_source.get(source_key, ()),
        track2p_sources_for_target=teacher_index.sources_by_target.get(target_key, ()),
        bayes_sources_for_target=bayes_index.sources_by_target.get(target_key, ()),
    )


def _precision_recall_f1(predicted: frozenset[EdgeKey], reference: frozenset[EdgeKey]) -> tuple[float, float, float]:
    true_positives = len(predicted & reference)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _as_object_matrix(track_matrix: Any) -> np.ndarray:
    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must be a two-dimensional tracks-by-sessions matrix")
    return matrix


def _coerce_roi_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null"}:
            return None
        value = stripped
    try:
        if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    roi = int(value)
    if roi < 0:
        return None
    return roi


def _normalize_session_names(session_names: Sequence[str] | None, n_sessions: int) -> tuple[str, ...]:
    if session_names is None:
        return tuple(str(index) for index in range(n_sessions))
    names = tuple(str(name) for name in session_names)
    if len(names) != n_sessions:
        raise ValueError(f"expected {n_sessions} session names, got {len(names)}")
    return names


def _validate_compatible_session_count(*indices: TrackEdgeIndex) -> None:
    counts = {index.n_sessions for index in indices}
    if len(counts) != 1:
        raise ValueError(f"track matrices must have the same number of sessions, got {sorted(counts)}")


def _category(gt_edge: bool, teacher_edge: bool, bayes_edge: bool) -> str:
    return f"GT{'+' if gt_edge else '-'}Track2p{'+' if teacher_edge else '-'}Bayes{'+' if bayes_edge else '-'}"


def _join_ints(values: Sequence[int]) -> str:
    return ";".join(str(value) for value in values)


def _slug(category: str) -> str:
    return category.replace("+", "_plus_").replace("-", "_minus_").strip("_").lower()
