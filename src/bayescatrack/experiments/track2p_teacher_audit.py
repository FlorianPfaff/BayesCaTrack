"""Track2p teacher/debug-oracle audit for manual-GT benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import AssociationCost

if TYPE_CHECKING:
    from bayescatrack.experiments.track2p_benchmark import (
        BenchmarkMethod,
        ReferenceKind,
        Track2pBenchmarkConfig,
    )

PairMode = Literal["all", "consecutive", "max-gap"]
OutputFormat = Literal["table", "json", "csv"]
TRACK2P_TEACHER_MISS_CATEGORY = "GT+Track2p+Bayes-"
_FIELDS = {
    "GT+Track2p+Bayes+": "edges_gt_track2p_bayes",
    TRACK2P_TEACHER_MISS_CATEGORY: "edges_gt_track2p_not_bayes",
    "GT+Track2p-Bayes+": "edges_gt_not_track2p_bayes",
    "GT+Track2p-Bayes-": "edges_gt_not_track2p_not_bayes",
    "GT-Track2p+Bayes+": "edges_not_gt_track2p_bayes",
    "GT-Track2p+Bayes-": "edges_not_gt_track2p_not_bayes",
    "GT-Track2p-Bayes+": "edges_not_gt_not_track2p_bayes",
}


@dataclass(frozen=True)
class Track2pTeacherAuditConfig:
    data: Path
    ground_truth_reference: Path | None = None
    track2p_reference: Path | None = None
    pair_mode: PairMode = "all"
    plane_name: str = "plane0"
    input_format: str = "auto"
    curated_only: bool = False
    seed_session: int = 0
    restrict_to_reference_seed_rois: bool = True
    cost: AssociationCost = "registered-iou"
    max_gap: int = 2
    transform_type: str = "affine"
    start_cost: float = 5.0
    end_cost: float = 5.0
    gap_penalty: float = 1.0
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    progress: bool = False


@dataclass(frozen=True)
class Track2pTeacherAuditResult:
    summary_rows: list[dict[str, Any]]
    edge_rows: list[dict[str, Any]]


def audit_track_matrices(
    *,
    subject: str,
    session_names: Sequence[str],
    ground_truth_tracks: Any,
    track2p_tracks: Any,
    bayes_tracks: Any,
    pair_mode: PairMode = "all",
    max_gap: int = 2,
    seed_session: int = 0,
    restrict_to_reference_seed_rois: bool = True,
) -> Track2pTeacherAuditResult:
    names = tuple(map(str, session_names))
    gt, t2p, bayes = (
        _norm(ground_truth_tracks),
        _norm(track2p_tracks),
        _norm(bayes_tracks),
    )
    for label, mat in (
        ("ground_truth_tracks", gt),
        ("track2p_tracks", t2p),
        ("bayes_tracks", bayes),
    ):
        if mat.ndim != 2 or mat.shape[1] != len(names):
            raise ValueError(f"{label} must have one column per session")
    seed_rois = {int(v) for v in gt[:, seed_session] if v is not None}
    if restrict_to_reference_seed_rois:
        gt, t2p, bayes = (
            _seed_filter(m, seed_rois, seed_session) for m in (gt, t2p, bayes)
        )
    pairs = _pairs(len(names), pair_mode, max_gap)
    gt_e, t2p_e, bayes_e = (_edge_map(m, pairs) for m in (gt, t2p, bayes))
    edge_rows = _edge_rows(subject, names, gt_e, t2p_e, bayes_e)
    summary = _summary(
        subject,
        len(names),
        pair_mode,
        max_gap,
        seed_session,
        len(seed_rois),
        restrict_to_reference_seed_rois,
        set(gt_e),
        set(t2p_e),
        set(bayes_e),
        edge_rows,
    )
    return Track2pTeacherAuditResult([summary], edge_rows)


def run_track2p_teacher_audit(
    config: Track2pTeacherAuditConfig,
) -> Track2pTeacherAuditResult:
    from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=import-outside-toplevel,protected-access
        GROUND_TRUTH_REFERENCE_SOURCE,
        _load_reference_for_subject,
        _load_subject_sessions,
        _predict_subject_tracks,
        _reference_matrix,
        _validate_reference_roi_indices,
        discover_subject_dirs,
    )

    summaries, edges = [], []
    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )
    progress = _Progress(len(subject_dirs), config.progress)
    for subject_dir in subject_dirs:
        progress.step(f"auditing {subject_dir.name}")
        gt_cfg = _bench_cfg(
            config,
            method="global-assignment",
            reference=config.ground_truth_reference,
            reference_kind="manual-gt",
        )
        gt_ref = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=gt_cfg
        )
        if gt_ref.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError("teacher-audit requires independent manual ground truth")
        _validate_reference_roi_indices(
            gt_ref, _load_subject_sessions(subject_dir, gt_cfg)
        )
        t_cfg = _bench_cfg(
            config,
            method="track2p-baseline",
            reference=config.track2p_reference,
            reference_kind="track2p-output",
            allow_track2p_as_reference_for_smoke_test=True,
        )
        t_ref = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=t_cfg
        )
        bayes_mat, _ = _predict_subject_tracks(subject_dir, gt_cfg, reference=gt_ref)
        res = audit_track_matrices(
            subject=subject_dir.name,
            session_names=gt_ref.session_names,
            ground_truth_tracks=_reference_matrix(
                gt_ref, curated_only=config.curated_only
            ),
            track2p_tracks=_reference_matrix(
                t_ref,
                curated_only=config.curated_only and t_ref.curated_mask is not None,
            ),
            bayes_tracks=bayes_mat,
            pair_mode=config.pair_mode,
            max_gap=config.max_gap,
            seed_session=config.seed_session,
            restrict_to_reference_seed_rois=config.restrict_to_reference_seed_rois,
        )
        summaries.extend(res.summary_rows)
        edges.extend(res.edge_rows)
    return Track2pTeacherAuditResult(summaries, edges)


def teacher_training_rows(
    edge_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "subject": r["subject"],
            "session_a": r["session_a"],
            "session_b": r["session_b"],
            "gap": r["gap"],
            "roi_a": r["roi_a"],
            "roi_b": r["roi_b"],
            "teacher_label": int(bool(r["in_track2p"])),
            "manual_gt_label": int(bool(r["in_ground_truth"])),
            "bayes_label": int(bool(r["in_bayes"])),
            "teacher_label_source": "track2p_output",
            "category": r["category"],
        }
        for r in edge_rows
    ]


def format_teacher_audit_table(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = [
        "subject",
        "pair_mode",
        "ground_truth_edges",
        "track2p_edges",
        "bayes_edges",
        "edges_gt_track2p_not_bayes",
        "track2p_vs_gt_f1",
        "bayes_vs_gt_f1",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(cols) - 1)) + " |",
    ]
    lines += [
        "| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |" for row in rows
    ]
    return "\n".join(lines)


def write_edge_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_path)


def write_summary_rows(
    rows: Sequence[Mapping[str, Any]], output_path: Path, output_format: OutputFormat
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
    elif output_format == "csv":
        _write_csv(rows, output_path)
    else:
        output_path.write_text(
            format_teacher_audit_table(rows) + "\n", encoding="utf-8"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bayescatrack benchmark track2p-teacher-audit")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--ground-truth-reference", type=Path)
    p.add_argument("--track2p-reference", type=Path)
    p.add_argument(
        "--pair-mode", choices=("all", "consecutive", "max-gap"), default="all"
    )
    p.add_argument("--plane", dest="plane_name", default="plane0")
    p.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    p.add_argument("--curated-only", action="store_true")
    p.add_argument("--seed-session", type=int, default=0)
    p.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--cost", default="registered-iou", choices=("registered-iou", "roi-aware")
    )
    p.add_argument("--max-gap", type=int, default=2)
    p.add_argument("--transform-type", default="affine")
    p.add_argument("--start-cost", type=float, default=5.0)
    p.add_argument("--end-cost", type=float, default=5.0)
    p.add_argument("--gap-penalty", type=float, default=1.0)
    p.add_argument("--cost-threshold", type=float, default=6.0)
    p.add_argument("--no-cost-threshold", action="store_true")
    p.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--include-non-cells", action="store_true")
    p.add_argument("--cell-probability-threshold", type=float, default=0.5)
    p.add_argument("--weighted-masks", action="store_true")
    p.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--order", default="xy", choices=("xy", "yx"))
    p.add_argument("--weighted-centroids", action="store_true")
    p.add_argument("--velocity-variance", type=float, default=25.0)
    p.add_argument("--regularization", type=float, default=1e-6)
    p.add_argument("--pairwise-cost-kwargs-json")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--edges-output", type=Path)
    p.add_argument("--focus-output", type=Path)
    p.add_argument("--teacher-output", type=Path)
    p.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_track2p_teacher_audit(_config(args))
    if args.output:
        write_summary_rows(
            result.summary_rows, args.output, cast(OutputFormat, args.format)
        )
    else:
        _stdout(result.summary_rows, cast(OutputFormat, args.format))
    if args.edges_output:
        write_edge_rows(result.edge_rows, args.edges_output)
    if args.focus_output:
        write_edge_rows(
            [
                r
                for r in result.edge_rows
                if r["category"] == TRACK2P_TEACHER_MISS_CATEGORY
            ],
            args.focus_output,
        )
    if args.teacher_output:
        write_edge_rows(teacher_training_rows(result.edge_rows), args.teacher_output)
    return 0


def _edge_rows(
    subject: str, names: Sequence[str], gt: dict, t2p: dict, bayes: dict
) -> list[dict[str, Any]]:
    rows = []
    for e in sorted(set(gt) | set(t2p) | set(bayes)):
        sa, sb, ra, rb = e
        ingt, int2p, inb = e in gt, e in t2p, e in bayes
        cat = f"GT{'+' if ingt else '-'}Track2p{'+' if int2p else '-'}Bayes{'+' if inb else '-'}"
        rows.append(
            {
                "subject": subject,
                "session_a": sa,
                "session_b": sb,
                "session_a_name": names[sa],
                "session_b_name": names[sb],
                "gap": sb - sa,
                "roi_a": ra,
                "roi_b": rb,
                "in_ground_truth": ingt,
                "in_track2p": int2p,
                "in_bayes": inb,
                "category": cat,
                "ground_truth_track_row": gt.get(e, ""),
                "track2p_track_row": t2p.get(e, ""),
                "bayes_track_row": bayes.get(e, ""),
            }
        )
    return rows


def _summary(
    subject: str,
    n: int,
    mode: str,
    max_gap: int,
    seed: int,
    n_seed: int,
    restrict: bool,
    gt: set,
    t2p: set,
    bayes: set,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    c = Counter(str(r["category"]) for r in rows)
    s_t, s_b = _scores(t2p, gt), _scores(bayes, gt)
    out: dict[str, Any] = {
        "subject": subject,
        "n_sessions": n,
        "pair_mode": mode,
        "max_gap": max_gap,
        "seed_session": seed,
        "reference_seed_rois": n_seed,
        "restrict_to_reference_seed_rois": int(restrict),
        "ground_truth_edges": len(gt),
        "track2p_edges": len(t2p),
        "bayes_edges": len(bayes),
        "track2p_vs_gt_precision": s_t[0],
        "track2p_vs_gt_recall": s_t[1],
        "track2p_vs_gt_f1": s_t[2],
        "bayes_vs_gt_precision": s_b[0],
        "bayes_vs_gt_recall": s_b[1],
        "bayes_vs_gt_f1": s_b[2],
    }
    for k, v in _FIELDS.items():
        out[v] = int(c.get(k, 0))
    denom = out["edges_gt_track2p_bayes"] + out["edges_gt_track2p_not_bayes"]
    out["bayes_miss_rate_on_gt_track2p_agreement"] = _ratio(
        out["edges_gt_track2p_not_bayes"], denom
    )
    return out


def _edge_map(
    m: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> dict[tuple[int, int, int, int], int]:
    out: dict[tuple[int, int, int, int], int] = {}
    for r, row in enumerate(m):
        for a, b in pairs:
            if row[a] is not None and row[b] is not None:
                out.setdefault((a, b, int(row[a]), int(row[b])), r)
    return out


def _pairs(n: int, mode: str, max_gap: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (a, b)
        for a in range(max(0, n - 1))
        for b in range(a + 1, n)
        if not (mode == "consecutive" and b - a != 1)
        and not (mode == "max-gap" and b - a > max_gap)
    )


def _norm(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=object)
    a = a.reshape(-1, 1) if a.ndim == 1 else a
    out = np.empty(a.shape, dtype=object)
    for idx, v in np.ndenumerate(a):
        out[idx] = _parse(v)
    return out


def _parse(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if isinstance(v, str):
        if v.strip().lower() in {"", "none", "nan", "null"}:
            return None
        v = v.strip()
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i >= 0 else None


def _seed_filter(m: np.ndarray, seed_rois: set[int], seed: int) -> np.ndarray:
    return (
        m[
            np.asarray(
                [v is not None and int(v) in seed_rois for v in m[:, seed]], dtype=bool
            )
        ]
        if seed_rois
        else m[:0]
    )


def _scores(pred: set, ref: set) -> tuple[float, float, float]:
    tp, fp, fn = len(pred & ref), len(pred - ref), len(ref - pred)
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return p, r, _ratio(2 * p * r, p + r)


def _bench_cfg(
    config: Track2pTeacherAuditConfig,
    *,
    method: "BenchmarkMethod",
    reference: Path | None,
    reference_kind: "ReferenceKind",
    allow_track2p_as_reference_for_smoke_test: bool = False,
) -> "Track2pBenchmarkConfig":
    from bayescatrack.experiments.track2p_benchmark import (
        Track2pBenchmarkConfig,
    )  # pylint: disable=import-outside-toplevel

    return Track2pBenchmarkConfig(
        data=config.data,
        method=method,
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=reference,
        reference_kind=reference_kind,
        allow_track2p_as_reference_for_smoke_test=allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        seed_session=config.seed_session,
        restrict_to_reference_seed_rois=config.restrict_to_reference_seed_rois,
        cost=config.cost,
        max_gap=config.max_gap,
        transform_type=config.transform_type,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        progress=config.progress,
    )


def _config(args: argparse.Namespace) -> Track2pTeacherAuditConfig:
    kwargs = None
    if args.pairwise_cost_kwargs_json:
        kwargs = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(kwargs, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    return Track2pTeacherAuditConfig(
        data=args.data,
        ground_truth_reference=args.ground_truth_reference,
        track2p_reference=args.track2p_reference,
        pair_mode=cast(PairMode, args.pair_mode),
        plane_name=args.plane_name,
        input_format=args.input_format,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=args.cost,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=kwargs,
        progress=args.progress,
    )


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=_fields(rows))
        w.writeheader()
        w.writerows(rows)


def _stdout(rows: Sequence[Mapping[str, Any]], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(list(rows), indent=2))
    elif fmt == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=_fields(rows))
        w.writeheader()
        w.writerows(rows)
    else:
        print(format_teacher_audit_table(rows))


def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(k for r in rows for k in r))


def _fmt(v: object) -> str:
    return f"{float(v):.3f}" if isinstance(v, (float, np.floating)) else str(v)


def _ratio(a: float, b: float) -> float:
    return 1.0 if b == 0 else float(a) / float(b)


class _Progress:
    def __init__(self, total: int, enabled: bool) -> None:
        self.total, self.enabled, self.i = max(total, 1), enabled, 0

    def step(self, msg: str) -> None:
        if self.enabled:
            self.i += 1
            print(
                f"teacher-audit {self.i}/{self.total} {msg}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
