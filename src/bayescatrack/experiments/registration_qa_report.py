"""Registration quality report for Track2p-style benchmark subjects."""

# pylint: disable=protected-access,too-many-locals,too-many-arguments,too-many-positional-arguments
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.association.activity_similarity import (
    add_activity_similarity_components,
)
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    CalibratedAssociationModel,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
    roi_aware_cost_kwargs,
    roi_aware_shifted_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.association.shifted_overlap import (
    install_shifted_overlap_cost_patch,
    pairwise_kwargs_use_shifted_overlap,
)
from bayescatrack.core.bridge import (
    CalciumPlaneData,
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair

RegistrationQACost = Literal[
    "registered-iou", "roi-aware", "roi-aware-shifted", "calibrated"
]
RegistrationQALevel = Literal["summary", "links", "backend-audit"]
OutputFormat = Literal["table", "json", "csv"]


@dataclass(frozen=True)
class RegistrationQAConfig:
    """Configuration for a registration QA report."""

    data: Path
    reference: Path | None = None
    reference_kind: ReferenceKind = "auto"
    allow_track2p_as_reference_for_smoke_test: bool = False
    curated_only: bool = False
    plane_name: str = "plane0"
    input_format: str = "auto"
    max_gap: int = 2
    transform_type: str = "affine"
    cost: RegistrationQACost = "registered-iou"
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    progress: bool = False


def run_registration_qa_report(config: RegistrationQAConfig) -> list[dict[str, Any]]:
    """Return one diagnostics row for each manual-GT link and audited edge."""

    if config.cost == "calibrated":
        return _run_calibrated_loso_registration_qa_report(config)

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    benchmark_config = _benchmark_config(config)
    rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        if config.progress:
            print(f"registration-qa: {subject_dir.name}", file=sys.stderr, flush=True)
        reference = _load_reference_for_subject(
            subject_dir,
            data_root=config.data,
            config=benchmark_config,
        )
        _validate_reference_for_benchmark(
            reference,
            subject_dir=subject_dir,
            config=benchmark_config,
        )
        sessions = _load_subject_sessions(subject_dir, benchmark_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference,
            curated_only=config.curated_only,
        )
        rows.extend(
            _audit_subject(
                subject_dir.name,
                sessions,
                reference_matrix,
                config,
            )
        )
    return rows


def _run_calibrated_loso_registration_qa_report(
    config: RegistrationQAConfig,
) -> list[dict[str, Any]]:
    """Return held-out edge diagnostics for LOSO calibrated costs."""

    if config.transform_type == "gt-affine-oracle":
        raise ValueError(
            "transform_type='gt-affine-oracle' is a manual-GT registration QA "
            "oracle and is not available for calibrated LOSO training."
        )

    from bayescatrack.experiments.track2p_loso_calibration import (
        _collect_training_examples,
        _load_subject_calibration_data,
        _loso_logistic_model_kwargs,
        _training_sample_weight,
    )

    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("Calibrated edge QA requires at least two subject directories")

    benchmark_config = _benchmark_config(config)
    subjects = tuple(
        _load_subject_calibration_data(subject_dir, config=benchmark_config)
        for subject_dir in subject_dirs
    )
    feature_names = tuple(DEFAULT_ASSOCIATION_FEATURES)
    logistic_model_kwargs = _loso_logistic_model_kwargs(None)
    rows: list[dict[str, Any]] = []

    for held_out_index, held_out in enumerate(subjects):
        if config.progress:
            print(
                f"registration-qa calibrated LOSO: {held_out.subject_name}",
                file=sys.stderr,
                flush=True,
            )
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        training_features, training_labels = _collect_training_examples(
            training_subjects,
            config=benchmark_config,
            feature_names=feature_names,
            progress=None,
            held_out_subject=held_out.subject_name,
        )
        weights = _training_sample_weight(
            training_labels,
            sample_weight=None,
            strategy="none",
        )
        calibrated_model = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kwargs=logistic_model_kwargs,
        )
        reference_matrix = _reference_matrix(
            held_out.reference,
            curated_only=config.curated_only,
        )
        subject_rows = _audit_subject(
            held_out.subject_name,
            held_out.sessions,
            reference_matrix,
            config,
            calibrated_model=calibrated_model,
        )
        for row in subject_rows:
            row["calibration_training_subjects"] = ",".join(
                subject.subject_name for subject in training_subjects
            )
            row["calibration_training_examples"] = int(training_labels.shape[0])
            row["calibration_positive_examples"] = int(np.sum(training_labels))
            row["calibration_negative_examples"] = int(
                training_labels.shape[0] - np.sum(training_labels)
            )
            row["calibration_sample_weight_strategy"] = "none"
            row["calibration_class_weight"] = "None"
        rows.extend(subject_rows)
    return rows


def summarize_registration_qa_links(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate link-level QA rows by subject and session edge."""

    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row.get("cost", "")),
            str(row["subject"]),
            str(row["source_session_name"]),
            str(row["target_session_name"]),
            int(row["session_gap"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (cost, subject, source_name, target_name, session_gap), group in sorted(
        grouped.items()
    ):
        summary.append(
            {
                "cost": cost,
                "subject": subject,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "session_gap": session_gap,
                "n_gt_links": len(group),
                "registration_backend": _mode(group, "registration_backend"),
                "registered_plane_source": _mode(group, "registered_plane_source"),
                "registration_backend_reason": _mode(
                    group,
                    "registration_backend_reason",
                ),
                "transform_type": _mode(group, "transform_type"),
                "median_registered_iou": _stat(group, "registered_iou"),
                "p10_registered_iou": _stat(group, "registered_iou", 10),
                "p90_registered_iou": _stat(group, "registered_iou", 90),
                "median_registered_centroid_distance": _stat(
                    group,
                    "registered_centroid_distance",
                ),
                "p90_registered_centroid_distance": _stat(
                    group,
                    "registered_centroid_distance",
                    90,
                ),
                "empty_registered_rois": int(
                    max(int(row["empty_registered_rois"]) for row in group)
                ),
                "empty_registered_fraction": float(
                    max(float(row["empty_registered_fraction"]) for row in group)
                ),
                "gt_top1_rate": _mean_bool(group, "gt_is_top1"),
                "gt_recall_at_1": _mean_bool(group, "gt_is_top1"),
                "gt_recall_at_5": _mean_bool(group, "gt_is_top5"),
                "gt_recall_at_10": _mean_bool(group, "gt_is_top10"),
                "gt_admissible_rate": _mean_bool(group, "gt_candidate_admissible"),
                "empty_gt_mask_rate": _mean_bool(group, "target_empty_registered_mask"),
                "gated_gt_rate": _mean_bool(group, "target_gated"),
                "median_gt_rank": _stat(group, "gt_rank"),
                "p90_gt_rank": _stat(group, "gt_rank", 90),
                "median_gt_probability": _stat(group, "gt_probability"),
                "p10_gt_probability": _stat(group, "gt_probability", 10),
                "p90_gt_probability": _stat(group, "gt_probability", 90),
                "median_gt_cost_percentile": _stat(group, "gt_cost_percentile"),
                "p90_gt_cost_percentile": _stat(group, "gt_cost_percentile", 90),
                "median_candidate_count": _stat(group, "candidate_count"),
                "median_finite_candidate_count": _stat(group, "finite_candidate_count"),
                "median_finite_false_candidate_count": _stat(
                    group,
                    "finite_false_candidate_count",
                ),
                "median_false_cost_min": _stat(group, "false_cost_min"),
                "median_false_cost_p10": _stat(group, "false_cost_p10"),
                "median_false_cost_median": _stat(group, "false_cost_median"),
                "median_false_cost_p90": _stat(group, "false_cost_p90"),
                "median_cost_margin": _stat(group, "cost_margin"),
            }
        )
    return summary


def format_registration_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format summary rows as a compact Markdown table."""

    columns = [
        "cost",
        "subject",
        "source_session_name",
        "target_session_name",
        "n_gt_links",
        "registration_backend",
        "median_registered_iou",
        "median_registered_centroid_distance",
        "gt_recall_at_1",
        "gt_recall_at_5",
        "gt_recall_at_10",
        "gt_admissible_rate",
        "empty_registered_rois",
        "median_gt_rank",
        "median_gt_probability",
        "median_gt_cost_percentile",
        "median_false_cost_median",
        "median_cost_margin",
    ]
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append(
            "| " + " | ".join(_format_value(row.get(col, "")) for col in columns) + " |"
        )
    return "\n".join(body)


def summarize_registration_backend_usage(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize which registration backend was actually used per audited edge."""

    edge_groups: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        edge_key = (
            str(row.get("cost", "")),
            str(row["subject"]),
            int(row["source_session_index"]),
            int(row["target_session_index"]),
        )
        edge_groups[edge_key].append(row)

    backend_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for edge_rows in edge_groups.values():
        representative = edge_rows[0]
        group_key = (
            str(representative.get("cost", "")),
            str(representative["registration_backend"]),
            str(representative["transform_type"]),
            str(representative.get("registered_plane_source", "")),
            str(representative.get("registration_backend_reason", "")),
        )
        backend_groups[group_key].append(
            {
                **dict(representative),
                "gt_link_rows": len(edge_rows),
            }
        )

    summary: list[dict[str, Any]] = []
    for (
        cost,
        registration_backend,
        transform_type,
        registered_plane_source,
        registration_backend_reason,
    ), backend_edge_rows in sorted(backend_groups.items()):
        subjects = sorted({str(row["subject"]) for row in backend_edge_rows})
        summary.append(
            {
                "cost": cost,
                "registration_backend": registration_backend,
                "transform_type": transform_type,
                "registered_plane_source": registered_plane_source,
                "registration_backend_reason": registration_backend_reason,
                "edge_count": len(backend_edge_rows),
                "gt_link_rows": int(
                    sum(int(row["gt_link_rows"]) for row in backend_edge_rows)
                ),
                "subject_count": len(subjects),
                "subjects": ",".join(subjects),
                "median_fov_translation_shift_y": _stat(
                    backend_edge_rows,
                    "fov_translation_shift_y",
                ),
                "median_fov_translation_shift_x": _stat(
                    backend_edge_rows,
                    "fov_translation_shift_x",
                ),
                "median_fov_translation_peak_correlation": _stat(
                    backend_edge_rows,
                    "fov_translation_peak_correlation",
                ),
            }
        )
    return summary


def format_registration_backend_audit_table(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Format registration-backend usage rows as a compact Markdown table."""

    columns = [
        "cost",
        "registration_backend",
        "transform_type",
        "registered_plane_source",
        "edge_count",
        "gt_link_rows",
        "subject_count",
        "subjects",
        "median_fov_translation_shift_y",
        "median_fov_translation_shift_x",
        "median_fov_translation_peak_correlation",
        "registration_backend_reason",
    ]
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append(
            "| " + " | ".join(_format_value(row.get(col, "")) for col in columns) + " |"
        )
    return "\n".join(body)


def write_registration_backend_audit_results(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write registration-backend audit rows as JSON, CSV, or Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(
        format_registration_backend_audit_table(rows) + "\n",
        encoding="utf-8",
    )


def write_registration_qa_results(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write registration QA rows as JSON, CSV, or Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_registration_qa_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark registration-qa",
        description="Report registration quality on manual-GT Track2p links.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "gt-affine-oracle", "none"),
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware", "roi-aware-shifted", "calibrated"),
    )
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument(
        "--level", default="summary", choices=("summary", "links", "backend-audit")
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", default="table", choices=("table", "json", "csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows: Sequence[Mapping[str, Any]] = run_registration_qa_report(
        _config_from_args(args)
    )
    if args.level == "summary":
        rows = summarize_registration_qa_links(rows)
    elif args.level == "backend-audit":
        rows = summarize_registration_backend_usage(rows)

    if args.output is not None:
        if args.level == "backend-audit":
            write_registration_backend_audit_results(rows, args.output, args.format)
        else:
            write_registration_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(list(rows), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        if args.level == "backend-audit":
            print(format_registration_backend_audit_table(rows))
        else:
            print(format_registration_qa_table(rows))
    return 0


def _audit_subject(
    subject: str,
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    config: RegistrationQAConfig,
    calibrated_model: CalibratedAssociationModel | None = None,
) -> list[dict[str, Any]]:
    if config.cost == "calibrated" and calibrated_model is None:
        raise ValueError("calibrated_model is required when cost='calibrated'")

    rows: list[dict[str, Any]] = []
    for source_index, target_index in session_edge_pairs(
        len(sessions), max_gap=config.max_gap
    ):
        reference_session = sessions[source_index]
        target_session = sessions[target_index]
        linked_source_rois = _linked_source_rois(
            reference_matrix,
            source_index,
            target_index,
        )
        if not linked_source_rois:
            continue
        cost_reference_session, cost_source_lookup = _subset_reference_session(
            reference_session,
            linked_source_rois,
        )
        registered_plane = _register_plane_pair_for_registration_qa(
            reference_session,
            target_session,
            reference_matrix,
            source_index,
            target_index,
            config,
        )
        registration_metadata = _registration_metadata(
            config.transform_type,
            registered_plane,
        )
        registered_plane, empty_registered_rois = replace_empty_registered_masks(
            registered_plane
        )
        raw_components = _raw_pairwise_components(
            cost_reference_session.plane_data,
            target_session.plane_data,
            config,
        )
        registered_bundle = _association_bundle(
            cost_reference_session,
            target_session,
            registered_plane,
            config,
        )
        probability_matrix = None
        if config.cost == "calibrated":
            assert calibrated_model is not None
            add_activity_similarity_components(
                registered_bundle.pairwise_components,
                cost_reference_session.plane_data,
                registered_plane,
            )
            cost_matrix = calibrated_model.pairwise_cost_matrix_from_bundle(
                registered_bundle,
                session_gap=target_index - source_index,
            )
            probability_matrix = (
                calibrated_model.pairwise_probability_matrix_from_bundle(
                    registered_bundle,
                    session_gap=target_index - source_index,
                )
            )
        else:
            cost_matrix = np.asarray(
                registered_bundle.pairwise_cost_matrix, dtype=float
            )
        rows.extend(
            _audit_reference_links(
                subject,
                source_index,
                target_index,
                reference_session,
                target_session,
                reference_matrix,
                cost_source_lookup,
                raw_components,
                registered_bundle.pairwise_components,
                np.asarray(cost_matrix, dtype=float),
                (
                    None
                    if probability_matrix is None
                    else np.asarray(probability_matrix, dtype=float)
                ),
                empty_registered_rois,
                registration_metadata,
                config,
            )
        )
    return rows


def _audit_reference_links(
    subject: str,
    source_index: int,
    target_index: int,
    source_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    cost_source_lookup: Mapping[int, int],
    raw_components: Mapping[str, np.ndarray],
    registered_components: Mapping[str, np.ndarray],
    cost_matrix: np.ndarray,
    probability_matrix: np.ndarray | None,
    empty_registered_rois: np.ndarray,
    registration_metadata: Mapping[str, Any],
    config: RegistrationQAConfig,
) -> list[dict[str, Any]]:
    source_lookup = _roi_lookup(source_session)
    target_lookup = _roi_lookup(target_session)
    target_roi_indices = _roi_indices(target_session)
    rows: list[dict[str, Any]] = []
    for track_index, track in enumerate(reference_matrix):
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_roi_int = int(source_roi)
        target_roi_int = int(target_roi)
        source_present = source_roi_int in source_lookup
        target_present = target_roi_int in target_lookup
        if not source_present or not target_present:
            continue
        source_local = cost_source_lookup[source_roi_int]
        target_local = target_lookup[target_roi_int]
        cost_row = cost_matrix[source_local]
        gt_cost = float(cost_row[target_local])
        probability_row = (
            None if probability_matrix is None else probability_matrix[source_local]
        )
        gt_probability = (
            np.nan if probability_row is None else float(probability_row[target_local])
        )
        finite_costs = cost_row[np.isfinite(cost_row)]
        gt_rank = int(1 + np.count_nonzero(cost_row < gt_cost))
        best_target_local = int(np.nanargmin(cost_row))
        false_costs: np.ndarray = np.delete(cost_row, target_local)
        finite_false_costs = false_costs[np.isfinite(false_costs)]
        false_cost_min = _array_stat(finite_false_costs, "min")
        gt_cost_percentile = (
            float(
                100.0
                * np.count_nonzero(finite_costs < gt_cost)
                / max(int(finite_costs.size) - 1, 1)
            )
            if np.isfinite(gt_cost) and finite_costs.size
            else np.nan
        )
        target_empty = bool(empty_registered_rois[target_local])
        target_gated = bool(
            _component_value(
                registered_components, "gated", source_local, target_local, False
            )
        )
        below_threshold = (
            True
            if config.cost_threshold is None
            else bool(gt_cost <= float(config.cost_threshold))
        )
        rows.append(
            {
                "cost": config.cost,
                "subject": subject,
                "source_session_index": source_index,
                "target_session_index": target_index,
                "source_session_name": source_session.session_name,
                "target_session_name": target_session.session_name,
                "session_gap": target_index - source_index,
                "track_index": track_index,
                "registration_backend": registration_metadata["registration_backend"],
                "registered_plane_source": registration_metadata[
                    "registered_plane_source"
                ],
                "registration_backend_reason": registration_metadata[
                    "registration_backend_reason"
                ],
                "fov_translation_shift_y": registration_metadata[
                    "fov_translation_shift_y"
                ],
                "fov_translation_shift_x": registration_metadata[
                    "fov_translation_shift_x"
                ],
                "fov_translation_peak_correlation": registration_metadata[
                    "fov_translation_peak_correlation"
                ],
                "transform_type": config.transform_type,
                "source_roi": source_roi_int,
                "target_roi": target_roi_int,
                "source_roi_present": source_present,
                "target_roi_present": target_present,
                "raw_mask_shape_matches": (
                    source_session.plane_data.image_shape
                    == target_session.plane_data.image_shape
                ),
                "raw_iou": _component_value(
                    raw_components, "iou", source_local, target_local
                ),
                "registered_iou": _component_value(
                    registered_components,
                    "iou",
                    source_local,
                    target_local,
                ),
                "raw_centroid_distance": _component_value(
                    raw_components,
                    "centroid_distance",
                    source_local,
                    target_local,
                ),
                "registered_centroid_distance": _component_value(
                    registered_components,
                    "centroid_distance",
                    source_local,
                    target_local,
                ),
                "gt_cost": gt_cost,
                "gt_probability": gt_probability,
                "gt_rank": gt_rank,
                "gt_is_top1": gt_rank == 1,
                "gt_is_top5": gt_rank <= 5,
                "gt_is_top10": gt_rank <= 10,
                "gt_cost_percentile": gt_cost_percentile,
                "candidate_count": int(cost_row.size),
                "finite_candidate_count": int(finite_costs.size),
                "false_candidate_count": int(false_costs.size),
                "finite_false_candidate_count": int(finite_false_costs.size),
                "best_target_roi": int(target_roi_indices[best_target_local]),
                "best_cost": float(cost_row[best_target_local]),
                "best_false_cost": false_cost_min,
                "false_cost_min": false_cost_min,
                "false_cost_p10": _array_stat(finite_false_costs, 10),
                "false_cost_median": _array_stat(finite_false_costs, 50),
                "false_cost_p90": _array_stat(finite_false_costs, 90),
                "cost_margin": (
                    false_cost_min - gt_cost if np.isfinite(false_cost_min) else np.nan
                ),
                "target_empty_registered_mask": target_empty,
                "target_gated": target_gated,
                "target_below_cost_threshold": below_threshold,
                "gt_candidate_admissible": (
                    (not target_empty) and (not target_gated) and below_threshold
                ),
                "empty_registered_rois": int(np.count_nonzero(empty_registered_rois)),
                "empty_registered_fraction": (
                    float(np.mean(empty_registered_rois))
                    if empty_registered_rois.size
                    else 0.0
                ),
            }
        )
    return rows


def _linked_source_rois(
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
) -> tuple[int, ...]:
    linked_rois: list[int] = []
    seen: set[int] = set()
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_roi_int = int(source_roi)
        if source_roi_int in seen:
            continue
        seen.add(source_roi_int)
        linked_rois.append(source_roi_int)
    return tuple(linked_rois)


def _register_plane_pair_for_registration_qa(
    reference_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    config: RegistrationQAConfig,
) -> CalciumPlaneData:
    if config.transform_type == "gt-affine-oracle":
        return _gt_affine_oracle_registered_plane(
            reference_session,
            target_session,
            reference_matrix,
            source_index,
            target_index,
            weighted_centroids=config.weighted_centroids,
        )
    return register_plane_pair(
        reference_session.plane_data,
        target_session.plane_data,
        transform_type=config.transform_type,
    )


def _gt_affine_oracle_registered_plane(
    reference_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    *,
    weighted_centroids: bool,
) -> CalciumPlaneData:
    reference_points, target_points = _gt_affine_oracle_points(
        reference_session,
        target_session,
        reference_matrix,
        source_index,
        target_index,
        weighted_centroids=weighted_centroids,
    )
    transform_yx = _fit_affine_yx_transform(
        moving_points_yx=target_points,
        reference_points_yx=reference_points,
    )
    target_plane = target_session.plane_data
    registered_masks = _warp_mask_stack_nearest(
        target_plane.roi_masks,
        moving_to_reference_yx=transform_yx,
        output_shape=reference_session.plane_data.image_shape,
    )
    registered_fov = (
        None
        if target_plane.fov is None
        else _warp_image_nearest(
            target_plane.fov,
            moving_to_reference_yx=transform_yx,
            output_shape=reference_session.plane_data.image_shape,
        )
    )
    ops = {} if target_plane.ops is None else dict(target_plane.ops)
    ops.update(
        {
            "registration_backend": "gt-affine-oracle",
            "registration_transform_type": "gt-affine-oracle",
            "registration_backend_reason": (
                "manual-GT affine oracle fit from linked ROI centroids"
            ),
            "gt_affine_oracle_link_count": int(reference_points.shape[0]),
            "gt_affine_oracle_matrix_yx": transform_yx,
        }
    )
    return target_plane.with_replaced_masks(
        registered_masks,
        fov=registered_fov,
        source=f"{target_plane.source}_gt_affine_oracle",
        ops=ops,
    )


def _gt_affine_oracle_points(
    reference_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    *,
    weighted_centroids: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source_lookup = _roi_lookup(reference_session)
    target_lookup = _roi_lookup(target_session)
    source_centroids = reference_session.plane_data.centroids(
        order="yx",
        weighted=weighted_centroids,
    ).T
    target_centroids = target_session.plane_data.centroids(
        order="yx",
        weighted=weighted_centroids,
    ).T
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_roi_int = int(source_roi)
        target_roi_int = int(target_roi)
        if source_roi_int not in source_lookup or target_roi_int not in target_lookup:
            continue
        source_points.append(source_centroids[source_lookup[source_roi_int]])
        target_points.append(target_centroids[target_lookup[target_roi_int]])
    if len(source_points) < 3:
        raise ValueError(
            "transform_type='gt-affine-oracle' requires at least three present "
            "manual-GT links on each audited session edge"
        )
    return np.asarray(source_points, dtype=float), np.asarray(
        target_points, dtype=float
    )


def _fit_affine_yx_transform(
    *,
    moving_points_yx: np.ndarray,
    reference_points_yx: np.ndarray,
) -> np.ndarray:
    if moving_points_yx.shape != reference_points_yx.shape:
        raise ValueError("Affine oracle point arrays must have the same shape")
    if moving_points_yx.ndim != 2 or moving_points_yx.shape[1] != 2:
        raise ValueError("Affine oracle points must have shape (n_points, 2)")
    design = np.column_stack(
        (
            moving_points_yx[:, 0],
            moving_points_yx[:, 1],
            np.ones(moving_points_yx.shape[0]),
        )
    )
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(
            "transform_type='gt-affine-oracle' requires at least three "
            "non-collinear manual-GT link centroids per audited session edge"
        )
    coefficients, *_ = np.linalg.lstsq(design, reference_points_yx, rcond=None)
    return np.asarray(
        [
            [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
            [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _warp_mask_stack_nearest(
    masks: np.ndarray,
    *,
    moving_to_reference_yx: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    mask_array = np.asarray(masks)
    if mask_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    if mask_array.shape[0] == 0:
        return np.zeros((0, output_shape[0], output_shape[1]), dtype=mask_array.dtype)
    return np.stack(
        [
            _warp_image_nearest(
                mask,
                moving_to_reference_yx=moving_to_reference_yx,
                output_shape=output_shape,
            )
            for mask in mask_array
        ],
        axis=0,
    )


def _warp_image_nearest(
    image: np.ndarray,
    *,
    moving_to_reference_yx: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("Images must have shape (height, width)")
    reference_y: np.ndarray
    reference_x: np.ndarray
    reference_y, reference_x = np.indices(output_shape, dtype=float)
    reference_homogeneous = np.vstack(
        (
            reference_y.ravel(),
            reference_x.ravel(),
            np.ones(reference_y.size, dtype=float),
        )
    )
    reference_to_moving_yx = np.linalg.inv(moving_to_reference_yx)
    moving_homogeneous = reference_to_moving_yx @ reference_homogeneous
    denominator = moving_homogeneous[2]
    valid_denominator = np.abs(denominator) > np.spacing(1.0)
    moving_y: np.ndarray = np.full(reference_y.size, -1, dtype=int)
    moving_x: np.ndarray = np.full(reference_x.size, -1, dtype=int)
    moving_y[valid_denominator] = np.rint(
        moving_homogeneous[0, valid_denominator] / denominator[valid_denominator]
    ).astype(int)
    moving_x[valid_denominator] = np.rint(
        moving_homogeneous[1, valid_denominator] / denominator[valid_denominator]
    ).astype(int)
    valid = (
        valid_denominator
        & (moving_y >= 0)
        & (moving_y < image_array.shape[0])
        & (moving_x >= 0)
        & (moving_x < image_array.shape[1])
    )
    output = np.zeros((output_shape[0] * output_shape[1],), dtype=image_array.dtype)
    output[valid] = image_array[moving_y[valid], moving_x[valid]]
    return output.reshape(output_shape)


def _subset_reference_session(
    session: Track2pSession,
    roi_values: Sequence[int],
) -> tuple[Track2pSession, dict[int, int]]:
    roi_lookup = _roi_lookup(session)
    local_indices = np.asarray(
        [roi_lookup[int(roi_value)] for roi_value in roi_values],
        dtype=int,
    )
    subset_plane = _subset_plane(session.plane_data, local_indices)
    subset_session = Track2pSession(
        session_dir=session.session_dir,
        session_name=session.session_name,
        session_date=session.session_date,
        plane_data=subset_plane,
        motion_energy=session.motion_energy,
    )
    return subset_session, {
        int(roi_value): subset_index
        for subset_index, roi_value in enumerate(roi_values)
    }


def _subset_plane(
    plane: CalciumPlaneData,
    local_indices: np.ndarray,
) -> CalciumPlaneData:
    return CalciumPlaneData(
        roi_masks=np.asarray(plane.roi_masks)[local_indices],
        traces=_slice_optional_roi_array(plane.traces, local_indices),
        fov=plane.fov,
        spike_traces=_slice_optional_roi_array(plane.spike_traces, local_indices),
        neuropil_traces=_slice_optional_roi_array(
            plane.neuropil_traces,
            local_indices,
        ),
        cell_probabilities=_slice_optional_roi_array(
            plane.cell_probabilities,
            local_indices,
        ),
        roi_indices=(
            None
            if plane.roi_indices is None
            else np.asarray(plane.roi_indices, dtype=int)[local_indices]
        ),
        roi_features={
            key: np.asarray(value)[local_indices]
            for key, value in plane.roi_features.items()
        },
        source=plane.source,
        plane_name=plane.plane_name,
        ops=plane.ops,
    )


def _slice_optional_roi_array(
    value: np.ndarray | None,
    local_indices: np.ndarray,
) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)[local_indices]


def _association_bundle(
    source_session: Track2pSession,
    target_session: Track2pSession,
    target_plane: Any,
    config: RegistrationQAConfig,
) -> Any:
    cost_kwargs = _cost_kwargs(config)
    previous_pairwise_cost_method = None
    if pairwise_kwargs_use_shifted_overlap(cost_kwargs):
        previous_pairwise_cost_method = install_shifted_overlap_cost_patch()
    try:
        return build_session_pair_association_bundle(
            source_session,
            target_session,
            measurement_plane_in_reference_frame=target_plane,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=cost_kwargs,
            return_pairwise_components=True,
        )
    finally:
        if previous_pairwise_cost_method is not None:
            CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
                previous_pairwise_cost_method
            )


def _raw_pairwise_components(
    reference_plane: CalciumPlaneData,
    target_plane: CalciumPlaneData,
    config: RegistrationQAConfig,
) -> dict[str, np.ndarray]:
    component_shape = (reference_plane.n_rois, target_plane.n_rois)
    components: dict[str, np.ndarray] = {
        "centroid_distance": reference_plane.pairwise_centroid_distances(
            target_plane,
            order=config.order,
            weighted=config.weighted_centroids,
        )
    }
    if reference_plane.image_shape != target_plane.image_shape:
        components["iou"] = np.full(component_shape, np.nan, dtype=float)
        return components

    cost_kwargs = _cost_kwargs(config)
    previous_pairwise_cost_method = None
    if pairwise_kwargs_use_shifted_overlap(cost_kwargs):
        previous_pairwise_cost_method = install_shifted_overlap_cost_patch()
    try:
        _, raw_components = reference_plane.build_pairwise_cost_matrix(
            target_plane,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            return_components=True,
            **cost_kwargs,
        )
    finally:
        if previous_pairwise_cost_method is not None:
            CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
                previous_pairwise_cost_method
            )
    components.update(raw_components)
    return components


def _cost_kwargs(config: RegistrationQAConfig) -> dict[str, Any]:
    if config.cost == "registered-iou":
        kwargs = registered_iou_cost_kwargs()
    elif config.cost == "roi-aware-shifted":
        kwargs = roi_aware_shifted_cost_kwargs()
    else:
        kwargs = roi_aware_cost_kwargs()
    kwargs.update(config.pairwise_cost_kwargs or {})
    return kwargs


def _benchmark_config(config: RegistrationQAConfig) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=config.data,
        method="global-assignment",
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=config.reference,
        reference_kind=config.reference_kind,
        allow_track2p_as_reference_for_smoke_test=config.allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        cost=config.cost,
        max_gap=config.max_gap,
        transform_type=config.transform_type,
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


def _config_from_args(args: argparse.Namespace) -> RegistrationQAConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        pairwise_cost_kwargs = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(pairwise_cost_kwargs, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    return RegistrationQAConfig(
        data=args.data,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        plane_name=args.plane_name,
        input_format=args.input_format,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        cost=args.cost,
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
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )


def _roi_lookup(session: Track2pSession) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(_roi_indices(session))}


def _roi_indices(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is None:
        return np.arange(plane.n_rois, dtype=int)
    return np.asarray(plane.roi_indices, dtype=int)


def _component_value(
    components: Mapping[str, np.ndarray],
    key: str,
    row: int,
    column: int,
    default: Any = np.nan,
) -> Any:
    if key not in components:
        return default
    return np.asarray(components[key])[row, column].item()


def _registration_metadata(
    transform_type: str,
    registered_plane: CalciumPlaneData,
) -> dict[str, Any]:
    ops = {} if registered_plane.ops is None else dict(registered_plane.ops)
    source = str(registered_plane.source)
    backend = str(
        ops.get("registration_backend") or _registration_backend(transform_type, source)
    )
    return {
        "registration_backend": backend,
        "registered_plane_source": source,
        "registration_backend_reason": str(
            ops.get("registration_backend_reason")
            or _registration_backend_reason(transform_type, backend, source)
        ),
        "fov_translation_shift_y": _fov_translation_shift_component(ops, 0),
        "fov_translation_shift_x": _fov_translation_shift_component(ops, 1),
        "fov_translation_peak_correlation": _float_ops_value(
            ops,
            "fov_registration_peak_correlation",
        ),
    }


def _registration_backend(transform_type: str, source: str) -> str:
    if transform_type == "none":
        return "none"
    if transform_type == "gt-affine-oracle":
        return "gt-affine-oracle"
    if "fov_registered" in source:
        return "fov-translation"
    if "registered" in source:
        return "track2p-elastix"
    return "unknown"


def _registration_backend_reason(transform_type: str, backend: str, source: str) -> str:
    if transform_type == "none":
        return "transform_type=none"
    if transform_type == "gt-affine-oracle":
        return "manual-GT affine oracle fit from linked ROI centroids"
    if backend == "fov-translation":
        return "registered plane source contains 'fov_registered'"
    if backend == "track2p-elastix":
        return "registered plane source contains 'registered' without 'fov_registered'"
    return (
        f"could not infer registration backend from registered plane source {source!r}"
    )


def _fov_translation_shift_component(
    ops: Mapping[str, Any],
    index: int,
) -> float:
    shift = ops.get("fov_registration_measurement_to_reference_shift_yx")
    if shift is None:
        return np.nan
    shift_array = np.asarray(shift, dtype=float).reshape(-1)
    if shift_array.size <= index:
        return np.nan
    return float(shift_array[index])


def _float_ops_value(ops: Mapping[str, Any], key: str) -> float:
    if key not in ops:
        return np.nan
    try:
        return float(ops[key])
    except (TypeError, ValueError):
        return np.nan


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _array_stat(values: np.ndarray, statistic: float | Literal["min"]) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if not finite_values.size:
        return np.nan
    if statistic == "min":
        return float(np.min(finite_values))
    return float(np.percentile(finite_values, statistic))


def _stat(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    percentile: float | None = None,
) -> float:
    values = _finite_values(rows, key)
    if not values.size:
        return np.nan
    if percentile is None:
        return float(np.median(values))
    return float(np.percentile(values, percentile))


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return np.nan
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def _mode(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    if not counts:
        return ""
    return max(counts, key=lambda value: counts[value])


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
