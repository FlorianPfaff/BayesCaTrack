"""Export out-of-fold Track2p association calibration rows."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    CalibratedAssociationModel,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import session_edge_pairs
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    SubjectCalibrationData,
    _collect_training_examples,
    _load_subject_calibration_data,
    _reference_training_options,
)


def export_loso_calibration_csv(
    config: Track2pBenchmarkConfig,
    output_path: Path,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Fit LOSO calibrated models and export held-out pairwise probabilities."""

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "Calibration export requires method='global-assignment' and cost='calibrated'"
        )
    if config.split != "leave-one-subject-out":
        raise ValueError("Calibration export requires split='leave-one-subject-out'")

    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("Calibration export requires at least two subject directories")

    subjects = tuple(
        _load_subject_calibration_data(subject_dir, config=config)
        for subject_dir in subject_dirs
    )
    feature_names = tuple(feature_names)
    rows: list[dict[str, float | int | str]] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        training_features, training_labels = _collect_training_examples(
            training_subjects,
            config=config,
            feature_names=feature_names,
        )
        calibrated_model = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=feature_names,
            sample_weight=sample_weight,
            model_kwargs=model_kwargs,
        )
        rows.extend(
            _held_out_rows(
                held_out,
                config=config,
                feature_names=feature_names,
                calibrated_model=calibrated_model,
                training_subject_names=tuple(
                    subject.subject_name for subject in training_subjects
                ),
            )
        )

    _write_calibration_csv(rows, output_path, feature_names=feature_names)
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.track2p_calibration_export",
        description="Export out-of-fold LOSO calibration rows for Track2p candidate ROI pairs.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="CSV path for calibration rows"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground-truth root or ground_truth.csv",
    )
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--curated-only", action="store_true")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        cost="calibrated",
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
    )
    n_rows = export_loso_calibration_csv(config, args.output)
    print(f"Wrote {n_rows} calibration rows to {args.output}")
    return 0


def _held_out_rows(
    subject: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    calibrated_model: CalibratedAssociationModel,
    training_subject_names: Sequence[str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    options = _reference_training_options(config, feature_names)
    edges = session_edge_pairs(len(subject.sessions), max_gap=config.max_gap)
    for block in collect_reference_pairwise_example_blocks(
        subject.sessions, subject.reference, session_edges=edges, options=options
    ):
        probabilities = _predict_match_probabilities(calibrated_model, block.features)
        link_costs = np.asarray(
            calibrated_model.model.pairwise_cost_matrix(block.features), dtype=float
        )
        labels = np.asarray(block.labels, dtype=int)
        for row_index, reference_roi_index in enumerate(block.reference_roi_indices):
            for column_index, measurement_roi_index in enumerate(
                block.measurement_roi_indices
            ):
                row: dict[str, float | int | str] = {
                    "held_out_subject": subject.subject_name,
                    "training_subjects": ",".join(training_subject_names),
                    "subject": subject.subject_name,
                    "session_a": int(block.session_a),
                    "session_b": int(block.session_b),
                    "session_a_name": subject.sessions[block.session_a].session_name,
                    "session_b_name": subject.sessions[block.session_b].session_name,
                    "session_gap": int(block.gap),
                    "reference_roi_index": int(reference_roi_index),
                    "measurement_roi_index": int(measurement_roi_index),
                    "label": int(labels[row_index, column_index]),
                    "p_same": float(probabilities[row_index, column_index]),
                    "link_cost": float(link_costs[row_index, column_index]),
                }
                for feature_index, feature_name in enumerate(feature_names):
                    row[f"feature_{feature_name}"] = float(
                        block.features[row_index, column_index, feature_index]
                    )
                rows.append(row)
    return rows


def _predict_match_probabilities(
    model: CalibratedAssociationModel, features: np.ndarray
) -> np.ndarray:
    if hasattr(model.model, "predict_match_probability"):
        return np.asarray(model.model.predict_match_probability(features), dtype=float)
    return np.exp(-np.asarray(model.model.pairwise_cost_matrix(features), dtype=float))


def _write_calibration_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: Path,
    *,
    feature_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(feature_names))
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(feature_names: Sequence[str]) -> list[str]:
    return [
        "held_out_subject",
        "training_subjects",
        "subject",
        "session_a",
        "session_b",
        "session_a_name",
        "session_b_name",
        "session_gap",
        "reference_roi_index",
        "measurement_roi_index",
        "label",
        "p_same",
        "link_cost",
        *[f"feature_{feature_name}" for feature_name in feature_names],
    ]


if __name__ == "__main__":
    raise SystemExit(main())
