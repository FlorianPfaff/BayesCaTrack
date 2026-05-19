"""Leave-one-subject-out monotone-ranking Track2p calibration benchmark."""

# jscpd:ignore-start
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.activity_similarity import ACTIVITY_TIEBREAKER_FEATURES
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    collect_reference_pairwise_example_blocks,
    collect_reference_training_examples,
    SPLIT_ROI_STAT_FEATURES,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _score_prediction_against_reference,
    discover_subject_dirs,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SubjectCalibrationData,
    _load_subject_calibration_data,
    _reference_training_options,
)

MonotoneFeatureSetName = Literal[
    "default",
    "legacy",
    "split-roi",
    "local-evidence",
    "activity-tiebreaker",
    "full",
]
MONOTONE_FEATURE_SET_NAMES: tuple[str, ...] = (
    "default",
    "legacy",
    "split-roi",
    "local-evidence",
    "activity-tiebreaker",
    "full",
)

_LEGACY_ASSOCIATION_FEATURES = (
    "centroid_distance",
    "mahalanobis_centroid_distance",
    "one_minus_iou",
    "one_minus_mask_cosine",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity_cost",
    "activity_similarity_available",
    "session_gap",
)


def monotone_feature_names_for_set(feature_set: str) -> tuple[str, ...]:
    """Return the calibrated feature schema for one monotone LOSO ablation set.

    The default calibrated schema already includes the split Suite2p ROI-stat
    patch when the package is imported.  The explicit ablation names below make
    that behavior reproducible and add opt-in local-evidence and activity
    tie-breaker schemas for the learned ranking model.
    """

    feature_set = _validate_feature_set(feature_set)
    if feature_set == "default":
        return tuple(DEFAULT_ASSOCIATION_FEATURES)
    if feature_set == "legacy":
        return _LEGACY_ASSOCIATION_FEATURES

    split_features = _replace_legacy_roi_scalar_with_split_features(
        _LEGACY_ASSOCIATION_FEATURES
    )
    if feature_set == "split-roi":
        return split_features
    if feature_set == "local-evidence":
        return _deduplicate_features(
            (*split_features, *LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
        )
    if feature_set == "activity-tiebreaker":
        return _deduplicate_features((*split_features, *ACTIVITY_TIEBREAKER_FEATURES))
    if feature_set == "full":
        return _deduplicate_features(
            (
                *split_features,
                *LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
                *ACTIVITY_TIEBREAKER_FEATURES,
            )
        )
    raise AssertionError(f"Unhandled monotone feature set: {feature_set!r}")


def config_with_feature_dependencies(
    config: Track2pBenchmarkConfig, feature_names: Sequence[str]
) -> Track2pBenchmarkConfig:
    """Enable pairwise component production required by selected features."""

    if not any(name in LOCAL_EVIDENCE_ASSOCIATION_FEATURES for name in feature_names):
        return config
    pairwise_cost_kwargs = dict(config.pairwise_cost_kwargs or {})
    pairwise_cost_kwargs["local_evidence_components"] = True
    return replace(config, pairwise_cost_kwargs=pairwise_cost_kwargs)


def monotone_options_for_feature_schema(
    options: MonotoneRankerOptions, feature_names: Sequence[str]
) -> MonotoneRankerOptions:
    """Use all selected cost-like features unless the caller supplied a subset."""

    if options.monotone_feature_names:
        return options
    return replace(
        options,
        monotone_feature_names=_default_monotone_feature_names(feature_names),
    )


def _validate_feature_set(feature_set: str) -> MonotoneFeatureSetName:
    if feature_set not in MONOTONE_FEATURE_SET_NAMES:
        raise ValueError(
            "feature_set must be one of " + ", ".join(MONOTONE_FEATURE_SET_NAMES)
        )
    return cast(MonotoneFeatureSetName, feature_set)


def _replace_legacy_roi_scalar_with_split_features(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    split_features = tuple(SPLIT_ROI_STAT_FEATURES)
    if not split_features:
        return tuple(feature_names)
    expanded: list[str] = []
    for feature_name in feature_names:
        if feature_name == "roi_feature_cost":
            expanded.extend(split_features)
        else:
            expanded.append(feature_name)
    return _deduplicate_features(expanded)


def _deduplicate_features(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(name) for name in feature_names))


def _default_monotone_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in feature_names if not str(name).endswith("_available"))


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_monotone_loso_calibration(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] | None = None,
    feature_set: MonotoneFeatureSetName = "full",
    monotone_options: MonotoneRankerOptions | None = None,
) -> LosoCalibrationResult:
    """Run LOSO global assignment with a monotone hard-negative ranking model."""

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "Monotone LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    feature_set_label = "custom" if feature_names is not None else feature_set
    feature_names = tuple(
        feature_names
        if feature_names is not None
        else monotone_feature_names_for_set(feature_set)
    )
    config = config_with_feature_dependencies(config, feature_names)
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO calibration requires at least two subject directories")

    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 2),
        enabled=config.progress,
        label="monotone-loso",
    )
    subjects: list[SubjectCalibrationData] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(_load_subject_calibration_data(subject_dir, config=config))

    options = monotone_options_for_feature_schema(
        monotone_options or MonotoneRankerOptions(), feature_names
    )
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        progress.step(f"collecting training blocks for {held_out.subject_name}")
        training_blocks = _collect_training_blocks(
            training_subjects,
            config=config,
            feature_names=feature_names,
        )
        progress.step(f"fitting monotone ranker for {held_out.subject_name}")
        calibrated_model = fit_monotone_ranking_association_model_from_blocks(
            training_blocks,
            options=options,
        )
        progress.step(f"scoring calibration for {held_out.subject_name}")
        calibration_scores = _score_holdout_calibration(
            calibrated_model,
            held_out,
            config=config,
            feature_names=feature_names,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=calibrated_model,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, held_out.reference, config=config
        )
        positives = int(calibrated_model.n_positive_examples)
        training_examples = int(calibrated_model.n_training_examples)
        scores: dict[str, float | int | str] = {
            **base_scores,
            "training_examples": training_examples,
            "positive_examples": positives,
            "negative_examples": int(training_examples - positives),
            "calibration_model": "monotone-ranker",
            "monotone_feature_set": str(feature_set_label),
            "monotone_feature_names": ",".join(calibrated_model.monotone_feature_names),
            "monotone_rank_constraints": int(calibrated_model.n_rank_constraints),
            "monotone_training_rank_loss": float(calibrated_model.training_rank_loss),
            "monotone_training_binary_loss": float(
                calibrated_model.training_binary_loss
            ),
            **_monotone_option_scores(options),
            **calibration_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant="Monotone ranker costs + LOSO global assignment",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=training_examples,
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def _collect_training_blocks(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> tuple[Any, ...]:
    blocks: list[Any] = []
    training_options = _reference_training_options(config, feature_names)
    for subject in training_subjects:
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                subject.sessions,
                subject.reference,
                session_edges=session_edge_pairs(
                    len(subject.sessions), max_gap=config.max_gap
                ),
                options=training_options,
            )
        )
    if not blocks:
        raise ValueError("At least one training block is required")
    return tuple(blocks)


def _score_holdout_calibration(
    calibrated_model: Any,
    held_out: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> dict[str, float | int]:
    features, labels = collect_reference_training_examples(
        held_out.sessions,
        held_out.reference,
        session_edges=session_edge_pairs(
            len(held_out.sessions), max_gap=config.max_gap
        ),
        options=_reference_training_options(config, feature_names),
    )
    probabilities = np.asarray(
        calibrated_model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    return calibration_summary(probabilities, np.asarray(labels).reshape(-1))


def _monotone_option_scores(
    options: MonotoneRankerOptions,
) -> dict[str, float | int | str]:
    values = asdict(options)
    return {
        f"monotone_option_{key}": ",".join(value) if isinstance(value, tuple) else value
        for key, value in values.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the monotone LOSO benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-monotone-loso",
        description="Run Track2p LOSO calibrated global assignment with monotone ranking costs.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
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
        "--feature-set",
        choices=MONOTONE_FEATURE_SET_NAMES,
        default="full",
        help="Calibrated feature ablation set used by the monotone ranker.",
    )
    parser.add_argument(
        "--feature-names",
        default=None,
        help="Comma-separated calibrated feature names; overrides --feature-set.",
    )
    parser.add_argument("--monotone-ranker-kwargs-json", default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    options = _monotone_options_from_args(args)
    feature_names = _feature_names_from_args(args)
    rows = [
        fold.benchmark.to_dict()
        for fold in run_track2p_monotone_loso_calibration(
            config,
            feature_names=feature_names,
            feature_set=args.feature_set,
            monotone_options=options,
        ).folds
    ]
    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost="calibrated",
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
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )


def _monotone_options_from_args(args: argparse.Namespace) -> MonotoneRankerOptions:
    if args.monotone_ranker_kwargs_json is None:
        return MonotoneRankerOptions()
    parsed = json.loads(args.monotone_ranker_kwargs_json)
    if not isinstance(parsed, dict):
        raise ValueError("--monotone-ranker-kwargs-json must decode to a JSON object")
    return MonotoneRankerOptions(**parsed)


def _feature_names_from_args(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.feature_names is None:
        return None
    return tuple(name.strip() for name in args.feature_names.split(",") if name.strip())


def _write_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    from bayescatrack.experiments.track2p_benchmark import format_benchmark_table

    print(format_benchmark_table(rows))


def _csv_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "calibration_model",
        "monotone_rank_constraints",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
# jscpd:ignore-end
