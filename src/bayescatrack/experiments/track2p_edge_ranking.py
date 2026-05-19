"""Export manual-GT edge-ranking diagnostics for Track2p-style datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferencePairwiseExamples,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    registered_iou_cost_kwargs,
    roi_aware_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.association.shifted_overlap import install_shifted_overlap_cost_patch
from bayescatrack.evaluation.edge_ranking import (
    DEFAULT_HIT_KS,
    ScoreDirection,
    missing_reference_edge_rows,
    rank_labeled_edges,
    score_matrices_from_feature_tensor,
    summarize_edge_ranking_rows,
)
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    collect_candidate_limited_training_examples,
)
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)

DEFAULT_EDGE_RANKING_FEATURES = (
    "pairwise_cost_matrix",
    "iou",
    "iou_cost",
    "centroid_distance",
    "mahalanobis_centroid_distance",
    "mask_cosine_similarity",
    "mask_cosine_cost",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity",
    "activity_similarity_cost",
)

DEFAULT_SHIFTED_EDGE_RANKING_FEATURES = (
    "shifted_iou",
    "shifted_iou_cost",
    "shifted_iou_shift_norm",
    "shifted_mask_cosine_similarity",
    "shifted_mask_cosine_cost",
    "iou_for_cost",
    "mask_cosine_for_cost",
)

DEFAULT_LEARNED_SCORE_FEATURES = DEFAULT_ASSOCIATION_FEATURES

LEARNED_SCORE_MODES = ("logistic-loso", "monotone-loso")

DEFAULT_SIMILARITY_FEATURES = (
    "iou",
    "mask_cosine_similarity",
    "activity_similarity",
    "shifted_iou",
    "shifted_mask_cosine_similarity",
    "iou_for_cost",
    "mask_cosine_for_cost",
    "logistic_loso_probability",
    "monotone_loso_probability",
)

_SHIFTED_OVERLAP_KWARG_NAMES = frozenset(
    {
        "shifted_iou_radius",
        "shifted_iou_weight",
        "shifted_mask_cosine_weight",
        "use_shifted_iou_for_iou_cost",
        "use_shifted_mask_cosine_for_mask_cosine_cost",
        "return_shifted_overlap_components",
    }
)

EDGE_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_a_name",
    "session_b_name",
    "session_gap",
    "reference_roi_index",
    "measurement_roi_index",
    "score_name",
    "score_direction",
    "edge_present",
    "missing_reason",
    "true_score",
    "true_is_finite",
    "row_rank",
    "column_rank",
    "row_better_count",
    "column_better_count",
    "row_tie_count",
    "column_tie_count",
    "row_candidate_count",
    "column_candidate_count",
    "row_finite_candidate_count",
    "column_finite_candidate_count",
    "best_false_row_score",
    "best_false_column_score",
    "best_false_row_roi_index",
    "best_false_column_roi_index",
    "row_margin",
    "column_margin",
]

SUMMARY_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_gap",
    "score_name",
    "gt_edges",
    "present_edges",
    "missing_edges",
    "finite_true_edges",
    "row_hit_at_1",
    "row_hit_at_3",
    "row_hit_at_5",
    "row_hit_at_10",
    "column_hit_at_1",
    "column_hit_at_3",
    "column_hit_at_5",
    "column_hit_at_10",
    "row_hit_at_1_present",
    "row_hit_at_3_present",
    "row_hit_at_5_present",
    "row_hit_at_10_present",
    "column_hit_at_1_present",
    "column_hit_at_3_present",
    "column_hit_at_5_present",
    "column_hit_at_10_present",
    "mutual_top1_rate",
    "mutual_top1_rate_present",
    "median_row_rank",
    "median_column_rank",
    "median_row_margin",
    "median_column_margin",
    "mean_row_margin",
    "mean_column_margin",
    "row_positive_margin_rate",
    "column_positive_margin_rate",
]


def run_track2p_edge_ranking(
    config: Track2pBenchmarkConfig,
    output_path: Path,
    *,
    summary_output_path: Path | None = None,
    feature_names: Sequence[str] = DEFAULT_EDGE_RANKING_FEATURES,
    similarity_features: Sequence[str] = DEFAULT_SIMILARITY_FEATURES,
    hit_ks: Sequence[int] = DEFAULT_HIT_KS,
    learned_score_modes: Sequence[str] = (),
    learned_feature_names: Sequence[str] = DEFAULT_LEARNED_SCORE_FEATURES,
    logistic_model_kwargs: Mapping[str, Any] | None = None,
    monotone_options: MonotoneRankerOptions | None = None,
) -> tuple[int, int]:
    """Export per-edge and per-session-pair edge-ranking diagnostics.

    The detailed CSV contains one row per manual-GT edge per score. The summary
    CSV groups those rows by subject, session pair, gap, and score name.
    """

    learned_score_modes = _normalize_learned_score_modes(learned_score_modes, cost=config.cost)
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    feature_names = tuple(dict.fromkeys(str(feature) for feature in feature_names))
    if not feature_names:
        raise ValueError("At least one feature/score name is required")

    learned_feature_names = tuple(
        dict.fromkeys(str(feature) for feature in learned_feature_names)
    )
    if learned_score_modes and not learned_feature_names:
        raise ValueError("At least one learned-score feature is required")
    collection_feature_names = _merge_feature_names(
        feature_names, learned_feature_names if learned_score_modes else ()
    )
    score_directions: dict[str, ScoreDirection] = {
        str(feature): "similarity" for feature in similarity_features
    }
    output_rows: list[dict[str, float | int | str]] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="edge-ranking"
    )
    if _uses_shifted_overlap_kwargs(config.pairwise_cost_kwargs):
        install_shifted_overlap_cost_patch()

    for subject_dir in subject_dirs:
        progress.step(f"ranking {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        sessions = _load_subject_sessions(subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)
        options = _reference_training_options(config, collection_feature_names)
        edges = session_edge_pairs(len(sessions), max_gap=config.max_gap)
        blocks = collect_reference_pairwise_example_blocks(
            sessions, reference, session_edges=edges, options=options
        )
        learned_models = _fit_loso_learned_score_models(
            subject_dirs,
            held_out_subject_dir=subject_dir,
            config=config,
            feature_names=learned_feature_names,
            score_modes=learned_score_modes,
            logistic_model_kwargs=logistic_model_kwargs,
            monotone_options=monotone_options,
        )
        for block in blocks:
            metadata = {
                "subject": subject_dir.name,
                "session_a": int(block.session_a),
                "session_b": int(block.session_b),
                "session_a_name": sessions[block.session_a].session_name,
                "session_b_name": sessions[block.session_b].session_name,
                "session_gap": int(block.gap),
            }
            score_matrices = _selected_score_matrices(block, feature_names)
            score_matrices.update(
                _learned_score_matrices_for_block(block, learned_models)
            )
            output_rows.extend(
                rank_labeled_edges(
                    block.labels,
                    score_matrices,
                    reference_roi_indices=block.reference_roi_indices,
                    measurement_roi_indices=block.measurement_roi_indices,
                    score_directions=score_directions,
                    metadata=metadata,
                )
            )
            output_rows.extend(
                missing_reference_edge_rows(
                    reference.pairwise_matches(
                        block.session_a,
                        block.session_b,
                        curated_only=config.curated_only,
                    ),
                    reference_roi_indices=block.reference_roi_indices,
                    measurement_roi_indices=block.measurement_roi_indices,
                    score_names=tuple(score_matrices),
                    score_directions=score_directions,
                    metadata=metadata,
                )
            )

    output_path = Path(output_path)
    summary_path = (
        Path(summary_output_path)
        if summary_output_path is not None
        else _default_summary_output_path(output_path)
    )
    summary_rows = summarize_edge_ranking_rows(output_rows, hit_ks=hit_ks)
    _write_csv(output_rows, output_path, preferred_fieldnames=EDGE_FIELDNAMES)
    _write_csv(summary_rows, summary_path, preferred_fieldnames=SUMMARY_FIELDNAMES)
    return len(output_rows), len(summary_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the edge-ranking diagnostic."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark edge-ranking",
        description="Export manual-GT pairwise edge-ranking diagnostics before global assignment.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Detailed per-GT-edge CSV output path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary CSV path; defaults to <output-stem>_summary.csv",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground_truth.csv file, ground-truth root, subject directory, or track2p folder",
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
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware", "calibrated"),
        help=(
            "Raw pairwise cost whose pairwise_cost_matrix should be ranked; "
            "cost='calibrated' adds a logistic LOSO learned score by default"
        ),
    )
    parser.add_argument(
        "--learned-score",
        dest="learned_score_modes",
        action="append",
        default=None,
        choices=LEARNED_SCORE_MODES,
        help=(
            "Add a LOSO learned score matrix to the ranking CSV; "
            "repeat for logistic-loso and monotone-loso"
        ),
    )
    parser.add_argument(
        "--learned-feature",
        dest="learned_features",
        action="append",
        default=None,
        help=(
            "Feature used by learned-score models; repeat to override the "
            "default calibrated feature set"
        ),
    )
    parser.add_argument(
        "--logistic-model-kwargs-json",
        default=None,
        help="JSON object passed to the logistic calibrated association model",
    )
    parser.add_argument(
        "--monotone-ranker-kwargs-json",
        default=None,
        help="JSON object passed to MonotoneRankerOptions",
    )
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
    parser.add_argument(
        "--shifted-iou-radius",
        type=int,
        default=0,
        help="Compute shifted-overlap diagnostic components with this integer-pixel radius",
    )
    parser.add_argument(
        "--shifted-iou-weight",
        type=float,
        default=0.0,
        help="Optional additive shifted-IoU cost weight for pairwise_cost_matrix",
    )
    parser.add_argument(
        "--shifted-mask-cosine-weight",
        type=float,
        default=0.0,
        help="Optional additive shifted-mask-cosine cost weight for pairwise_cost_matrix",
    )
    parser.add_argument(
        "--use-shifted-iou-for-iou-cost",
        action="store_true",
        help="Replace the exact IoU term in pairwise_cost_matrix with shifted IoU",
    )
    parser.add_argument(
        "--use-shifted-mask-cosine-for-mask-cosine-cost",
        action="store_true",
        help="Replace the exact mask-cosine term in pairwise_cost_matrix with shifted mask cosine",
    )
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs before ranking pairwise_cost_matrix",
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=None,
        help="Feature/component to rank; repeat to override the default feature set",
    )
    parser.add_argument(
        "--similarity-feature",
        dest="similarity_features",
        action="append",
        default=None,
        help="Declare a ranked feature where larger values are better; repeat as needed",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress to stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    feature_names = (
        tuple(args.features)
        if args.features is not None
        else DEFAULT_EDGE_RANKING_FEATURES
    )
    if args.features is None and args.shifted_iou_radius > 0:
        feature_names = _merge_feature_names(
            feature_names, DEFAULT_SHIFTED_EDGE_RANKING_FEATURES
        )
    similarity_features = (
        tuple(args.similarity_features)
        if args.similarity_features is not None
        else DEFAULT_SIMILARITY_FEATURES
    )
    edge_rows, summary_rows = run_track2p_edge_ranking(
        config,
        args.output,
        summary_output_path=args.summary_output,
        feature_names=feature_names,
        similarity_features=similarity_features,
        learned_score_modes=tuple(args.learned_score_modes or ()),
        learned_feature_names=(
            tuple(args.learned_features)
            if args.learned_features is not None
            else DEFAULT_LEARNED_SCORE_FEATURES
        ),
        logistic_model_kwargs=_json_object_or_none(
            args.logistic_model_kwargs_json, "--logistic-model-kwargs-json"
        ),
        monotone_options=_monotone_options_from_args(args),
    )
    summary_path = (
        args.summary_output
        if args.summary_output is not None
        else _default_summary_output_path(args.output)
    )
    print(
        f"Wrote {edge_rows} edge-ranking rows to {args.output} and {summary_rows} summary rows to {summary_path}"
    )
    return 0


def _reference_training_options(
    config: Track2pBenchmarkConfig, feature_names: Sequence[str]
) -> ReferenceTrainingOptions:
    return ReferenceTrainingOptions(
        curated_only=config.curated_only,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        feature_names=tuple(feature_names),
        pairwise_cost_kwargs=_pairwise_cost_kwargs_for_config(
            config.cost, config.pairwise_cost_kwargs
        ),
    )


def _pairwise_cost_kwargs_for_config(
    cost: AssociationCost,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if cost == "registered-iou":
        kwargs = registered_iou_cost_kwargs()
    elif cost == "roi-aware":
        kwargs = roi_aware_cost_kwargs()
    elif cost == "calibrated":
        kwargs = roi_aware_cost_kwargs()
    else:
        raise ValueError(f"Unsupported edge-ranking cost: {cost!r}")
    if overrides is not None:
        kwargs.update(dict(overrides))
    return kwargs


def _selected_score_matrices(
    block: ReferencePairwiseExamples, score_names: Sequence[str]
) -> dict[str, np.ndarray]:
    matrices = score_matrices_from_feature_tensor(block.features, block.feature_names)
    missing = [score_name for score_name in score_names if score_name not in matrices]
    if missing:
        raise ValueError(
            "Requested edge-ranking features are absent from the feature tensor: "
            + ", ".join(missing)
        )
    return {str(score_name): matrices[str(score_name)] for score_name in score_names}


def _fit_loso_learned_score_models(
    subject_dirs: Sequence[Path],
    *,
    held_out_subject_dir: Path,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    score_modes: Sequence[str],
    logistic_model_kwargs: Mapping[str, Any] | None,
    monotone_options: MonotoneRankerOptions | None,
) -> dict[str, Any]:
    if not score_modes:
        return {}
    if len(subject_dirs) < 2:
        raise ValueError("LOSO learned-score edge-ranking requires at least two subjects")
    training_blocks: list[ReferencePairwiseExamples] = []
    options = _reference_training_options(config, feature_names)
    for subject_dir in subject_dirs:
        if Path(subject_dir) == Path(held_out_subject_dir):
            continue
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        sessions = _load_subject_sessions(subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)
        training_blocks.extend(
            collect_reference_pairwise_example_blocks(
                sessions,
                reference,
                session_edges=session_edge_pairs(len(sessions), max_gap=config.max_gap),
                options=options,
            )
        )
    if not training_blocks:
        raise ValueError("No LOSO training blocks were collected for learned scores")

    models: dict[str, Any] = {}
    if "logistic-loso" in score_modes:
        training_features, training_labels = collect_candidate_limited_training_examples(
            tuple(training_blocks), options=CandidateHardNegativeOptions()
        )
        models["logistic-loso"] = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=tuple(feature_names),
            model_kwargs=logistic_model_kwargs,
        )
    if "monotone-loso" in score_modes:
        models["monotone-loso"] = fit_monotone_ranking_association_model_from_blocks(
            tuple(training_blocks),
            options=monotone_options or MonotoneRankerOptions(),
        )
    return models


def _learned_score_matrices_for_block(
    block: ReferencePairwiseExamples, models: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for score_mode, model in models.items():
        features = _block_feature_subset(block, tuple(model.feature_names))
        probabilities = np.asarray(model.predict_match_probability(features), dtype=float)
        if score_mode == "logistic-loso":
            prefix = "logistic_loso"
            matrices[f"{prefix}_probability"] = probabilities
            matrices[f"{prefix}_cost"] = _negative_log_probability(probabilities)
        elif score_mode == "monotone-loso":
            prefix = "monotone_loso"
            matrices[f"{prefix}_probability"] = probabilities
            matrices[f"{prefix}_cost"] = np.asarray(
                model.pairwise_cost_matrix(features), dtype=float
            )
            matrices[f"{prefix}_raw_cost"] = np.asarray(
                model.raw_cost_score(features), dtype=float
            )
    return matrices


def _block_feature_subset(
    block: ReferencePairwiseExamples, feature_names: Sequence[str]
) -> np.ndarray:
    block_feature_names = tuple(block.feature_names)
    missing = [
        feature_name
        for feature_name in feature_names
        if feature_name not in block_feature_names
    ]
    if missing:
        raise ValueError(
            "Learned-score features are absent from the held-out feature tensor: "
            + ", ".join(missing)
        )
    indices = [block_feature_names.index(feature_name) for feature_name in feature_names]
    return np.asarray(block.features[..., indices], dtype=float)


def _negative_log_probability(probabilities: Any) -> np.ndarray:
    return -np.log(np.clip(np.asarray(probabilities, dtype=float), 1.0e-12, 1.0))


def _merge_feature_names(*feature_groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for feature_group in feature_groups:
        for feature_name in feature_group:
            feature_name = str(feature_name)
            if feature_name not in seen:
                seen.add(feature_name)
                merged.append(feature_name)
    return tuple(merged)


def _normalize_learned_score_modes(
    learned_score_modes: Sequence[str], *, cost: AssociationCost
) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(str(mode) for mode in learned_score_modes))
    if cost == "calibrated" and not modes:
        modes = ("logistic-loso",)
    invalid = [mode for mode in modes if mode not in LEARNED_SCORE_MODES]
    if invalid:
        raise ValueError("Unsupported learned-score mode(s): " + ", ".join(invalid))
    return modes


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = _pairwise_cost_kwargs_from_args(args)
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="subject",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        cost=args.cost,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
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


def _pairwise_cost_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    pairwise_cost_kwargs = _json_object_or_none(
        args.pairwise_cost_kwargs_json, "--pairwise-cost-kwargs-json"
    )
    kwargs = dict(pairwise_cost_kwargs or {})
    if args.shifted_iou_radius < 0:
        raise ValueError("--shifted-iou-radius must be non-negative")
    if args.shifted_iou_weight < 0.0:
        raise ValueError("--shifted-iou-weight must be non-negative")
    if args.shifted_mask_cosine_weight < 0.0:
        raise ValueError("--shifted-mask-cosine-weight must be non-negative")
    if args.shifted_iou_radius > 0:
        kwargs["shifted_iou_radius"] = int(args.shifted_iou_radius)
        kwargs.setdefault("return_shifted_overlap_components", True)
    if args.shifted_iou_weight > 0.0:
        kwargs["shifted_iou_weight"] = float(args.shifted_iou_weight)
    if args.shifted_mask_cosine_weight > 0.0:
        kwargs["shifted_mask_cosine_weight"] = float(args.shifted_mask_cosine_weight)
    if args.use_shifted_iou_for_iou_cost:
        kwargs["use_shifted_iou_for_iou_cost"] = True
    if args.use_shifted_mask_cosine_for_mask_cosine_cost:
        kwargs["use_shifted_mask_cosine_for_mask_cosine_cost"] = True
    if _positive_shifted_radius(kwargs):
        kwargs.setdefault("return_shifted_overlap_components", True)
    if not kwargs:
        return None
    return kwargs


def _json_object_or_none(value: str | None, option_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def _monotone_options_from_args(args: argparse.Namespace) -> MonotoneRankerOptions | None:
    parsed = _json_object_or_none(
        args.monotone_ranker_kwargs_json, "--monotone-ranker-kwargs-json"
    )
    if parsed is None:
        return None
    return MonotoneRankerOptions(**parsed)


def _positive_shifted_radius(kwargs: Mapping[str, Any]) -> bool:
    try:
        return int(kwargs.get("shifted_iou_radius", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _uses_shifted_overlap_kwargs(kwargs: Mapping[str, Any] | None) -> bool:
    return bool(kwargs and _SHIFTED_OVERLAP_KWARG_NAMES.intersection(kwargs))


def _default_summary_output_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    suffix = output_path.suffix or ".csv"
    return output_path.with_name(f"{output_path.stem}_summary{suffix}")


def _write_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: Path,
    *,
    preferred_fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows, preferred_fieldnames)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(
    rows: Sequence[Mapping[str, float | int | str]], preferred: Sequence[str]
) -> list[str]:
    row_keys = {key for row in rows for key in row}
    return [key for key in preferred if key in row_keys] + sorted(
        row_keys - set(preferred)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
