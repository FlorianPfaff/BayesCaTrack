"""Configurable hard-negative LOSO calibration for Track2p benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    CalibratedAssociationModel,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    balanced_binary_sample_weights,
    collect_candidate_limited_training_examples,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _score_prediction_against_reference,
    discover_subject_dirs,
    format_benchmark_table,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SubjectCalibrationData,
    _load_subject_calibration_data,
    _reference_training_options,
    _score_holdout_calibration,
)

SampleWeightStrategy = Literal["none", "balanced"]
CalibrationModelKind = Literal["logistic", "hist-gradient-boosting"]


@dataclass
class SklearnPairwiseProbabilityAdapter:
    """Adapter that makes sklearn classifiers compatible with pairwise tensors."""

    estimator: Any

    def fit(
        self,
        features: Any,
        labels: Any,
        *,
        sample_weight: Any | None = None,
    ) -> SklearnPairwiseProbabilityAdapter:
        x, _shape = _flatten_feature_array(features)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if y.shape[0] != x.shape[0]:
            raise ValueError("labels must have one entry for each feature vector")
        kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != x.shape[0]:
                raise ValueError(
                    "sample_weight must have one entry for each feature vector"
                )
            kwargs["sample_weight"] = w
        self.estimator.fit(x, y, **kwargs)
        return self

    def predict_match_probability(self, features: Any) -> np.ndarray:
        x, shape = _flatten_feature_array(features)
        prob = np.asarray(self.estimator.predict_proba(x), dtype=float)
        if prob.ndim == 2:
            classes = list(getattr(self.estimator, "classes_", (0, 1)))
            positive_index = classes.index(1) if 1 in classes else prob.shape[1] - 1
            prob = prob[:, positive_index]
        return np.asarray(prob, dtype=float).reshape(shape)


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_configurable_loso_calibration(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kind: CalibrationModelKind = "logistic",
    model_kwargs: Mapping[str, Any] | None = None,
    hard_negative_options: CandidateHardNegativeOptions | None = None,
) -> LosoCalibrationResult:
    """Run LOSO calibrated assignment with configurable hard-negative sampling."""

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    subjects = _load_subjects(config)
    feature_names = tuple(feature_names)
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    model_kind = _validate_calibration_model_kind(model_kind)
    model_kwargs = _model_kwargs(model_kind, model_kwargs)
    hard_negative_options = hard_negative_options or CandidateHardNegativeOptions()
    progress = ProgressReporter(
        len(subjects) * (len(subjects) + 2),
        enabled=config.progress,
        label="LOSO",
    )
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        features, labels = _collect_examples(
            training_subjects,
            config,
            feature_names,
            hard_negative_options,
            progress,
            held_out.subject_name,
        )
        weights = _training_sample_weight(
            labels,
            sample_weight=sample_weight,
            strategy=sample_weight_strategy,
        )
        progress.step(f"fitting model for {held_out.subject_name}")
        model = _fit_model(
            features,
            labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
        )
        progress.step(f"scoring calibration for {held_out.subject_name}")
        calibration_scores = _score_holdout_calibration(
            model,
            held_out,
            config=config,
            feature_names=feature_names,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=model,
        )
        predicted = tracks_to_suite2p_index_matrix(
            assignment.result.tracks,
            held_out.sessions,
        )
        positives = int(np.sum(labels))
        scores: dict[str, float | int | str] = {
            **_score_prediction_against_reference(
                predicted,
                held_out.reference,
                config=config,
            ),
            "training_examples": int(labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(labels.shape[0] - positives),
            "calibration_model": model_kind,
            "calibration_model_kwargs": json.dumps(
                dict(model_kwargs),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "calibration_sample_weight_strategy": sample_weight_strategy,
            "calibration_class_weight": _class_weight_label(model_kwargs),
            **_hard_negative_scores(hard_negative_options),
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
                    variant="Configurable calibrated costs + LOSO global assignment",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=int(labels.shape[0]),
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds),
        feature_names=feature_names,
        max_gap=int(config.max_gap),
    )


def _load_subjects(config: Track2pBenchmarkConfig) -> tuple[SubjectCalibrationData, ...]:
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("LOSO calibration requires at least two subject directories")
    progress = ProgressReporter(
        len(subject_dirs),
        enabled=config.progress,
        label="LOSO",
    )
    subjects = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(_load_subject_calibration_data(subject_dir, config=config))
    return tuple(subjects)


def _collect_examples(
    subjects: Sequence[SubjectCalibrationData],
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    hard_negative_options: CandidateHardNegativeOptions,
    progress: ProgressReporter,
    held_out_subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    options = _reference_training_options(config, feature_names)
    for subject in subjects:
        progress.step(
            f"collecting {subject.subject_name} training features for {held_out_subject}"
        )
        blocks = collect_reference_pairwise_example_blocks(
            subject.sessions,
            subject.reference,
            session_edges=session_edge_pairs(
                len(subject.sessions),
                max_gap=config.max_gap,
            ),
            options=options,
        )
        features, labels = collect_candidate_limited_training_examples(
            blocks,
            options=hard_negative_options,
        )
        feature_blocks.append(features)
        label_blocks.append(labels)
    if not feature_blocks:
        raise ValueError("At least one training subject is required")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def _fit_model(
    features: Any,
    labels: Any,
    *,
    feature_names: Sequence[str],
    sample_weight: Any | None,
    model_kind: CalibrationModelKind,
    model_kwargs: Mapping[str, Any],
) -> CalibratedAssociationModel:
    if model_kind == "logistic":
        return fit_logistic_association_model(
            features,
            labels,
            feature_names=feature_names,
            sample_weight=sample_weight,
            model_kwargs=model_kwargs,
        )
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "model_kind='hist-gradient-boosting' requires scikit-learn to be installed"
        ) from exc
    adapter = SklearnPairwiseProbabilityAdapter(
        HistGradientBoostingClassifier(**dict(model_kwargs))
    )
    adapter.fit(features, labels, sample_weight=sample_weight)
    return CalibratedAssociationModel(model=adapter, feature_names=tuple(feature_names))


def _training_sample_weight(
    labels: np.ndarray,
    *,
    sample_weight: Any | None,
    strategy: SampleWeightStrategy,
) -> Any | None:
    if sample_weight is not None:
        if strategy != "none":
            raise ValueError(
                "sample_weight_strategy must be 'none' when explicit sample_weight is supplied"
            )
        return sample_weight
    if strategy == "none":
        return None
    if strategy == "balanced":
        return balanced_binary_sample_weights(labels)
    raise ValueError(f"Unsupported sample_weight_strategy: {strategy!r}")


def _validate_sample_weight_strategy(strategy: str) -> SampleWeightStrategy:
    if strategy not in {"none", "balanced"}:
        raise ValueError("sample_weight_strategy must be either 'none' or 'balanced'")
    return cast(SampleWeightStrategy, strategy)


def _validate_calibration_model_kind(kind: str) -> CalibrationModelKind:
    if kind not in {"logistic", "hist-gradient-boosting"}:
        raise ValueError(
            "model_kind must be either 'logistic' or 'hist-gradient-boosting'"
        )
    return cast(CalibrationModelKind, kind)


def _model_kwargs(
    model_kind: CalibrationModelKind,
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(values or {})
    if model_kind == "logistic":
        kwargs.setdefault("class_weight", None)
    else:
        kwargs.setdefault("random_state", 0)
    return kwargs


def _class_weight_label(model_kwargs: Mapping[str, Any]) -> str:
    class_weight = model_kwargs.get("class_weight")
    return "None" if class_weight is None else str(class_weight)


def _hard_negative_scores(
    options: CandidateHardNegativeOptions,
) -> dict[str, float | int | str]:
    return {
        "hard_negative_negative_to_positive_ratio": float(
            options.negative_to_positive_ratio
        ),
        "hard_negative_candidate_top_k_per_anchor": (
            "None"
            if options.candidate_top_k_per_anchor is None
            else int(options.candidate_top_k_per_anchor)
        ),
        "hard_negative_include_column_candidates": int(
            options.include_column_candidates
        ),
        "hard_negative_hardness_feature_names": ",".join(
            options.hardness_feature_names
        ),
    }


def _flatten_feature_array(features: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(features, dtype=float)
    if array.ndim < 2:
        raise ValueError("features must have shape (..., n_features)")
    return array.reshape(-1, array.shape[-1]), tuple(
        int(value) for value in array.shape[:-1]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-loso-calibration"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
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
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument(
        "--sample-weight-strategy", default="none", choices=("none", "balanced")
    )
    parser.add_argument(
        "--calibration-model",
        default="logistic",
        choices=("logistic", "hist-gradient-boosting"),
    )
    parser.add_argument("--calibration-model-kwargs-json", default=None)
    parser.add_argument("--hard-negative-ratio", type=float, default=4.0)
    parser.add_argument("--hard-negative-top-k", type=_none_or_positive_int, default=20)
    parser.add_argument(
        "--hard-negative-column-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hard-negative-features", default="")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    result = run_track2p_configurable_loso_calibration(
        config,
        sample_weight_strategy=args.sample_weight_strategy,
        model_kind=args.calibration_model,
        model_kwargs=_json_object(
            args.calibration_model_kwargs_json,
            "--calibration-model-kwargs-json",
        ),
        hard_negative_options=_hard_negative_options(args),
    )
    rows = result.to_rows()
    if args.output is None:
        _write_stdout(rows, args.format)
    else:
        write_results(rows, args.output, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
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
        pairwise_cost_kwargs=_json_object(
            args.pairwise_cost_kwargs_json,
            "--pairwise-cost-kwargs-json",
        ),
        progress=args.progress,
    )


def _hard_negative_options(args: argparse.Namespace) -> CandidateHardNegativeOptions:
    return CandidateHardNegativeOptions(
        negative_to_positive_ratio=args.hard_negative_ratio,
        candidate_top_k_per_anchor=args.hard_negative_top_k,
        include_column_candidates=args.hard_negative_column_candidates,
        hardness_feature_names=tuple(
            token.strip()
            for token in args.hard_negative_features.split(",")
            if token.strip()
        ),
    )


def _json_object(value: str | None, option_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def _none_or_positive_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.casefold() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, or 'none'")
    return parsed


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
    print(format_benchmark_table(rows))


def _csv_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "pairwise_f1",
        "complete_track_f1",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "calibration_model",
        "calibration_sample_weight_strategy",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
