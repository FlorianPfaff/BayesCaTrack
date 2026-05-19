"""Track2p-as-teacher distillation benchmark.

This command turns Track2p output into pseudo-labels for the monotone ranking
association model, then evaluates the resulting BayesCaTrack global assignment
against independent manual ground truth.  The purpose is diagnostic: it answers
whether the current feature family and monotone ranker can learn Track2p-like
edge rankings when Track2p is used as a teacher/debug oracle.

The default mode is leave-one-subject-out with respect to manual-GT subjects:
for each held-out manual subject, the ranker is trained from Track2p pseudo
labels from the other manual subjects and then scored against the held-out
manual ground truth.  Optional ``--teacher-data`` can add teacher-only subjects
without manual GT to every fold.
"""

# jscpd:ignore-start
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    collect_reference_training_examples,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _resolve_track2p_reference_path,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    _reference_training_options,
)
from bayescatrack.reference import Track2pReference, load_track2p_reference


@dataclass(frozen=True)
class Track2pTeacherDistillationConfig:
    """Configuration for Track2p-teacher distillation."""

    benchmark: Track2pBenchmarkConfig
    track2p_reference: Path | None = None
    teacher_data: Path | None = None
    teacher_track2p_reference: Path | None = None
    include_held_out_teacher_labels: bool = False
    teacher_curated_only: bool = False


@dataclass(frozen=True)
class TeacherDistillationSubject:
    """Loaded sessions plus manual/teacher references for one subject."""

    subject_dir: Path
    sessions: tuple[Track2pSession, ...]
    teacher_reference: Track2pReference
    manual_reference: Track2pReference | None = None

    @property
    def subject_name(self) -> str:
        """Return a stable subject identifier for result tables."""

        return self.subject_dir.name


# pylint: disable=too-many-locals,too-many-arguments
def run_track2p_teacher_distillation(
    config: Track2pTeacherDistillationConfig,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    monotone_options: MonotoneRankerOptions | None = None,
) -> LosoCalibrationResult:
    """Train Track2p-teacher monotone rankers and evaluate on manual GT folds."""

    benchmark = config.benchmark
    if benchmark.method != "global-assignment" or benchmark.cost != "calibrated":
        raise ValueError(
            "Teacher distillation requires method='global-assignment' and cost='calibrated'"
        )
    if benchmark.split != "leave-one-subject-out":
        raise ValueError("Teacher distillation requires split='leave-one-subject-out'")

    evaluation_dirs = tuple(discover_subject_dirs(benchmark.data))
    if not evaluation_dirs:
        raise ValueError(
            f"No Track2p-style evaluation subject directories found under {benchmark.data}"
        )

    if config.teacher_data is None:
        teacher_dirs: tuple[Path, ...] = ()
    else:
        teacher_dirs = tuple(discover_subject_dirs(config.teacher_data))
        if not teacher_dirs:
            raise ValueError(
                "No Track2p-style teacher subject directories found under "
                f"{config.teacher_data}"
            )

    progress = ProgressReporter(
        len(evaluation_dirs) + len(teacher_dirs) + 4 * len(evaluation_dirs),
        enabled=benchmark.progress,
        label="teacher-distill",
    )
    evaluation_subjects = tuple(
        _load_evaluation_subject(subject_dir, config=config, progress=progress)
        for subject_dir in evaluation_dirs
    )
    teacher_subjects = tuple(
        _load_teacher_subject(subject_dir, config=config, progress=progress)
        for subject_dir in teacher_dirs
    )

    feature_names = tuple(feature_names)
    options = monotone_options or MonotoneRankerOptions()
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(evaluation_subjects):
        manual_reference = _require_manual_reference(held_out)
        fold_teacher_subjects = _training_subjects_for_fold(
            evaluation_subjects,
            teacher_subjects,
            held_out_index=held_out_index,
            include_held_out_teacher_labels=config.include_held_out_teacher_labels,
        )
        progress.step(f"collecting Track2p-teacher blocks for {held_out.subject_name}")
        training_blocks = _collect_teacher_training_blocks(
            fold_teacher_subjects,
            benchmark=benchmark,
            feature_names=feature_names,
            teacher_curated_only=config.teacher_curated_only,
        )
        progress.step(f"fitting Track2p-teacher ranker for {held_out.subject_name}")
        teacher_model = fit_monotone_ranking_association_model_from_blocks(
            training_blocks,
            options=options,
        )
        progress.step(f"scoring teacher model diagnostics for {held_out.subject_name}")
        calibration_scores = {
            **_score_model_against_reference(
                teacher_model,
                held_out,
                manual_reference,
                benchmark=benchmark,
                feature_names=feature_names,
                prefix="manual_holdout",
                curated_only=benchmark.curated_only,
            ),
            **_score_model_against_reference(
                teacher_model,
                held_out,
                held_out.teacher_reference,
                benchmark=benchmark,
                feature_names=feature_names,
                prefix="teacher_holdout",
                curated_only=config.teacher_curated_only,
            ),
        }
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            benchmark,
            cost="calibrated",
            calibrated_model=teacher_model,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, manual_reference, config=benchmark
        )
        track2p_teacher_scores = _prefixed_scores(
            _score_prediction_against_reference(
                normalize_track_matrix(held_out.teacher_reference.suite2p_indices),
                manual_reference,
                config=benchmark,
            ),
            prefix="track2p_teacher",
        )
        training_examples = int(teacher_model.n_training_examples)
        positives = int(teacher_model.n_positive_examples)
        scores: dict[str, float | int | str] = {
            **base_scores,
            "calibration_model": "track2p-teacher-monotone-ranker",
            "training_reference_source": "track2p_teacher_output",
            "teacher_training_subjects": ",".join(
                subject.subject_name for subject in fold_teacher_subjects
            ),
            "include_held_out_teacher_labels": int(
                config.include_held_out_teacher_labels
            ),
            "teacher_curated_only": int(config.teacher_curated_only),
            "training_examples": training_examples,
            "positive_examples": positives,
            "negative_examples": int(training_examples - positives),
            "monotone_feature_names": ",".join(teacher_model.monotone_feature_names),
            "monotone_rank_constraints": int(teacher_model.n_rank_constraints),
            "monotone_training_rank_loss": float(teacher_model.training_rank_loss),
            "monotone_training_binary_loss": float(
                teacher_model.training_binary_loss
            ),
            **_monotone_option_scores(options),
            **calibration_scores,
            **track2p_teacher_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in fold_teacher_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant="Track2p-teacher monotone ranker + LOSO global assignment",
                    method=benchmark.method,
                    scores=scores,
                    n_sessions=manual_reference.n_sessions,
                    reference_source=manual_reference.source,
                ),
                training_examples=training_examples,
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(benchmark.max_gap)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Track2p-teacher distillation parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-teacher-distill",
        description=(
            "Train monotone association costs from Track2p pseudo-labels and "
            "evaluate the resulting global assignment against manual GT."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Manual-GT CSV/root used for evaluation subjects",
    )
    parser.add_argument("--reference-kind", default="manual-gt", choices=("auto", "manual-gt"))
    parser.add_argument(
        "--track2p-reference",
        type=Path,
        default=None,
        help="Track2p output root/folder used as teacher for evaluation subjects",
    )
    parser.add_argument(
        "--teacher-data",
        type=Path,
        default=None,
        help="Optional extra Track2p-style subject root used only for teacher training",
    )
    parser.add_argument(
        "--teacher-track2p-reference",
        type=Path,
        default=None,
        help=(
            "Optional Track2p output root/folder for --teacher-data subjects; "
            "defaults to --track2p-reference and then subject/track2p"
        ),
    )
    parser.add_argument(
        "--include-held-out-teacher-labels",
        action="store_true",
        help=(
            "Train each fold on the held-out subject's Track2p output too. "
            "This is a transductive debug upper bound, not a clean paper result."
        ),
    )
    parser.add_argument(
        "--teacher-curated-only",
        action="store_true",
        help="Use only teacher-reference curated tracks when collecting pseudo-labels",
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
    parser.add_argument("--monotone-ranker-kwargs-json", default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    options = _monotone_options_from_args(args)
    rows = [
        fold.to_dict()
        for fold in run_track2p_teacher_distillation(
            config, monotone_options=options
        ).folds
    ]
    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> Track2pTeacherDistillationConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    benchmark = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
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
    return Track2pTeacherDistillationConfig(
        benchmark=benchmark,
        track2p_reference=args.track2p_reference,
        teacher_data=args.teacher_data,
        teacher_track2p_reference=args.teacher_track2p_reference,
        include_held_out_teacher_labels=args.include_held_out_teacher_labels,
        teacher_curated_only=args.teacher_curated_only,
    )


def _monotone_options_from_args(args: argparse.Namespace) -> MonotoneRankerOptions:
    if args.monotone_ranker_kwargs_json is None:
        return MonotoneRankerOptions()
    parsed = json.loads(args.monotone_ranker_kwargs_json)
    if not isinstance(parsed, dict):
        raise ValueError("--monotone-ranker-kwargs-json must decode to a JSON object")
    return MonotoneRankerOptions(**parsed)


def _load_evaluation_subject(
    subject_dir: Path,
    *,
    config: Track2pTeacherDistillationConfig,
    progress: ProgressReporter,
) -> TeacherDistillationSubject:
    progress.step(f"loading evaluation subject {subject_dir.name}")
    benchmark = config.benchmark
    sessions = tuple(_load_subject_sessions(subject_dir, benchmark))
    manual_reference = _load_reference_for_subject(
        subject_dir, data_root=benchmark.data, config=benchmark
    )
    _validate_reference_for_benchmark(
        manual_reference, subject_dir=subject_dir, config=benchmark
    )
    if manual_reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(
            f"Subject {subject_dir.name!r} needs independent manual GT for teacher distillation evaluation"
        )
    _validate_reference_roi_indices(manual_reference, sessions)
    teacher_reference = _load_teacher_reference(
        subject_dir,
        data_root=benchmark.data,
        reference_root=config.track2p_reference,
        plane_name=benchmark.plane_name,
    )
    _validate_teacher_reference_for_sessions(
        teacher_reference, sessions, subject=subject_dir.name
    )
    return TeacherDistillationSubject(
        subject_dir=subject_dir,
        sessions=sessions,
        manual_reference=manual_reference,
        teacher_reference=teacher_reference,
    )


def _load_teacher_subject(
    subject_dir: Path,
    *,
    config: Track2pTeacherDistillationConfig,
    progress: ProgressReporter,
) -> TeacherDistillationSubject:
    progress.step(f"loading teacher subject {subject_dir.name}")
    benchmark = config.benchmark
    teacher_data_root = config.teacher_data or benchmark.data
    teacher_reference_root = (
        config.teacher_track2p_reference
        if config.teacher_track2p_reference is not None
        else config.track2p_reference
    )
    sessions = tuple(_load_subject_sessions(subject_dir, benchmark))
    teacher_reference = _load_teacher_reference(
        subject_dir,
        data_root=teacher_data_root,
        reference_root=teacher_reference_root,
        plane_name=benchmark.plane_name,
    )
    _validate_teacher_reference_for_sessions(
        teacher_reference, sessions, subject=subject_dir.name
    )
    return TeacherDistillationSubject(
        subject_dir=subject_dir, sessions=sessions, teacher_reference=teacher_reference
    )


def _load_teacher_reference(
    subject_dir: Path,
    *,
    data_root: Path,
    reference_root: Path | None,
    plane_name: str,
) -> Track2pReference:
    root = Path(reference_root) if reference_root is not None else subject_dir / "track2p"
    reference_path = _resolve_track2p_reference_path(
        subject_dir, data_root=Path(data_root), reference_root=root
    )
    if reference_path is None:
        raise FileNotFoundError(
            f"Could not resolve Track2p teacher output for subject {subject_dir.name!r} under {root}"
        )
    return load_track2p_reference(reference_path, plane_name=plane_name)


def _validate_teacher_reference_for_sessions(
    reference: Track2pReference, sessions: Sequence[Track2pSession], *, subject: str
) -> None:
    session_names = tuple(session.session_name for session in sessions)
    if reference.n_sessions != len(sessions):
        raise ValueError(
            f"Subject {subject!r}: Track2p teacher has {reference.n_sessions} sessions, "
            f"loaded data have {len(sessions)} sessions"
        )
    if reference.session_names != session_names:
        raise ValueError(
            f"Subject {subject!r}: Track2p teacher session order {reference.session_names!r} "
            f"does not match loaded sessions {session_names!r}"
        )


def _require_manual_reference(subject: TeacherDistillationSubject) -> Track2pReference:
    if subject.manual_reference is None:
        raise ValueError(f"Subject {subject.subject_name!r} has no manual reference")
    return subject.manual_reference


def _training_subjects_for_fold(
    evaluation_subjects: Sequence[TeacherDistillationSubject],
    teacher_subjects: Sequence[TeacherDistillationSubject],
    *,
    held_out_index: int,
    include_held_out_teacher_labels: bool,
) -> tuple[TeacherDistillationSubject, ...]:
    held_out_key = _subject_key(evaluation_subjects[held_out_index])
    selected: list[TeacherDistillationSubject] = [
        subject
        for subject in teacher_subjects
        if include_held_out_teacher_labels or _subject_key(subject) != held_out_key
    ]
    teacher_keys = {_subject_key(subject) for subject in selected}
    for index, subject in enumerate(evaluation_subjects):
        if index == held_out_index and not include_held_out_teacher_labels:
            continue
        key = _subject_key(subject)
        if key not in teacher_keys:
            selected.append(subject)
            teacher_keys.add(key)
    if not selected:
        raise ValueError(
            "No teacher-training subjects are available for this fold. Add --teacher-data "
            "or pass --include-held-out-teacher-labels for a transductive debug run."
        )
    return tuple(selected)


def _subject_key(subject: TeacherDistillationSubject) -> tuple[str, str]:
    return (subject.subject_name, str(subject.subject_dir.resolve()))


def _collect_teacher_training_blocks(
    training_subjects: Sequence[TeacherDistillationSubject],
    *,
    benchmark: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    teacher_curated_only: bool,
) -> tuple[Any, ...]:
    blocks: list[Any] = []
    training_options = _teacher_training_options(
        benchmark, feature_names, teacher_curated_only=teacher_curated_only
    )
    for subject in training_subjects:
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                subject.sessions,
                subject.teacher_reference,
                session_edges=session_edge_pairs(
                    len(subject.sessions), max_gap=benchmark.max_gap
                ),
                options=training_options,
            )
        )
    if not blocks:
        raise ValueError("At least one Track2p-teacher training block is required")
    return tuple(blocks)


def _teacher_training_options(
    benchmark: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    *,
    teacher_curated_only: bool,
) -> ReferenceTrainingOptions:
    return replace(
        _reference_training_options(benchmark, feature_names),
        curated_only=teacher_curated_only,
    )


def _score_model_against_reference(
    calibrated_model: Any,
    subject: TeacherDistillationSubject,
    reference: Track2pReference,
    *,
    benchmark: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    prefix: str,
    curated_only: bool,
) -> dict[str, float | int | str]:
    options = replace(
        _reference_training_options(benchmark, feature_names), curated_only=curated_only
    )
    features, labels = collect_reference_training_examples(
        subject.sessions,
        reference,
        session_edges=session_edge_pairs(len(subject.sessions), max_gap=benchmark.max_gap),
        options=options,
    )
    probabilities = np.asarray(
        calibrated_model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    summary = calibration_summary(probabilities, np.asarray(labels).reshape(-1))
    return _prefixed_scores(summary, prefix=prefix)


def _prefixed_scores(
    scores: Mapping[str, Any], *, prefix: str
) -> dict[str, float | int | str]:
    return {
        f"{prefix}_{key}": value
        for key, value in scores.items()
        if isinstance(value, (float, int, str, np.floating, np.integer))
    }


def _monotone_option_scores(
    options: MonotoneRankerOptions,
) -> dict[str, float | int | str]:
    values = asdict(options)
    return {
        f"monotone_option_{key}": ",".join(value) if isinstance(value, tuple) else value
        for key, value in values.items()
    }


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
        "track2p_teacher_pairwise_f1",
        "track2p_teacher_complete_track_f1",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "calibration_model",
        "monotone_rank_constraints",
        "teacher_training_subjects",
        "include_held_out_teacher_labels",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
# jscpd:ignore-end
