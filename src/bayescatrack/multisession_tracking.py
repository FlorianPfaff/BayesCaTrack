#!/usr/bin/env python3
"""Global multi-session tracking for BayesCaTrack.

BayesCaTrack provides the pairwise preprocessing: loading Track2p/Suite2p
session folders, reconstructing ROI masks, building ROI-aware pairwise cost
matrices, and packaging pairwise association inputs for PyRecEst.

This module turns those pairwise costs into a single longitudinal identity
assignment across all sessions. Registration is treated as an input to this
layer, not as something reimplemented here. Whenever you have later-session ROIs
already transformed into earlier-session coordinates, pass them via
``pairwise_measurement_planes_in_reference_frames``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bayescatrack import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_session_pair_association_bundle,
    load_track2p_subject,
)
from bayescatrack.core._bridge_impl import _suite2p_kwargs_from_args


@dataclass(frozen=True)
class MultisessionTrackingConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for global longitudinal neuron tracking."""

    max_session_gap: int = 1
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    start_cost: float = 0.2
    end_cost: float = 0.2
    gap_penalty: float = 0.3
    cost_threshold: float | None = None
    return_pairwise_components: bool = True
    pairwise_cost_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_session_gap < 1:
            raise ValueError("max_session_gap must be at least 1")
        if self.order not in {"xy", "yx"}:
            raise ValueError("order must be either 'xy' or 'yx'")
        if self.velocity_variance < 0.0:
            raise ValueError("velocity_variance must be non-negative")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        for attribute_name in ("start_cost", "end_cost", "gap_penalty"):
            if getattr(self, attribute_name) < 0.0:
                raise ValueError(f"{attribute_name} must be non-negative")
        if self.cost_threshold is not None and self.cost_threshold < 0.0:
            raise ValueError("cost_threshold must be non-negative when specified")


@dataclass(frozen=True)
class PairwiseTrackingBundle:
    """Pairwise association output annotated with session indices."""

    source_session_index: int
    target_session_index: int
    bundle: SessionAssociationBundle

    @property
    def pairwise_cost_matrix(self) -> np.ndarray:
        return self.bundle.pairwise_cost_matrix


@dataclass(frozen=True)
class LongitudinalTrackingResult:
    """Global tracking result in Track2p-friendly matrix form."""

    tracks: tuple[dict[int, int], ...]
    track_matrix: np.ndarray
    track_roi_index_matrix: np.ndarray
    session_names: tuple[str, ...]
    session_dates: tuple[str | None, ...]
    pairwise_bundles: tuple[PairwiseTrackingBundle, ...]
    total_cost: float | None = None

    def summary(self) -> dict[str, Any]:
        if self.track_matrix.ndim != 2:
            raise ValueError("track_matrix must be two-dimensional")
        track_lengths = np.sum(self.track_matrix >= 0, axis=1, dtype=int)
        n_sessions = int(self.track_matrix.shape[1]) if self.track_matrix.size else len(self.session_names)
        return {
            "n_sessions": n_sessions,
            "n_tracks": int(self.track_matrix.shape[0]),
            "n_complete_tracks": int(np.sum(track_lengths == n_sessions)) if n_sessions > 0 else 0,
            "mean_track_length": float(np.mean(track_lengths)) if track_lengths.size > 0 else 0.0,
            "total_cost": None if self.total_cost is None else float(self.total_cost),
        }


def _import_multisession_solver() -> Callable[..., Any]:
    """Import a PyRecEst multisession-assignment solver.

    The exact module path may move while the relevant PRs are iterated on, so we
    try a small set of likely locations and only fail at call time.
    """

    candidates = [
        ("pyrecest.assignment.multisession", "solve_multisession_assignment"),
        (
            "pyrecest.multiobjecttracking.multisession_assignment",
            "solve_multisession_assignment",
        ),
        (
            "pyrecest.multiobjecttracking.assignment.multisession_assignment",
            "solve_multisession_assignment",
        ),
        ("pyrecest.utils.multisession_assignment", "solve_multisession_assignment"),
    ]

    errors: list[str] = []
    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(module_name)
            solver = getattr(module, attribute_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{attribute_name}: {exc}")
            continue
        if callable(solver):
            return solver
        errors.append(f"{module_name}.{attribute_name}: not callable")

    joined_errors = "; ".join(errors) if errors else "no candidates tried"
    raise ImportError(
        "Could not import a PyRecEst multisession solver. Tried: "
        f"{joined_errors}"
    )


def build_multisession_pairwise_costs(
    sessions: Sequence[Track2pSession],
    *,
    config: MultisessionTrackingConfig | None = None,
    pairwise_measurement_planes_in_reference_frames: Mapping[
        tuple[int, int], CalciumPlaneData | None
    ]
    | None = None,
) -> tuple[dict[tuple[int, int], np.ndarray], tuple[PairwiseTrackingBundle, ...]]:
    """Build pairwise cost matrices for all session pairs up to ``max_session_gap``.

    Parameters
    ----------
    sessions
        Ordered sequence of per-session Track2p data.
    config
        Global tracking configuration.
    pairwise_measurement_planes_in_reference_frames
        Optional mapping ``(i, j) -> plane`` where ``plane`` is session ``j``
        already expressed in session ``i``'s coordinate frame. This is the right
        hook when you want Track2p-style registration-aware overlap costs.
    """

    config = MultisessionTrackingConfig() if config is None else config
    pairwise_measurement_planes_in_reference_frames = dict(
        pairwise_measurement_planes_in_reference_frames or {}
    )

    pairwise_costs: dict[tuple[int, int], np.ndarray] = {}
    pairwise_bundles: list[PairwiseTrackingBundle] = []

    sessions = list(sessions)
    for source_session_index in range(len(sessions)):
        max_target_index = min(
            len(sessions), source_session_index + config.max_session_gap + 1
        )
        for target_session_index in range(source_session_index + 1, max_target_index):
            measurement_plane_in_reference_frame = (
                pairwise_measurement_planes_in_reference_frames.get(
                    (source_session_index, target_session_index)
                )
            )
            bundle = build_session_pair_association_bundle(
                sessions[source_session_index],
                sessions[target_session_index],
                measurement_plane_in_reference_frame=measurement_plane_in_reference_frame,
                order=config.order,
                weighted_centroids=config.weighted_centroids,
                velocity_variance=config.velocity_variance,
                regularization=config.regularization,
                pairwise_cost_kwargs=config.pairwise_cost_kwargs,
                return_pairwise_components=config.return_pairwise_components,
            )
            pairwise_costs[(source_session_index, target_session_index)] = np.asarray(
                bundle.pairwise_cost_matrix,
                dtype=float,
            )
            pairwise_bundles.append(
                PairwiseTrackingBundle(
                    source_session_index=source_session_index,
                    target_session_index=target_session_index,
                    bundle=bundle,
                )
            )

    return pairwise_costs, tuple(pairwise_bundles)


def _call_multisession_solver(
    solver: Callable[..., Any],
    pairwise_costs: Mapping[tuple[int, int], np.ndarray],
    session_sizes: Sequence[int],
    config: MultisessionTrackingConfig,
) -> Any:
    """Call the solver while tolerating small API drift across PR revisions."""

    attempts: list[dict[str, Any]] = [
        {
            "session_sizes": list(session_sizes),
            "start_cost": config.start_cost,
            "end_cost": config.end_cost,
            "gap_penalty": config.gap_penalty,
            "cost_threshold": config.cost_threshold,
        },
        {
            "session_sizes": list(session_sizes),
            "birth_cost": config.start_cost,
            "death_cost": config.end_cost,
            "gap_cost": config.gap_penalty,
            "cost_threshold": config.cost_threshold,
        },
        {
            "session_sizes": list(session_sizes),
            "start_cost": config.start_cost,
            "end_cost": config.end_cost,
            "gap_penalty": config.gap_penalty,
        },
        {"session_sizes": list(session_sizes)},
        {},
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return solver(pairwise_costs, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is None:
        raise RuntimeError("No multisession-solver call attempts were made")
    raise TypeError(
        "Could not call solve_multisession_assignment with any supported signature"
    ) from last_error


def _coerce_solver_tracks(raw_result: Any) -> tuple[tuple[dict[int, int], ...], float | None]:
    """Normalize different solver return conventions to ``tuple[dict, ...]``."""

    if isinstance(raw_result, dict):
        tracks = raw_result.get("tracks")
        total_cost = raw_result.get("total_cost")
    else:
        tracks = getattr(raw_result, "tracks", raw_result)
        total_cost = getattr(raw_result, "total_cost", None)

    if tracks is None:
        raise ValueError("Solver result does not contain tracks")

    normalized_tracks: list[dict[int, int]] = []
    for track in tracks:
        if isinstance(track, Mapping):
            normalized_tracks.append(
                {int(session_index): int(detection_index) for session_index, detection_index in track.items()}
            )
            continue
        raise TypeError(
            "Each returned track must be a mapping from session index to detection index"
        )

    return tuple(normalized_tracks), None if total_cost is None else float(total_cost)


def _tracks_to_matrix(tracks: Sequence[Mapping[int, int]], n_sessions: int) -> np.ndarray:
    track_matrix = np.full((len(tracks), n_sessions), -1, dtype=int)
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in track.items():
            if session_index < 0 or session_index >= n_sessions:
                raise ValueError(
                    f"Track references session index {session_index} outside 0..{n_sessions - 1}"
                )
            track_matrix[track_index, session_index] = int(detection_index)
    return track_matrix


def _track_matrix_to_roi_index_matrix(
    track_matrix: np.ndarray,
    sessions: Sequence[Track2pSession],
) -> np.ndarray:
    roi_index_matrix = np.full(track_matrix.shape, -1, dtype=int)
    for session_index, session in enumerate(sessions):
        if session.plane_data.roi_indices is None:
            lookup = np.arange(session.plane_data.n_rois, dtype=int)
        else:
            lookup = np.asarray(session.plane_data.roi_indices, dtype=int)
        valid = track_matrix[:, session_index] >= 0
        if np.any(valid):
            roi_index_matrix[valid, session_index] = lookup[track_matrix[valid, session_index]]
    return roi_index_matrix


def track_sessions_multisession(
    sessions: Sequence[Track2pSession],
    *,
    config: MultisessionTrackingConfig | None = None,
    pairwise_measurement_planes_in_reference_frames: Mapping[
        tuple[int, int], CalciumPlaneData | None
    ]
    | None = None,
    solver: Callable[..., Any] | None = None,
) -> LongitudinalTrackingResult:
    """Solve the global cross-session identity assignment problem.

    This is the step between BayesCaTrack's pairwise ROI-aware costs and a
    Track2p-style benchmark output.
    """

    sessions = list(sessions)
    config = MultisessionTrackingConfig() if config is None else config

    if len(sessions) == 0:
        return LongitudinalTrackingResult(
            tracks=tuple(),
            track_matrix=np.zeros((0, 0), dtype=int),
            track_roi_index_matrix=np.zeros((0, 0), dtype=int),
            session_names=tuple(),
            session_dates=tuple(),
            pairwise_bundles=tuple(),
            total_cost=None,
        )

    if len(sessions) == 1:
        single_tracks = tuple({0: roi_index} for roi_index in range(sessions[0].plane_data.n_rois))
        track_matrix = _tracks_to_matrix(single_tracks, 1)
        return LongitudinalTrackingResult(
            tracks=single_tracks,
            track_matrix=track_matrix,
            track_roi_index_matrix=_track_matrix_to_roi_index_matrix(track_matrix, sessions),
            session_names=(sessions[0].session_name,),
            session_dates=(
                None
                if sessions[0].session_date is None
                else sessions[0].session_date.isoformat(),
            ),
            pairwise_bundles=tuple(),
            total_cost=0.0,
        )

    pairwise_costs, pairwise_bundles = build_multisession_pairwise_costs(
        sessions,
        config=config,
        pairwise_measurement_planes_in_reference_frames=pairwise_measurement_planes_in_reference_frames,
    )

    if solver is None:
        solver = _import_multisession_solver()
    session_sizes = [session.plane_data.n_rois for session in sessions]
    raw_result = _call_multisession_solver(solver, pairwise_costs, session_sizes, config)
    tracks, total_cost = _coerce_solver_tracks(raw_result)
    track_matrix = _tracks_to_matrix(tracks, len(sessions))
    track_roi_index_matrix = _track_matrix_to_roi_index_matrix(track_matrix, sessions)

    return LongitudinalTrackingResult(
        tracks=tracks,
        track_matrix=track_matrix,
        track_roi_index_matrix=track_roi_index_matrix,
        session_names=tuple(session.session_name for session in sessions),
        session_dates=tuple(
            None if session.session_date is None else session.session_date.isoformat()
            for session in sessions
        ),
        pairwise_bundles=pairwise_bundles,
        total_cost=total_cost,
    )


def _subject_load_kwargs(
    *,
    plane_name: str,
    input_format: str,
    include_behavior: bool,
    suite2p_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {
        "plane_name": plane_name,
        "input_format": input_format,
        "include_behavior": include_behavior,
    }
    load_kwargs.update(suite2p_kwargs)
    return load_kwargs


def track_subject_multisession(  # pylint: disable=too-many-arguments
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    config: MultisessionTrackingConfig | None = None,
    pairwise_measurement_planes_in_reference_frames: Mapping[
        tuple[int, int], CalciumPlaneData | None
    ]
    | None = None,
    solver: Callable[..., Any] | None = None,
    **suite2p_kwargs: Any,
) -> LongitudinalTrackingResult:
    sessions = load_track2p_subject(
        subject_dir,
        **_subject_load_kwargs(
            plane_name=plane_name,
            input_format=input_format,
            include_behavior=include_behavior,
            suite2p_kwargs=suite2p_kwargs,
        ),
    )
    return track_sessions_multisession(
        sessions,
        config=config,
        pairwise_measurement_planes_in_reference_frames=pairwise_measurement_planes_in_reference_frames,
        solver=solver,
    )


def save_tracking_result_npz(
    result: LongitudinalTrackingResult,
    output_path: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        track_matrix=result.track_matrix,
        track_roi_index_matrix=result.track_roi_index_matrix,
        session_names=np.asarray(result.session_names, dtype=object),
        session_dates=np.asarray(result.session_dates, dtype=object),
    )
    summary = result.summary()
    summary.update({"output_path": str(output_path)})
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Global multi-session Track2p/Suite2p neuron tracking using PyRecEst "
            "assignment solvers."
        )
    )
    parser.add_argument("subject_dir", type=Path, help="Track2p-style subject directory")
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        help="Optional destination .npz file for the recovered tracks",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format to load",
    )
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load motion_energy_glob.npy when present",
    )
    parser.add_argument(
        "--include-non-cells",
        action="store_true",
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=0.5,
        help="Suite2p iscell probability threshold",
    )
    parser.add_argument(
        "--weighted-masks",
        action="store_true",
        help="Reconstruct Suite2p masks using lam weights",
    )
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )
    parser.add_argument(
        "--max-session-gap",
        type=int,
        default=1,
        help="Maximum temporal gap size for allowed inter-session links",
    )
    parser.add_argument(
        "--start-cost",
        type=float,
        default=0.2,
        help="Track-start cost passed to the PyRecEst multisession solver",
    )
    parser.add_argument(
        "--end-cost",
        type=float,
        default=0.2,
        help="Track-end cost passed to the PyRecEst multisession solver",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.3,
        help="Skipped-session penalty passed to the PyRecEst multisession solver",
    )
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=None,
        help="Optional hard cap for admissible pairwise link costs",
    )
    parser.add_argument(
        "--order",
        default="xy",
        choices=("xy", "yx"),
        help="Coordinate order used in pairwise association computations",
    )
    parser.add_argument(
        "--weighted-centroids",
        action="store_true",
        help="Use weighted centroids/covariances when masks contain weights",
    )
    parser.add_argument(
        "--velocity-variance",
        type=float,
        default=25.0,
        help="Velocity variance for the pairwise state embedding",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
        help="Small diagonal regularization added to ROI covariance matrices",
    )
    parser.add_argument(
        "--pairwise-cost-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing keyword arguments forwarded to "
            "CalciumPlaneData.build_pairwise_cost_matrix"
        ),
    )
    return parser


def _load_pairwise_cost_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.pairwise_cost_json is None:
        return None
    with Path(args.pairwise_cost_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("pairwise cost JSON must decode to an object/dict")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = MultisessionTrackingConfig(
        max_session_gap=args.max_session_gap,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        cost_threshold=args.cost_threshold,
        pairwise_cost_kwargs=_load_pairwise_cost_kwargs(args),
    )
    result = track_subject_multisession(
        args.subject_dir,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        config=config,
        **_suite2p_kwargs_from_args(args),
    )

    summary = result.summary()
    if args.output_path is not None:
        summary = save_tracking_result_npz(result, args.output_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
