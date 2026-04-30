"""Synthetic Track2p/Suite2p fixtures for benchmark development."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CenterYX = tuple[float, float]


@dataclass(frozen=True)
class SyntheticFalsePositiveRoi:
    """One synthetic ROI that is not part of the ground-truth tracks."""

    session_index: int
    center_yx: CenterYX
    cell_probability: float = 0.95


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class SyntheticTrack2pSubjectConfig:
    """Configuration for a deterministic synthetic Track2p-style subject."""

    subject_name: str = "jm_synthetic"
    session_names: tuple[str, ...] = (
        "2024-05-01_a",
        "2024-05-02_a",
        "2024-05-03_a",
    )
    plane_name: str = "plane0"
    image_shape: tuple[int, int] = (32, 32)
    base_centers_yx: tuple[CenterYX, ...] = (
        (8.0, 8.0),
        (8.0, 22.0),
        (22.0, 8.0),
        (22.0, 22.0),
    )
    drift_per_session_yx: CenterYX = (0.5, 1.0)
    roi_radius: int = 2
    n_timepoints: int = 5
    missing_detections: tuple[tuple[int, int], ...] = ()
    non_cell_tracks: tuple[int, ...] = ()
    false_positive_rois: tuple[SyntheticFalsePositiveRoi, ...] = ()
    cell_probability: float = 0.95
    non_cell_probability: float = 0.1

    def __post_init__(self) -> None:
        if not self.subject_name:
            raise ValueError("subject_name must not be empty")
        if not self.session_names:
            raise ValueError("session_names must not be empty")
        if not self.base_centers_yx:
            raise ValueError("base_centers_yx must not be empty")
        if self.roi_radius < 0:
            raise ValueError("roi_radius must be non-negative")
        if self.n_timepoints < 1:
            raise ValueError("n_timepoints must be at least one")
        if len(self.image_shape) != 2 or any(
            int(size) <= 0 for size in self.image_shape
        ):
            raise ValueError("image_shape must contain two positive dimensions")
        _validate_probability(self.cell_probability, "cell_probability")
        _validate_probability(self.non_cell_probability, "non_cell_probability")

        n_tracks = len(self.base_centers_yx)
        n_sessions = len(self.session_names)
        for track_index, session_index in self.missing_detections:
            _validate_track_session_index(
                track_index, session_index, n_tracks, n_sessions
            )
        for track_index in self.non_cell_tracks:
            if track_index < 0 or track_index >= n_tracks:
                raise ValueError(
                    f"non_cell_tracks contains invalid track index {track_index}"
                )
        for false_positive in self.false_positive_rois:
            if (
                false_positive.session_index < 0
                or false_positive.session_index >= n_sessions
            ):
                raise ValueError(
                    f"false_positive_rois contains invalid session index {false_positive.session_index}"
                )
            _validate_probability(
                false_positive.cell_probability, "false_positive.cell_probability"
            )


@dataclass(frozen=True)
class SyntheticTrack2pSubject:
    """Paths and reference matrix for a generated synthetic subject."""

    subject_dir: Path
    ground_truth_csv: Path
    session_names: tuple[str, ...]
    plane_name: str
    suite2p_indices: np.ndarray
    stat_rows_per_session: tuple[int, ...]


def write_synthetic_track2p_subject(
    root_dir: str | Path,
    config: SyntheticTrack2pSubjectConfig | None = None,
) -> SyntheticTrack2pSubject:
    """Write a deterministic Suite2p-style subject plus ``ground_truth.csv``.

    The ground-truth matrix uses Suite2p ``stat.npy`` row indices. Tracks listed
    in ``non_cell_tracks`` are still written into ``ground_truth.csv`` but are
    marked as non-cells in ``iscell.npy``; this intentionally exercises the
    benchmark's ``--include-non-cells`` validation path.
    """

    config = SyntheticTrack2pSubjectConfig() if config is None else config
    root_dir = Path(root_dir)
    subject_dir = root_dir / config.subject_name
    subject_dir.mkdir(parents=True, exist_ok=True)

    suite2p_indices = np.empty(
        (len(config.base_centers_yx), len(config.session_names)), dtype=object
    )
    suite2p_indices[:] = None
    stat_rows_per_session: list[int] = []
    missing = set(config.missing_detections)

    for session_index, session_name in enumerate(config.session_names):
        stat_rows = _write_session(
            subject_dir,
            session_index=session_index,
            session_name=session_name,
            config=config,
            suite2p_indices=suite2p_indices,
            missing=missing,
        )
        stat_rows_per_session.append(stat_rows)

    ground_truth_csv = subject_dir / "ground_truth.csv"
    _write_ground_truth_csv(
        ground_truth_csv,
        session_names=config.session_names,
        suite2p_indices=suite2p_indices,
    )

    return SyntheticTrack2pSubject(
        subject_dir=subject_dir,
        ground_truth_csv=ground_truth_csv,
        session_names=config.session_names,
        plane_name=config.plane_name,
        suite2p_indices=suite2p_indices,
        stat_rows_per_session=tuple(stat_rows_per_session),
    )


def _write_session(  # pylint: disable=too-many-locals
    subject_dir: Path,
    *,
    session_index: int,
    session_name: str,
    config: SyntheticTrack2pSubjectConfig,
    suite2p_indices: np.ndarray,
    missing: set[tuple[int, int]],
) -> int:
    plane_dir = subject_dir / session_name / "suite2p" / config.plane_name
    plane_dir.mkdir(parents=True, exist_ok=True)

    stat_entries: list[dict[str, Any]] = []
    iscell_rows: list[tuple[float, float]] = []
    traces: list[np.ndarray] = []
    mean_image = np.zeros(config.image_shape, dtype=float)

    for track_index, base_center in enumerate(config.base_centers_yx):
        if (track_index, session_index) in missing:
            continue
        center = _drifted_center(
            base_center, session_index, config.drift_per_session_yx
        )
        stat_entries.append(_stat_entry(center, config=config))
        suite2p_indices[track_index, session_index] = len(stat_entries) - 1

        is_cell = track_index not in set(config.non_cell_tracks)
        probability = (
            config.cell_probability if is_cell else config.non_cell_probability
        )
        iscell_rows.append((1.0 if is_cell else 0.0, probability))
        traces.append(_trace(track_index, session_index, config.n_timepoints))
        _paint_mean_image(mean_image, stat_entries[-1], intensity=1.0 + track_index)

    for false_positive_index, false_positive in enumerate(
        _false_positives_for_session(config.false_positive_rois, session_index)
    ):
        stat_entries.append(_stat_entry(false_positive.center_yx, config=config))
        iscell_rows.append((1.0, false_positive.cell_probability))
        traces.append(
            _trace(100 + false_positive_index, session_index, config.n_timepoints)
        )
        _paint_mean_image(mean_image, stat_entries[-1], intensity=0.5)

    np.save(
        plane_dir / "stat.npy",
        np.asarray(stat_entries, dtype=object),
        allow_pickle=True,
    )
    np.save(plane_dir / "iscell.npy", np.asarray(iscell_rows, dtype=float))
    np.save(
        plane_dir / "ops.npy",
        np.asarray(
            {
                "Ly": config.image_shape[0],
                "Lx": config.image_shape[1],
                "meanImg": mean_image,
            },
            dtype=object,
        ),
        allow_pickle=True,
    )
    trace_matrix = (
        np.vstack(traces) if traces else np.zeros((0, config.n_timepoints), dtype=float)
    )
    np.save(plane_dir / "F.npy", trace_matrix)
    return len(stat_entries)


def _write_ground_truth_csv(
    output_path: Path,
    *,
    session_names: tuple[str, ...],
    suite2p_indices: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", *session_names])
        for track_index, row in enumerate(suite2p_indices):
            writer.writerow([track_index, *[_csv_roi_value(value) for value in row]])


def _csv_roi_value(value: object) -> int:
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"Expected integer Suite2p index or None, got {value!r}")


def _drifted_center(
    base_center: CenterYX, session_index: int, drift_per_session: CenterYX
) -> CenterYX:
    return (
        float(base_center[0]) + session_index * float(drift_per_session[0]),
        float(base_center[1]) + session_index * float(drift_per_session[1]),
    )


def _stat_entry(
    center_yx: CenterYX, *, config: SyntheticTrack2pSubjectConfig
) -> dict[str, Any]:
    ypix, xpix = _square_roi_pixels(
        center_yx,
        image_shape=config.image_shape,
        radius=config.roi_radius,
    )
    npix = int(ypix.shape[0])
    return {
        "ypix": ypix,
        "xpix": xpix,
        "lam": np.ones((npix,), dtype=float),
        "overlap": np.zeros((npix,), dtype=bool),
        "med": np.asarray(center_yx, dtype=float),
        "npix": float(npix),
        "radius": float(max(config.roi_radius, 1)),
        "compact": 1.0,
        "footprint": float(npix),
    }


def _square_roi_pixels(
    center_yx: CenterYX,
    *,
    image_shape: tuple[int, int],
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    center_y = int(round(float(center_yx[0])))
    center_x = int(round(float(center_yx[1])))
    y_start = max(0, center_y - radius)
    y_stop = min(int(image_shape[0]), center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_stop = min(int(image_shape[1]), center_x + radius + 1)
    if y_start >= y_stop or x_start >= x_stop:
        raise ValueError(
            f"ROI centered at {center_yx!r} falls outside image_shape {image_shape!r}"
        )
    grid_y, grid_x = np.meshgrid(
        np.arange(y_start, y_stop, dtype=int),
        np.arange(x_start, x_stop, dtype=int),
        indexing="ij",
    )
    return grid_y.reshape(-1), grid_x.reshape(-1)


def _paint_mean_image(
    mean_image: np.ndarray, stat_entry: dict[str, Any], *, intensity: float
) -> None:
    ypix = np.asarray(stat_entry["ypix"], dtype=int)
    xpix = np.asarray(stat_entry["xpix"], dtype=int)
    mean_image[ypix, xpix] += float(intensity)


def _trace(track_index: int, session_index: int, n_timepoints: int) -> np.ndarray:
    time = np.arange(n_timepoints, dtype=float)
    return float(track_index + 1) + float(session_index) / 10.0 + time / 100.0


def _false_positives_for_session(
    false_positive_rois: Iterable[SyntheticFalsePositiveRoi],
    session_index: int,
) -> tuple[SyntheticFalsePositiveRoi, ...]:
    return tuple(
        false_positive
        for false_positive in false_positive_rois
        if false_positive.session_index == session_index
    )


def _validate_track_session_index(
    track_index: int, session_index: int, n_tracks: int, n_sessions: int
) -> None:
    if track_index < 0 or track_index >= n_tracks:
        raise ValueError(f"Invalid track index {track_index}")
    if session_index < 0 or session_index >= n_sessions:
        raise ValueError(f"Invalid session index {session_index}")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


__all__ = (
    "SyntheticFalsePositiveRoi",
    "SyntheticTrack2pSubject",
    "SyntheticTrack2pSubjectConfig",
    "write_synthetic_track2p_subject",
)
