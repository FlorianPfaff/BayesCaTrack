"""Regression tests for Suite2p loader side-array validation."""

from pathlib import Path

import numpy as np
import pytest

from bayescatrack import load_suite2p_plane


def _write_minimal_suite2p_plane(plane_dir: Path, *, n_rois: int = 2) -> None:
    plane_dir.mkdir(parents=True)

    stat = np.empty((n_rois,), dtype=object)
    for roi_index in range(n_rois):
        stat[roi_index] = {
            "ypix": np.asarray([roi_index + 1], dtype=int),
            "xpix": np.asarray([roi_index + 1], dtype=int),
            "lam": np.asarray([1.0], dtype=float),
            "npix": 1.0,
        }

    np.save(plane_dir / "stat.npy", stat)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": n_rois + 3, "Lx": n_rois + 3, "meanImg": np.zeros((n_rois + 3, n_rois + 3))},
    )
    np.save(plane_dir / "iscell.npy", np.column_stack([np.ones(n_rois), np.full(n_rois, 0.99)]))
    np.save(plane_dir / "F.npy", np.zeros((n_rois, 4)))
    np.save(plane_dir / "spks.npy", np.zeros((n_rois, 4)))
    np.save(plane_dir / "Fneu.npy", np.zeros((n_rois, 4)))


@pytest.mark.parametrize(
    ("filename", "loader_kwargs"),
    [
        ("F.npy", {"load_traces": True}),
        ("spks.npy", {"load_spike_traces": True}),
        ("Fneu.npy", {"load_neuropil_traces": True}),
    ],
)
def test_suite2p_loader_rejects_trace_side_arrays_with_wrong_roi_count(
    tmp_path: Path,
    filename: str,
    loader_kwargs: dict[str, bool],
) -> None:
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir, n_rois=2)
    np.save(plane_dir / filename, np.zeros((3, 4)))

    with pytest.raises(ValueError, match=rf"{filename} first dimension .* stat\.npy"):
        load_suite2p_plane(
            plane_dir,
            load_traces=False,
            load_spike_traces=False,
            load_neuropil_traces=False,
            **loader_kwargs,
        )


def test_suite2p_loader_rejects_iscell_with_wrong_roi_count(tmp_path: Path) -> None:
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir, n_rois=2)
    np.save(plane_dir / "iscell.npy", np.ones((1, 2)))

    with pytest.raises(ValueError, match=r"iscell\.npy first dimension .* stat\.npy"):
        load_suite2p_plane(plane_dir)


def test_suite2p_loader_keeps_selected_rows_after_validation(tmp_path: Path) -> None:
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir, n_rois=3)
    iscell = np.asarray([[1.0, 0.95], [0.0, 0.20], [1.0, 0.90]])
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(plane_dir / "F.npy", np.arange(12, dtype=float).reshape(3, 4))

    plane = load_suite2p_plane(plane_dir, load_spike_traces=False)

    np.testing.assert_array_equal(plane.roi_indices, np.asarray([0, 2]))
    np.testing.assert_array_equal(plane.traces, np.asarray([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]))
