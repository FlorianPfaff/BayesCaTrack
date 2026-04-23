import json
import os
import subprocess  # nosec B404
import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
MODULE_RUN_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC_PATH)
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
}

from track2p_pyrecest_bridge import (  # noqa: E402
    CalciumPlaneData,
    build_consecutive_session_association_bundles,
    build_session_pair_association_bundle,
    find_track2p_session_dirs,
    load_raw_npy_plane,
    load_suite2p_plane,
    load_track2p_subject,
)


def _install_fake_pyrecest_modules(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_distributions = types.ModuleType("pyrecest.distributions")
    fake_filters = types.ModuleType("pyrecest.filters")
    fake_kf_module = types.ModuleType("pyrecest.filters.kalman_filter")

    class GaussianDistribution:
        def __init__(self, mu, C):
            self.mu = np.asarray(mu)
            self.C = np.asarray(C)

    class KalmanFilter:
        def __init__(self, initial_state):
            self.initial_state = initial_state

    fake_distributions.GaussianDistribution = GaussianDistribution
    fake_kf_module.KalmanFilter = KalmanFilter

    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.distributions", fake_distributions)
    monkeypatch.setitem(sys.modules, "pyrecest.filters", fake_filters)
    monkeypatch.setitem(sys.modules, "pyrecest.filters.kalman_filter", fake_kf_module)


def test_calcium_plane_data_centroids_state_moments_and_pyrecest_adapter(monkeypatch):
    roi_masks = np.zeros((1, 3, 4), dtype=bool)
    roi_masks[0, 0, 1] = True
    roi_masks[0, 2, 1] = True
    plane = CalciumPlaneData(roi_masks=roi_masks)

    npt.assert_allclose(plane.centroids(order="xy"), np.array([[1.0], [1.0]]))
    npt.assert_allclose(plane.centroids(order="yx"), np.array([[1.0], [1.0]]))

    position_covariances = plane.position_covariances(order="xy", regularization=0.0)
    expected_position_covariance = np.array([[[0.0], [0.0]], [[0.0], [1.0]]]).reshape((2, 2, 1))
    npt.assert_allclose(position_covariances, expected_position_covariance)

    means, covariances = plane.to_constant_velocity_state_moments(
        order="xy",
        velocity_variance=9.0,
        regularization=0.0,
    )
    npt.assert_allclose(means, np.array([[1.0], [0.0], [1.0], [0.0]]))
    expected_covariance = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
        ]
    )
    npt.assert_allclose(covariances[:, :, 0], expected_covariance)

    _install_fake_pyrecest_modules(monkeypatch)
    gaussians = plane.to_pyrecest_gaussian_distributions(order="xy", velocity_variance=9.0, regularization=0.0)
    filters = plane.to_pyrecest_kalman_filters(order="xy", velocity_variance=9.0, regularization=0.0)
    assert len(gaussians) == 1
    assert len(filters) == 1
    npt.assert_allclose(gaussians[0].mu, np.array([1.0, 0.0, 1.0, 0.0]))
    npt.assert_allclose(gaussians[0].C, expected_covariance)
    assert filters[0].initial_state is gaussians[0] or np.allclose(filters[0].initial_state.mu, gaussians[0].mu)


def test_load_suite2p_plane_filters_cells_and_reconstructs_weighted_masks(tmp_path):
    plane_dir = tmp_path / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.array(
        [
            {
                "ypix": np.array([0, 0, 1]),
                "xpix": np.array([0, 1, 1]),
                "lam": np.array([1.0, 2.0, 1.0]),
                "overlap": np.array([False, True, False]),
                "radius": 2.0,
                "aspect_ratio": 1.1,
                "compact": 1.2,
                "footprint": 3.0,
                "skew": 0.5,
                "std": 4.0,
                "npix": 3,
                "npix_norm": 1.0,
            },
            {
                "ypix": np.array([2, 2]),
                "xpix": np.array([2, 3]),
                "lam": np.array([1.0, 1.0]),
                "overlap": np.array([False, False]),
                "radius": 1.0,
                "aspect_ratio": 1.0,
                "compact": 1.0,
                "footprint": 2.0,
                "skew": 0.0,
                "std": 1.0,
                "npix": 2,
                "npix_norm": 0.5,
            },
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9], [0.0, 0.1]]))
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]))
    np.save(plane_dir / "spks.npy", np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]]))
    np.save(plane_dir / "ops.npy", {"Ly": 4, "Lx": 5, "meanImg": np.arange(20).reshape(4, 5)}, allow_pickle=True)

    plane = load_suite2p_plane(
        plane_dir,
        include_non_cells=False,
        weighted_masks=True,
        exclude_overlapping_pixels=True,
    )

    assert plane.n_rois == 1
    assert plane.source == "suite2p"
    npt.assert_allclose(plane.cell_probabilities, np.array([0.9]))
    npt.assert_array_equal(plane.roi_indices, np.array([0]))
    npt.assert_allclose(plane.traces, np.array([[1.0, 2.0, 3.0]]))
    npt.assert_allclose(plane.spike_traces, np.array([[0.1, 0.2, 0.3]]))
    npt.assert_allclose(plane.fov, np.arange(20).reshape(4, 5))

    expected_mask = np.zeros((4, 5), dtype=float)
    expected_mask[0, 0] = 1.0
    expected_mask[1, 1] = 1.0
    npt.assert_allclose(plane.roi_masks[0], expected_mask)

    weighted_centroid = plane.centroids(order="xy", weighted=True)
    npt.assert_allclose(weighted_centroid, np.array([[0.5], [0.5]]))
    npt.assert_allclose(plane.roi_features["radius"], np.array([2.0]))
    npt.assert_allclose(plane.roi_features["npix"], np.array([3.0]))


def test_load_raw_npy_plane_validates_shapes(tmp_path):
    plane_dir = tmp_path / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((2, 3, 4), dtype=bool)
    roi_masks[0, 0, 0] = True
    roi_masks[1, 1, 2] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.arange(10).reshape(2, 5))
    np.save(plane_dir / "fov.npy", np.arange(12).reshape(3, 4))

    plane = load_raw_npy_plane(plane_dir)
    assert plane.n_rois == 2
    assert plane.source == "raw_npy"
    npt.assert_array_equal(plane.roi_indices, np.array([0, 1]))
    npt.assert_allclose(plane.to_measurement_matrix(order="xy"), np.array([[0.0, 2.0], [0.0, 1.0]]))


def test_find_and_load_track2p_subject_sorts_sessions_and_loads_behavior(tmp_path):
    subject_dir = tmp_path / "jm999"
    session_a = subject_dir / "2024-05-02_a"
    session_b = subject_dir / "2024-05-01_a"
    ignored_dir = subject_dir / "notes"
    ignored_dir.mkdir(parents=True)

    for session_dir, offset in ((session_a, 10), (session_b, 0)):
        plane_dir = session_dir / "data_npy" / "plane0"
        plane_dir.mkdir(parents=True)
        roi_masks = np.zeros((1, 2, 2), dtype=bool)
        roi_masks[0, 0, 0] = True
        np.save(plane_dir / "rois.npy", roi_masks)
        np.save(plane_dir / "F.npy", np.array([[offset, offset + 1, offset + 2]], dtype=float))
        np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float) * offset)

    move_deve_dir = session_b / "move_deve"
    move_deve_dir.mkdir(parents=True)
    np.save(move_deve_dir / "motion_energy_glob.npy", np.array([1.0, 2.0, 3.0]))

    session_dirs = find_track2p_session_dirs(subject_dir)
    assert [path.name for path in session_dirs] == ["2024-05-01_a", "2024-05-02_a"]

    sessions = load_track2p_subject(subject_dir, plane_name="plane0", input_format="auto", include_behavior=True)
    assert [session.session_name for session in sessions] == ["2024-05-01_a", "2024-05-02_a"]
    assert str(sessions[0].session_date) == "2024-05-01"
    assert str(sessions[1].session_date) == "2024-05-02"
    npt.assert_allclose(sessions[0].motion_energy, np.array([1.0, 2.0, 3.0]))
    assert sessions[1].motion_energy is None
    npt.assert_allclose(sessions[0].plane_data.traces, np.array([[0.0, 1.0, 2.0]]))
    npt.assert_allclose(sessions[1].plane_data.traces, np.array([[10.0, 11.0, 12.0]]))


def test_registered_pairwise_costs_and_association_bundles(  # pylint: disable=too-many-locals
    tmp_path,
    registered_pair_masks,
    write_raw_npy_session,
):
    subject_dir = tmp_path / "jm314"
    ref_masks, mov_masks_raw, mov_masks_registered = registered_pair_masks

    for session_name, roi_masks, offset in (
        ("2024-05-01_a", ref_masks, 0.0),
        ("2024-05-02_a", mov_masks_raw, 10.0),
    ):
        write_raw_npy_session(subject_dir, session_name, roi_masks, offset=offset)

    sessions = load_track2p_subject(subject_dir, plane_name="plane0", input_format="auto")
    assert len(sessions) == 2

    registered_measurement_plane = sessions[1].plane_data.with_replaced_masks(
        mov_masks_registered,
        source="registered_npy",
    )
    npt.assert_array_equal(
        registered_measurement_plane.roi_indices,
        sessions[1].plane_data.roi_indices,
    )

    pairwise_cost_matrix, components = sessions[0].plane_data.build_pairwise_cost_matrix(
        registered_measurement_plane,
        max_centroid_distance=5.0,
        roi_feature_weight=0.0,
        return_components=True,
    )
    assert pairwise_cost_matrix.shape == (2, 2)
    assert pairwise_cost_matrix[0, 0] < pairwise_cost_matrix[0, 1]
    assert pairwise_cost_matrix[1, 1] < pairwise_cost_matrix[1, 0]
    npt.assert_allclose(np.diag(components["iou"]), np.ones(2))
    assert np.all(np.diag(components["mask_cosine_similarity"]) > 0.99)

    bundle = build_session_pair_association_bundle(
        sessions[0],
        sessions[1],
        measurement_plane_in_reference_frame=registered_measurement_plane,
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )
    npt.assert_allclose(
        bundle.measurement_matrix,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    )
    npt.assert_allclose(
        bundle.measurements,
        registered_measurement_plane.to_measurement_matrix(order="xy"),
    )
    npt.assert_allclose(bundle.pairwise_cost_matrix, pairwise_cost_matrix)
    npt.assert_array_equal(bundle.reference_roi_indices, np.array([0, 1]))
    npt.assert_array_equal(bundle.measurement_roi_indices, np.array([0, 1]))
    update_kwargs = bundle.to_pyrecest_update_kwargs()
    assert set(update_kwargs) == {"measurements", "measurement_matrix", "covMatsMeas", "pairwise_cost_matrix"}

    sequential_bundles = build_consecutive_session_association_bundles(
        sessions,
        measurement_planes_in_reference_frames=[registered_measurement_plane],
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )
    assert len(sequential_bundles) == 1
    npt.assert_allclose(sequential_bundles[0].pairwise_cost_matrix, pairwise_cost_matrix)


def test_cli_summary_and_export(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float))

    summary_proc = subprocess.run(  # nosec B603
        [sys.executable, "-m", "track2p_pyrecest_bridge", "summary", str(subject_dir)],
        check=True,
        env=MODULE_RUN_ENV,
        capture_output=True,
        text=True,
    )
    summary = json.loads(summary_proc.stdout)
    assert summary["n_sessions"] == 1
    assert summary["sessions"][0]["n_rois"] == 1

    output_path = tmp_path / "jm123_plane0.npz"
    export_proc = subprocess.run(  # nosec B603
        [sys.executable, "-m", "track2p_pyrecest_bridge", "export", str(subject_dir), str(output_path)],
        check=True,
        env=MODULE_RUN_ENV,
        capture_output=True,
        text=True,
    )
    export_summary = json.loads(export_proc.stdout)
    assert export_summary["n_sessions"] == 1
    assert output_path.exists()
    exported = np.load(output_path, allow_pickle=True)
    npt.assert_allclose(exported["session_0__measurements"], np.array([[1.0], [0.0]]))
    npt.assert_allclose(exported["session_0__state_means"], np.array([[1.0], [0.0], [0.0], [0.0]]))
