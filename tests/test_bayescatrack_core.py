import json

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData
from tests._support import run_module


def test_calcium_plane_data_builds_measurements_and_state_moments():
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 2] = True
    plane = CalciumPlaneData(roi_masks=roi_masks)

    npt.assert_allclose(plane.to_measurement_matrix(order="xy"), np.array([[2.0], [1.0]]))
    means, covariances = plane.to_constant_velocity_state_moments(order="xy")

    assert means.shape == (4, 1)
    assert covariances.shape == (4, 4, 1)


def test_cli_summary_and_export(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float))

    summary_proc = run_module("-m", "bayescatrack", "summary", str(subject_dir))
    summary = json.loads(summary_proc.stdout)
    assert summary["n_sessions"] == 1
    assert summary["sessions"][0]["n_rois"] == 1

    output_path = tmp_path / "jm123_plane0.npz"
    export_proc = run_module("-m", "bayescatrack", "export", str(subject_dir), str(output_path))
    export_summary = json.loads(export_proc.stdout)
    assert export_summary["n_sessions"] == 1
    assert output_path.exists()
