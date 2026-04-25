# jscpd:ignore-start
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
# jscpd:ignore-end

from track2p_pyrecest_bridge import CalciumPlaneData, load_track2p_subject  # noqa: E402
from track2p_pyrecest_bridge.track2p_registration import (  # noqa: E402
    build_registered_subject_association_bundles,
    register_consecutive_session_measurement_planes,
    register_plane_pair,
)


def _make_shift_left_registration_backend():
    def reg_img_elastix(reference_fov, moving_fov, track_ops):
        del reference_fov
        assert track_ops.transform_type in {"affine", "rigid"}
        registered_fov = np.zeros_like(moving_fov)
        registered_fov[:, :-1] = moving_fov[:, 1:]
        return registered_fov, {"shift_y": 0, "shift_x": -1}

    def itk_reg_all_roi(all_roi_hw_n, reg_params):
        shift_y = int(reg_params.get("shift_y", 0))
        shift_x = int(reg_params.get("shift_x", 0))
        result = np.zeros_like(all_roi_hw_n)
        src_y_start = max(0, -shift_y)
        src_y_end = all_roi_hw_n.shape[0] - max(0, shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        src_x_start = max(0, -shift_x)
        src_x_end = all_roi_hw_n.shape[1] - max(0, shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = all_roi_hw_n[
            src_y_start:src_y_end, src_x_start:src_x_end, :
        ]
        return result

    return reg_img_elastix, itk_reg_all_roi


def test_register_plane_pair_uses_track2p_registration_backend(monkeypatch):
    reference_masks = np.zeros((2, 4, 5), dtype=bool)
    reference_masks[0, 1, 1] = True
    reference_masks[1, 2, 3] = True
    moving_masks = np.zeros((2, 4, 5), dtype=bool)
    moving_masks[0, 1, 2] = True
    moving_masks[1, 2, 4] = True
    reference_plane = CalciumPlaneData(roi_masks=reference_masks, fov=np.arange(20).reshape(4, 5))
    moving_plane = CalciumPlaneData(
        roi_masks=moving_masks,
        fov=np.arange(20, 40).reshape(4, 5),
        roi_indices=np.array([5, 7], dtype=int),
        source="raw_npy",
    )
    monkeypatch.setattr(
        "track2p_pyrecest_bridge.track2p_registration._load_track2p_registration_backend",
        _make_shift_left_registration_backend,
    )
    registered_plane = register_plane_pair(reference_plane, moving_plane, transform_type="affine")
    npt.assert_array_equal(registered_plane.roi_masks, reference_masks)
    npt.assert_array_equal(registered_plane.roi_indices, moving_plane.roi_indices)
    assert registered_plane.source == "raw_npy_registered"


def test_build_registered_subject_association_bundles_registers_before_costing(
    tmp_path,
    monkeypatch,
    registered_pair_masks,
    write_raw_npy_session,
):
    subject_dir = tmp_path / "jm271"
    ref_masks, mov_masks_raw, _ = registered_pair_masks
    for session_name, roi_masks, offset in (
        ("2024-05-01_a", ref_masks, 0.0),
        ("2024-05-02_a", mov_masks_raw, 10.0),
    ):
        write_raw_npy_session(subject_dir, session_name, roi_masks, offset=offset)
    monkeypatch.setattr(
        "track2p_pyrecest_bridge.track2p_registration._load_track2p_registration_backend",
        _make_shift_left_registration_backend,
    )
    sessions = load_track2p_subject(subject_dir, plane_name="plane0", input_format="auto")
    registered_planes = register_consecutive_session_measurement_planes(sessions, transform_type="affine")
    assert len(registered_planes) == 1
    bundles = build_registered_subject_association_bundles(
        subject_dir,
        plane_name="plane0",
        input_format="auto",
        transform_type="affine",
        pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
    )
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.pairwise_cost_matrix[0, 0] < bundle.pairwise_cost_matrix[0, 1]
    assert bundle.pairwise_cost_matrix[1, 1] < bundle.pairwise_cost_matrix[1, 0]
    npt.assert_allclose(np.diag(bundle.pairwise_components["iou"]), np.ones(2))
