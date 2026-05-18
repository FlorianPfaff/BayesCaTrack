import json

import numpy as np
import numpy.testing as npt
from bayescatrack import CalciumPlaneData
from bayescatrack.association.calibrated_costs import (
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    pairwise_feature_tensor,
)
from tests._support import run_module


def test_calcium_plane_data_builds_measurements_and_state_moments():
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 2] = True
    plane = CalciumPlaneData(roi_masks=roi_masks)

    npt.assert_allclose(
        plane.to_measurement_matrix(order="xy"), np.array([[2.0], [1.0]])
    )
    means, covariances = plane.to_constant_velocity_state_moments(order="xy")

    assert means.shape == (4, 1)
    assert covariances.shape == (4, 4, 1)


def test_local_image_and_weighted_mask_evidence_components_rank_diagonal_pairs():
    image_shape = (12, 12)
    reference_masks = np.zeros((2, *image_shape), dtype=float)
    measurement_masks = np.zeros((2, *image_shape), dtype=float)
    reference_masks[0, 2:4, 2:4] = np.array([[1.0, 2.0], [3.0, 4.0]])
    measurement_masks[0, 2:4, 2:4] = np.array([[4.0, 3.0], [2.0, 1.0]])
    reference_masks[1, 7:9, 7:9] = 1.0
    measurement_masks[1, 7:9, 7:9] = 1.0

    fov = np.zeros(image_shape, dtype=float)
    fov[3, 1:6] = 1.0
    fov[6:11, 8] = 1.0
    reference = CalciumPlaneData(roi_masks=reference_masks, fov=fov)
    measurement = CalciumPlaneData(roi_masks=measurement_masks, fov=fov.copy())

    cost, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        weighted_dice_weight=1.0,
        overlap_fraction_weight=1.0,
        distance_transform_weight=1.0,
        image_patch_weight=1.0,
        centroid_rank_weight=1.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=0.0,
        local_evidence_components=True,
        patch_radius=2,
        return_components=True,
    )

    assert cost.shape == (2, 2)
    assert np.all(np.isfinite(cost))
    assert components["weighted_dice_similarity"][0, 0] > components[
        "weighted_dice_similarity"
    ][0, 1]
    assert components["overlap_min_fraction"][0, 0] > components[
        "overlap_min_fraction"
    ][0, 1]
    assert components["distance_transform_cost"][0, 0] < components[
        "distance_transform_cost"
    ][0, 1]
    assert components["image_patch_cost"][0, 0] < components["image_patch_cost"][0, 1]
    assert components["centroid_rank_cost"][0, 0] == 0.0
    assert cost[0, 0] < cost[0, 1]
    assert cost[1, 1] < cost[1, 0]

    features = pairwise_feature_tensor(
        components,
        feature_names=LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    )
    assert features.shape == (2, 2, len(LOCAL_EVIDENCE_ASSOCIATION_FEATURES))
    assert features[0, 0, 0] < features[0, 1, 0]


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
    export_proc = run_module(
        "-m", "bayescatrack", "export", str(subject_dir), str(output_path)
    )
    export_summary = json.loads(export_proc.stdout)
    assert export_summary["n_sessions"] == 1
    assert output_path.exists()
