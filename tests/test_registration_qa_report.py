from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bayescatrack.experiments.registration_qa_report import (
    RegistrationQAConfig,
    format_registration_backend_audit_table,
    format_registration_qa_table,
    run_registration_qa_report,
    summarize_registration_backend_usage,
    summarize_registration_qa_links,
)


def _write_ground_truth_csv(
    subject_dir: Path,
    session_names: tuple[str, ...],
    rows: tuple[tuple[int, ...], ...],
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def _write_raw_npy_masks(
    subject_dir: Path, session_name: str, masks: np.ndarray
) -> None:
    plane_dir = subject_dir / session_name / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    np.save(plane_dir / "rois.npy", masks)
    np.save(plane_dir / "F.npy", np.zeros((masks.shape[0], 2), dtype=float))
    np.save(plane_dir / "fov.npy", np.asarray(masks, dtype=float).sum(axis=0))


def test_registration_qa_report_summarizes_manual_gt_links(
    tmp_path,
    write_raw_npy_session,
):
    subject_dir = tmp_path / "jm001"
    masks: np.ndarray = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 3:5, 3:5] = True
    write_raw_npy_session(subject_dir, "2024-05-01_a", masks, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks.copy(), offset=1.0)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (1, 1)),
    )

    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=subject_dir,
            reference_kind="manual-gt",
            input_format="npy",
            transform_type="none",
            max_gap=1,
            cost="registered-iou",
        )
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["cost"] == "registered-iou"
    assert first["registration_backend"] == "none"
    assert first["registered_plane_source"] == "raw_npy"
    assert first["registration_backend_reason"] == "transform_type=none"
    assert first["registered_iou"] == pytest.approx(1.0)
    assert first["raw_iou"] == pytest.approx(1.0)
    assert first["registered_centroid_distance"] == pytest.approx(0.0)
    assert np.isnan(first["gt_probability"])
    assert first["gt_rank"] == 1
    assert first["gt_is_top1"] is True
    assert first["gt_is_top5"] is True
    assert first["gt_is_top10"] is True
    assert first["gt_cost_percentile"] == pytest.approx(0.0)
    assert first["candidate_count"] == 2
    assert first["finite_candidate_count"] == 2
    assert first["finite_false_candidate_count"] == 1
    assert first["false_cost_min"] > first["gt_cost"]
    assert first["false_cost_median"] == pytest.approx(first["false_cost_min"])
    assert first["gt_candidate_admissible"] is True

    summary = summarize_registration_qa_links(rows)
    assert len(summary) == 1
    assert summary[0]["n_gt_links"] == 2
    assert summary[0]["median_registered_iou"] == pytest.approx(1.0)
    assert summary[0]["median_registered_centroid_distance"] == pytest.approx(0.0)
    assert summary[0]["gt_top1_rate"] == pytest.approx(1.0)
    assert summary[0]["gt_recall_at_1"] == pytest.approx(1.0)
    assert summary[0]["gt_recall_at_5"] == pytest.approx(1.0)
    assert summary[0]["gt_recall_at_10"] == pytest.approx(1.0)
    assert summary[0]["median_gt_cost_percentile"] == pytest.approx(0.0)
    assert summary[0]["median_finite_candidate_count"] == pytest.approx(2.0)
    assert summary[0]["median_finite_false_candidate_count"] == pytest.approx(1.0)
    assert summary[0]["median_false_cost_min"] > 0.0
    assert summary[0]["median_false_cost_median"] == pytest.approx(
        summary[0]["median_false_cost_min"]
    )
    assert summary[0]["gt_admissible_rate"] == pytest.approx(1.0)

    table = format_registration_qa_table(summary)
    assert "median_registered_iou" in table
    assert "gt_recall_at_5" in table
    assert "jm001" in table

    backend_audit = summarize_registration_backend_usage(rows)
    assert len(backend_audit) == 1
    assert backend_audit[0]["cost"] == "registered-iou"
    assert backend_audit[0]["registration_backend"] == "none"
    assert backend_audit[0]["transform_type"] == "none"
    assert backend_audit[0]["registered_plane_source"] == "raw_npy"
    assert backend_audit[0]["registration_backend_reason"] == "transform_type=none"
    assert backend_audit[0]["edge_count"] == 1
    assert backend_audit[0]["gt_link_rows"] == 2
    assert backend_audit[0]["subject_count"] == 1
    assert backend_audit[0]["subjects"] == "jm001"
    assert np.isnan(backend_audit[0]["median_fov_translation_shift_y"])
    assert np.isnan(backend_audit[0]["median_fov_translation_shift_x"])
    assert np.isnan(backend_audit[0]["median_fov_translation_peak_correlation"])
    assert "registration_backend" in format_registration_backend_audit_table(
        backend_audit
    )


def test_registration_qa_report_supports_gt_affine_oracle(tmp_path):
    subject_dir = tmp_path / "jm001"
    reference_masks = np.zeros((3, 8, 8), dtype=bool)
    reference_masks[0, 1, 1] = True
    reference_masks[1, 1, 4] = True
    reference_masks[2, 4, 1] = True
    target_masks = np.zeros_like(reference_masks)
    target_masks[0, 2, 3] = True
    target_masks[1, 2, 6] = True
    target_masks[2, 5, 3] = True
    _write_raw_npy_masks(subject_dir, "2024-05-01_a", reference_masks)
    _write_raw_npy_masks(subject_dir, "2024-05-02_a", target_masks)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (1, 1), (2, 2)),
    )

    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=subject_dir,
            reference_kind="manual-gt",
            input_format="npy",
            transform_type="gt-affine-oracle",
            max_gap=1,
            cost="registered-iou",
        )
    )

    assert len(rows) == 3
    assert {row["registration_backend"] for row in rows} == {"gt-affine-oracle"}
    assert {row["registered_plane_source"] for row in rows} == {
        "raw_npy_gt_affine_oracle"
    }
    assert {row["registration_backend_reason"] for row in rows} == {
        "manual-GT affine oracle fit from linked ROI centroids"
    }
    assert all(row["registered_iou"] == pytest.approx(1.0) for row in rows)
    assert all(
        row["registered_centroid_distance"] == pytest.approx(0.0) for row in rows
    )
    assert all(row["gt_candidate_admissible"] is True for row in rows)

    backend_audit = summarize_registration_backend_usage(rows)
    assert backend_audit[0]["registration_backend"] == "gt-affine-oracle"
    assert backend_audit[0]["transform_type"] == "gt-affine-oracle"


def test_registration_qa_report_tolerates_raw_mask_shape_mismatch(
    tmp_path,
    write_raw_npy_session,
):
    subject_dir = tmp_path / "jm001"
    reference_masks: np.ndarray = np.zeros((2, 5, 5), dtype=bool)
    reference_masks[0, 1:3, 1:3] = True
    reference_masks[1, 3:5, 3:5] = True
    target_masks: np.ndarray = np.zeros((2, 6, 5), dtype=bool)
    target_masks[:, :5, :] = reference_masks
    write_raw_npy_session(subject_dir, "2024-05-01_a", reference_masks, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", target_masks, offset=0.0)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0),),
    )

    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=subject_dir,
            reference_kind="manual-gt",
            input_format="npy",
            transform_type="fov-translation",
            max_gap=1,
            cost="registered-iou",
        )
    )

    assert len(rows) == 1
    assert rows[0]["registration_backend"] == "fov-translation"
    assert rows[0]["target_roi_present"] is True
    assert rows[0]["registration_backend"] == "fov-translation"
    assert rows[0]["registration_backend_reason"] == (
        "explicit transform_type='fov-translation'"
    )
    assert np.isfinite(rows[0]["fov_translation_peak_correlation"])
    assert rows[0]["raw_mask_shape_matches"] is False
    assert np.isnan(rows[0]["raw_iou"])
    assert rows[0]["registered_iou"] == pytest.approx(1.0)
    assert rows[0]["gt_candidate_admissible"] is True


def test_registration_qa_report_emits_calibrated_loso_probabilities(
    tmp_path,
    write_raw_npy_session,
):
    masks: np.ndarray = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 3:5, 3:5] = True
    session_names = ("2024-05-01_a", "2024-05-02_a")
    for index, subject_name in enumerate(("jm001", "jm002")):
        subject_dir = tmp_path / subject_name
        write_raw_npy_session(
            subject_dir,
            session_names[0],
            masks,
            offset=float(index),
        )
        write_raw_npy_session(
            subject_dir,
            session_names[1],
            masks.copy(),
            offset=float(index + 1),
        )
        _write_ground_truth_csv(subject_dir, session_names, ((0, 0), (1, 1)))

    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=tmp_path,
            reference_kind="manual-gt",
            input_format="npy",
            transform_type="none",
            max_gap=1,
            cost="calibrated",
            include_behavior=False,
        )
    )

    assert len(rows) == 4
    probabilities = np.array([row["gt_probability"] for row in rows], dtype=float)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert {row["calibration_sample_weight_strategy"] for row in rows} == {"none"}
    assert {row["calibration_class_weight"] for row in rows} == {"None"}

    summary = summarize_registration_qa_links(rows)
    assert len(summary) == 2
    assert all(np.isfinite(row["median_gt_probability"]) for row in summary)
