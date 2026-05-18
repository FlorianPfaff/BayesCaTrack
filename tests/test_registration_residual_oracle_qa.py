from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayescatrack.experiments.registration_residual_oracle_qa import (
    RegistrationResidualOracleQAConfig,
    run_registration_residual_oracle_qa_report,
    summarize_registration_residual_oracle_qa_links,
)
from bayescatrack.experiments.registration_qa_report import RegistrationQAConfig


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
    subject_dir: Path,
    session_name: str,
    masks: np.ndarray,
) -> None:
    plane_dir = subject_dir / session_name / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    np.save(plane_dir / "rois.npy", masks)
    np.save(plane_dir / "F.npy", np.zeros((masks.shape[0], 2), dtype=float))
    np.save(plane_dir / "fov.npy", np.asarray(masks, dtype=float).sum(axis=0))


def test_registration_residual_oracle_qa_reports_residuals_and_rank_upper_bound(
    tmp_path,
):
    subject_dir = tmp_path / "jm001"
    reference_masks = np.zeros((3, 8, 8), dtype=bool)
    reference_masks[0, 1, 1] = True
    reference_masks[1, 1, 4] = True
    reference_masks[2, 4, 1] = True

    target_masks = np.zeros((4, 8, 8), dtype=bool)
    # Three manual-GT ROIs are translated by +2 px in x and +1 px in y.
    target_masks[0, 2, 3] = True
    target_masks[1, 2, 6] = True
    target_masks[2, 5, 3] = True
    # Extra non-GT target ROI overlaps source ROI 0 before registration; this
    # makes the baseline row rank worse than the manual-GT oracle row rank.
    target_masks[3, 1, 1] = True

    session_names = ("2024-05-01_a", "2024-05-02_a")
    _write_raw_npy_masks(subject_dir, session_names[0], reference_masks)
    _write_raw_npy_masks(subject_dir, session_names[1], target_masks)
    _write_ground_truth_csv(subject_dir, session_names, ((0, 0), (1, 1), (2, 2)))

    rows = run_registration_residual_oracle_qa_report(
        RegistrationResidualOracleQAConfig(
            registration=RegistrationQAConfig(
                data=subject_dir,
                reference_kind="manual-gt",
                input_format="npy",
                transform_type="none",
                max_gap=1,
                cost="registered-iou",
                progress=False,
            )
        )
    )

    assert len(rows) == 3
    assert rows[0]["raw_residual_x"] == pytest.approx(2.0)
    assert rows[0]["raw_residual_y"] == pytest.approx(1.0)
    assert rows[0]["baseline_residual_x"] == pytest.approx(2.0)
    assert rows[0]["baseline_residual_y"] == pytest.approx(1.0)
    assert rows[0]["oracle_residual_norm"] == pytest.approx(0.0)
    assert rows[0]["baseline_iou"] == pytest.approx(0.0)
    assert rows[0]["oracle_iou"] == pytest.approx(1.0)
    assert rows[0]["baseline_iou_row_rank"] > rows[0]["oracle_iou_row_rank"]
    assert rows[0]["oracle_iou_row_rank_is_1"] is True
    assert rows[0]["oracle_iou_column_rank_is_1"] is True
    assert rows[0]["oracle_iou_mutual_rank_is_1"] is True
    assert rows[0]["baseline_iou_row_margin"] <= 0.0
    assert rows[0]["oracle_iou_row_margin"] > 0.0
    assert rows[0]["oracle_fit_rms_residual"] == pytest.approx(0.0)

    summary = summarize_registration_residual_oracle_qa_links(rows)
    assert len(summary) == 1
    assert summary[0]["median_baseline_iou"] == pytest.approx(0.0)
    assert summary[0]["median_oracle_iou"] == pytest.approx(1.0)
    assert summary[0]["baseline_iou_row_hit1_rate"] < summary[0]["oracle_iou_row_hit1_rate"]
    assert summary[0]["oracle_iou_row_hit1_rate"] == pytest.approx(1.0)
    assert summary[0]["oracle_iou_mutual_hit1_rate"] == pytest.approx(1.0)
    assert summary[0]["median_oracle_residual_norm"] == pytest.approx(0.0)
