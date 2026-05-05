from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayescatrack.experiments.track2p_input_validator import (
    Track2pInputValidationConfig,
    format_validation_markdown,
    run_track2p_input_validation,
)


def _write_ground_truth_csv(
    subject_dir: Path, session_names: tuple[str, ...], rows: tuple[tuple[int, ...], ...]
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def _write_suite2p_session(subject_dir: Path, session_name: str, n_rois: int) -> None:
    plane_dir = subject_dir / session_name / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = []
    for index in range(n_rois):
        stat.append(
            {
                "ypix": np.array([index % 4, index % 4]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            }
        )
    np.save(plane_dir / "stat.npy", np.asarray(stat, dtype=object), allow_pickle=True)
    np.save(plane_dir / "iscell.npy", np.ones((n_rois, 2), dtype=float))
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 8, "Lx": 8, "meanImg": np.zeros((8, 8), dtype=float)},
        allow_pickle=True,
    )
    np.save(plane_dir / "F.npy", np.zeros((n_rois, 2), dtype=float))


def test_track2p_input_validator_reports_compatible_manual_gt(tmp_path):
    subject_dir = tmp_path / "jm001"
    sessions = ("2024-05-01_a", "2024-05-02_a")
    for session_name in sessions:
        _write_suite2p_session(subject_dir, session_name, n_rois=3)
    _write_ground_truth_csv(subject_dir, sessions, ((0, 1), (2, 2)))

    result = run_track2p_input_validation(
        Track2pInputValidationConfig(
            data=tmp_path,
            input_format="suite2p",
            include_behavior=False,
            include_non_cells=True,
        )
    )

    assert result.compatible
    assert result.incompatible_subjects == ()
    assert {row.missing_rois for row in result.rows} == {0}
    assert "compatible: `true`" in format_validation_markdown(result)


def test_track2p_input_validator_reports_missing_manual_gt_rois(tmp_path):
    subject_dir = tmp_path / "jm002"
    sessions = ("2024-05-01_a", "2024-05-02_a")
    for session_name in sessions:
        _write_suite2p_session(subject_dir, session_name, n_rois=3)
    _write_ground_truth_csv(subject_dir, sessions, ((0, 10), (2, 2)))

    result = run_track2p_input_validation(
        Track2pInputValidationConfig(
            data=tmp_path,
            input_format="suite2p",
            include_behavior=False,
            include_non_cells=True,
        )
    )

    assert not result.compatible
    assert result.incompatible_subjects == ("jm002",)
    missing_rows = [row for row in result.rows if row.missing_rois]
    assert len(missing_rows) == 1
    assert missing_rows[0].session == "2024-05-02_a"
    assert missing_rows[0].referenced_max == 10
    assert missing_rows[0].loaded_max == 2
    assert missing_rows[0].index_space_hint == "loaded_roi_subset_or_reindexed_public_data"


def test_track2p_input_validator_rejects_non_manual_references(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm003"
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    write_raw_npy_session(subject_dir, "2024-05-01_a", masks, offset=0.0)

    with pytest.raises(ValueError, match="not independent manual ground truth"):
        run_track2p_input_validation(
            Track2pInputValidationConfig(
                data=tmp_path,
                reference_kind="auto",
                input_format="npy",
            )
        )
