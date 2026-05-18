from __future__ import annotations

from pathlib import Path

import numpy as np
from bayescatrack.core.bridge import load_suite2p_plane
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)
from bayescatrack.experiments.track2p_raw_benchmark_data import (
    prepare_raw_suite2p_benchmark_data,
)
from tests.test_track2p_benchmark import _write_ground_truth_csv, _write_suite2p_session


def _write_track2p_suite2p_indices(
    subject_dir: Path,
    session_names: tuple[str, ...],
    suite2p_indices: np.ndarray,
) -> None:
    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.asarray(
                [str(subject_dir / session_name) for session_name in session_names],
                dtype=object,
            ),
            "vector_curation_plane_0": np.ones(suite2p_indices.shape[0], dtype=float),
        },
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        suite2p_indices,
        allow_pickle=True,
    )


def _write_raw_suite2p_subject(root: Path, subject_name: str) -> Path:
    subject_dir = root / "raw_export" / subject_name
    iscell = np.array([[1, 0.9], [1, 0.8], [0, 0.1]], dtype=float)
    for session_name in ("2024-05-01_a", "2024-05-02_a"):
        plane_dir = _write_suite2p_session(subject_dir, session_name, iscell=iscell)
        (plane_dir / "F.npy").unlink()
    return subject_dir


def _write_metadata_subject(
    root: Path, subject_name: str, *, include_track2p: bool = True
) -> Path:
    session_names = ("2024-05-01_a", "2024-05-02_a")
    subject_dir = root / "metadata_export" / subject_name
    subject_dir.mkdir(parents=True)
    _write_ground_truth_csv(subject_dir, session_names, ((0, 0), (2, 2)))
    if include_track2p:
        _write_track2p_suite2p_indices(
            subject_dir,
            session_names,
            np.array([[0, 0], [2, 2]], dtype=object),
        )
    return subject_dir


def test_prepare_raw_suite2p_benchmark_data_combines_raw_sessions_with_metadata(
    tmp_path,
):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    _write_raw_suite2p_subject(raw_root, "jm038")
    _write_metadata_subject(metadata_root, "jm038")

    preparation = prepare_raw_suite2p_benchmark_data(
        raw_root=raw_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "prepared",
        diagnostics_dir=tmp_path / "results",
    )

    assert preparation.included == ("jm038",)
    assert (preparation.output_root / "jm038" / "ground_truth.csv").is_file()
    assert (
        preparation.output_root / "jm038" / "track2p" / "plane0_suite2p_indices.npy"
    ).is_file()
    assert all(diagnostic.compatible for diagnostic in preparation.diagnostics)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=preparation.output_root,
            reference=preparation.output_root,
            reference_kind="manual-gt",
            method="track2p-baseline",
            include_non_cells=True,
        )
    )
    result = rows[0].to_dict()
    assert result["subject"] == "jm038"
    assert result["reference_source"] == "ground_truth_csv"
    assert result["complete_track_f1"] == 1.0

    plane = load_suite2p_plane(
        preparation.output_root / "jm038" / "2024-05-01_a" / "suite2p" / "plane0",
        include_non_cells=True,
    )
    assert plane.fov is not None
    assert plane.ops is not None and plane.ops.get("meanImg") is not None


def test_prepare_raw_suite2p_benchmark_data_rejects_missing_ops_mean_image(
    tmp_path,
):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    raw_subject = _write_raw_suite2p_subject(raw_root, "jm038")
    (raw_subject / "2024-05-01_a" / "suite2p" / "plane0" / "ops.npy").unlink()
    _write_metadata_subject(metadata_root, "jm038")

    try:
        prepare_raw_suite2p_benchmark_data(
            raw_root=raw_root,
            metadata_root=metadata_root,
            output_root=tmp_path / "prepared",
            diagnostics_dir=tmp_path / "results",
        )
    except ValueError as exc:
        assert "Need at least 1 raw Suite2p manual-GT subject" in str(exc)
    else:  # pragma: no cover
        raise AssertionError(
            "Expected missing ops.npy to reject the raw benchmark tree"
        )

    summary = (tmp_path / "results" / "raw_suite2p_benchmark_subjects.json").read_text(
        encoding="utf-8"
    )
    assert "missing suite2p/plane0/ops.npy" in summary
    assert "ROI-mask occupancy fallback" in summary


def test_prepare_raw_suite2p_benchmark_data_accepts_raw_track2p_bridge(tmp_path):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    raw_subject = _write_raw_suite2p_subject(raw_root, "jm039")
    _write_track2p_suite2p_indices(
        raw_subject,
        ("2024-05-01_a", "2024-05-02_a"),
        np.array([[0, 0], [5, 2]], dtype=object),
    )
    _write_metadata_subject(metadata_root, "jm039", include_track2p=False)

    preparation = prepare_raw_suite2p_benchmark_data(
        raw_root=raw_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "prepared",
        diagnostics_dir=tmp_path / "results",
    )

    assert preparation.included == ("jm039",)
    assert (
        preparation.output_root / "jm039" / "track2p" / "plane0_suite2p_indices.npy"
    ).is_file()
    assert any(
        diagnostic.source == "track2p_suite2p_indices"
        and diagnostic.missing_rois == 1
        and not diagnostic.compatible
        for diagnostic in preparation.diagnostics
    )


def test_prepare_raw_suite2p_benchmark_data_uses_track2p_session_subset(tmp_path):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    raw_subject = raw_root / "raw_export" / "jm046"
    iscell = np.array([[1, 0.9], [1, 0.8], [0, 0.1]], dtype=float)
    for session_name in (
        "2024-09-02_a",
        "2024-09-03_a",
        "2024-09-04_a",
    ):
        _write_suite2p_session(raw_subject, session_name, iscell=iscell)
    _write_track2p_suite2p_indices(
        raw_subject,
        ("2024-09-03_a", "2024-09-04_a"),
        np.array([[0, 0], [2, 2]], dtype=object),
    )

    metadata_subject = metadata_root / "metadata_export" / "jm046"
    metadata_subject.mkdir(parents=True)
    _write_ground_truth_csv(
        metadata_subject,
        ("2024-09-03_a", "2024-09-04_a"),
        ((0, 0), (2, 2)),
    )

    preparation = prepare_raw_suite2p_benchmark_data(
        raw_root=raw_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "prepared",
        diagnostics_dir=tmp_path / "results",
    )

    prepared_subject = preparation.output_root / "jm046"
    assert preparation.included == ("jm046",)
    assert not (prepared_subject / "2024-09-02_a").exists()
    assert (prepared_subject / "2024-09-03_a").is_dir()
    assert (prepared_subject / "2024-09-04_a").is_dir()
    assert {
        diagnostic.session
        for diagnostic in preparation.diagnostics
        if diagnostic.source == "manual_gt"
    } == {"2024-09-03_a", "2024-09-04_a"}


def test_prepare_raw_suite2p_benchmark_data_rejects_missing_raw_indices(tmp_path):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    _write_raw_suite2p_subject(raw_root, "jm046")
    metadata_subject = _write_metadata_subject(metadata_root, "jm046")
    _write_ground_truth_csv(
        metadata_subject,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (5, 5)),
    )
    _write_track2p_suite2p_indices(
        metadata_subject,
        ("2024-05-01_a", "2024-05-02_a"),
        np.array([[0, 0], [5, 5]], dtype=object),
    )

    try:
        prepare_raw_suite2p_benchmark_data(
            raw_root=raw_root,
            metadata_root=metadata_root,
            output_root=tmp_path / "prepared",
            diagnostics_dir=tmp_path / "results",
        )
    except ValueError as exc:
        assert "Need at least 1 raw Suite2p manual-GT subject" in str(exc)
    else:  # pragma: no cover
        raise AssertionError(
            "Expected raw Suite2p preparation to reject missing indices"
        )

    diagnostics = (tmp_path / "results" / "raw_suite2p_roi_diagnostics.csv").read_text(
        encoding="utf-8"
    )
    assert "jm046" in diagnostics
    assert "false" in diagnostics


def test_prepare_raw_suite2p_benchmark_data_can_filter_missing_manual_gt_rows(
    tmp_path,
):
    raw_root = tmp_path / "raw"
    metadata_root = tmp_path / "metadata"
    _write_raw_suite2p_subject(raw_root, "jm046")
    metadata_subject = _write_metadata_subject(
        metadata_root,
        "jm046",
        include_track2p=False,
    )
    _write_ground_truth_csv(
        metadata_subject,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (5, 5)),
    )

    preparation = prepare_raw_suite2p_benchmark_data(
        raw_root=raw_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "prepared",
        diagnostics_dir=tmp_path / "results",
        require_track2p_suite2p_indices=False,
        filter_missing_manual_rois=True,
    )

    assert preparation.included == ("jm046",)
    assert preparation.excluded_no_track2p_suite2p_indices == ()
    assert preparation.excluded_incompatible == ()
    assert preparation.filtered_manual_gt_tracks == ("jm046:1",)
    prepared_ground_truth = (
        preparation.output_root / "jm046" / "ground_truth.csv"
    ).read_text(encoding="utf-8")
    assert "5,5" not in prepared_ground_truth
    assert prepared_ground_truth.count("\n") == 2
    assert "Filtered manual-GT tracks with absent raw Suite2p ROIs: jm046:1" in (
        tmp_path / "results" / "raw_suite2p_roi_diagnostics.md"
    ).read_text(encoding="utf-8")

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=preparation.output_root,
            reference=preparation.output_root,
            reference_kind="manual-gt",
            method="global-assignment",
            input_format="suite2p",
            include_non_cells=True,
            include_behavior=False,
            transform_type="none",
            max_gap=1,
        )
    )
    assert rows[0].to_dict()["subject"] == "jm046"
