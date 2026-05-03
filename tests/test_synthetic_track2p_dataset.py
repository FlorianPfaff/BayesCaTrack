from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack import load_track2p_subject
from bayescatrack.datasets.track2p import (
    SyntheticFalsePositiveRoi,
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)
from bayescatrack.ground_truth_eval import load_track2p_ground_truth_csv


def _run_baseline_benchmark(generated, *, include_non_cells: bool = True):
    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=generated.subject_dir,
            method="track2p-baseline",
            input_format="suite2p",
            include_behavior=False,
            include_non_cells=include_non_cells,
        )
    )
    return rows[0].to_dict()


def test_synthetic_track2p_subject_writes_suite2p_and_ground_truth(tmp_path):
    generated = write_synthetic_track2p_subject(
        tmp_path,
        SyntheticTrack2pSubjectConfig(
            subject_name="jm900",
            missing_detections=((1, 2),),
            false_positive_rois=(
                SyntheticFalsePositiveRoi(session_index=1, center_yx=(15.0, 15.0)),
            ),
        ),
    )

    sessions = load_track2p_subject(
        generated.subject_dir,
        input_format="suite2p",
        include_behavior=False,
        include_non_cells=True,
    )
    ground_truth = load_track2p_ground_truth_csv(generated.ground_truth_csv)

    assert [session.session_name for session in sessions] == list(
        generated.session_names
    )
    assert [session.plane_data.n_rois for session in sessions] == [4, 5, 3]
    assert generated.stat_rows_per_session == (4, 5, 3)
    assert ground_truth.session_names == generated.session_names
    expected_tracks = np.vectorize(lambda value: -1 if value is None else int(value))(
        generated.suite2p_indices
    )
    npt.assert_array_equal(ground_truth.tracks, expected_tracks)

    result = _run_baseline_benchmark(generated)
    assert result["reference_source"] == "ground_truth_csv"
    assert result["reference_pairwise_links"] == 7
    assert float(result["pairwise_recall"]) < 1.0


def test_synthetic_track2p_subject_exercises_non_cell_stat_row_validation(tmp_path):
    generated = write_synthetic_track2p_subject(
        tmp_path,
        SyntheticTrack2pSubjectConfig(
            subject_name="jm901",
            non_cell_tracks=(1,),
        ),
    )

    config = Track2pBenchmarkConfig(
        data=generated.subject_dir,
        method="track2p-baseline",
        input_format="suite2p",
        include_behavior=False,
    )
    with pytest.raises(ValueError, match="--include-non-cells"):
        run_track2p_benchmark(config)

    result = _run_baseline_benchmark(generated)
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
