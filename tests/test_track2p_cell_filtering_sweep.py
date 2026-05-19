from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_cell_filtering_sweep import (
    CellFilteringSweepConfig,
    _parse_filter_modes,
    _parse_probability_values,
    run_track2p_cell_filtering_sweep,
)
from tests.test_track2p_benchmark import _write_ground_truth_csv, _write_suite2p_session


def test_cell_filtering_sweep_records_filtered_manual_gt_roi_failure(tmp_path):
    subject_dir = tmp_path / "jm003"
    iscell = np.array([[1.0, 0.95], [0.0, 0.1], [1.0, 0.9]], dtype=float)
    _write_suite2p_session(subject_dir, "2024-05-01_a", iscell=iscell)
    _write_suite2p_session(subject_dir, "2024-05-02_a", iscell=iscell)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (1, 1)),
    )

    rows = [
        result.to_dict()
        for result in run_track2p_cell_filtering_sweep(
            CellFilteringSweepConfig(
                benchmark=Track2pBenchmarkConfig(
                    data=subject_dir,
                    method="track2p-baseline",
                    input_format="suite2p",
                    progress=False,
                ),
                cell_probability_thresholds=(0.5,),
                filter_modes=("filtered", "include-non-cells"),
            )
        )
    ]

    assert len(rows) == 2
    filtered = next(row for row in rows if row["include_non_cells"] == "false")
    unfiltered = next(row for row in rows if row["include_non_cells"] == "true")
    assert filtered["status"] == "error"
    assert "--include-non-cells" in filtered["error_message"]
    assert unfiltered["status"] == "ok"
    assert unfiltered["pairwise_recall"] == pytest.approx(1.0)
    assert unfiltered["dropped_prediction_tracks"] == 1


def test_cell_filtering_sweep_parses_probability_thresholds_and_modes():
    assert _parse_probability_values("0, 0.25, 1", name="thresholds") == (
        0.0,
        0.25,
        1.0,
    )
    assert _parse_filter_modes("cells-only, all-rois") == (
        "filtered",
        "include-non-cells",
    )
    with pytest.raises(ValueError, match="\[0, 1\]"):
        _parse_probability_values("1.5", name="thresholds")
