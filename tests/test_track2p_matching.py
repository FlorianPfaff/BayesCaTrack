import csv
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_PATH))

from track2p_pyrecest_bridge import (  # noqa: E402
    CalciumPlaneData,
    Track2pSession,
    build_consecutive_session_association_bundles,
)
from track2p_pyrecest_bridge.matching import (  # noqa: E402
    build_track_rows_from_bundles,
    export_track_rows_csv,
    solve_bundle_linear_assignment,
)


@pytest.mark.skipif(
    pytest.importorskip("scipy.optimize") is None,
    reason="scipy.optimize is required for linear assignment",
)
def test_bundle_assignment_and_track_row_export(tmp_path):
    image_shape = (4, 6)

    session0_masks = np.zeros((2, *image_shape), dtype=bool)
    session0_masks[0, 0:2, 0:2] = True
    session0_masks[1, 1:3, 3:5] = True

    session1_masks = np.zeros((2, *image_shape), dtype=bool)
    session1_masks[0, 1:3, 3:5] = True
    session1_masks[1, 0:2, 0:2] = True

    session2_masks = np.zeros((1, *image_shape), dtype=bool)
    session2_masks[0, 1:3, 3:5] = True

    session0 = Track2pSession(
        session_dir=tmp_path / "s0",
        session_name="s0",
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=session0_masks,
            roi_indices=np.array([10, 20], dtype=int),
            traces=np.array([[1.0, 2.0], [3.0, 4.0]]),
        ),
    )
    session1 = Track2pSession(
        session_dir=tmp_path / "s1",
        session_name="s1",
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=session1_masks,
            roi_indices=np.array([101, 202], dtype=int),
            traces=np.array([[5.0, 6.0], [7.0, 8.0]]),
        ),
    )
    session2 = Track2pSession(
        session_dir=tmp_path / "s2",
        session_name="s2",
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=session2_masks,
            roi_indices=np.array([303], dtype=int),
            traces=np.array([[9.0, 10.0]]),
        ),
    )

    bundles = build_consecutive_session_association_bundles(
        [session0, session1, session2],
        pairwise_cost_kwargs={"max_centroid_distance": 10.0, "roi_feature_weight": 0.0},
    )
    assert len(bundles) == 2

    first_match = solve_bundle_linear_assignment(bundles[0], max_cost=10.0)
    assert first_match.reference_session_name == "s0"
    assert first_match.measurement_session_name == "s1"
    npt.assert_array_equal(first_match.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(first_match.measurement_roi_indices, np.array([202, 101]))
    assert np.all(first_match.costs >= 0.0)

    session_names, track_rows, match_results = build_track_rows_from_bundles(
        bundles,
        max_cost=10.0,
        start_roi_indices=np.array([10, 20], dtype=int),
    )
    assert session_names == ("s0", "s1", "s2")
    assert len(match_results) == 2
    npt.assert_array_equal(
        track_rows,
        np.array(
            [
                [10, 202, -1],
                [20, 101, 303],
            ],
            dtype=int,
        ),
    )

    output_path = export_track_rows_csv(
        tmp_path / "predicted_tracks.csv",
        session_names,
        track_rows,
    )
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.reader(handle))

    assert reader[0] == ["track_id", "s0", "s1", "s2"]
    assert reader[1] == ["0", "10", "202", "-1"]
    assert reader[2] == ["1", "20", "101", "303"]
