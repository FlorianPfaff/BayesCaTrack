import numpy as np
import numpy.testing as npt
from bayescatrack.matching import build_track_rows_from_matches


def test_build_track_rows_from_consecutive_matches():
    rows = build_track_rows_from_matches(
        ("s1", "s2", "s3"),
        [np.array([[0, 1], [2, 3]]), np.array([[1, 5], [3, 6]])],
        start_roi_indices=np.array([0, 2]),
    )

    npt.assert_array_equal(rows, np.array([[0, 1, 5], [2, 3, 6]]))
