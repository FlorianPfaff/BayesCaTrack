from pathlib import Path

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData, Track2pSession
from bayescatrack.multisession_tracking import track_sessions_multisession


def test_single_session_multisession_tracking_short_circuits():
    plane = CalciumPlaneData(roi_masks=np.ones((1, 2, 2), dtype=bool), roi_indices=np.array([7]))
    session = Track2pSession(
        session_dir=Path("s1"),
        session_name="s1",
        session_date=None,
        plane_data=plane,
    )

    result = track_sessions_multisession([session])

    npt.assert_array_equal(result.track_matrix, np.array([[0]]))
    npt.assert_array_equal(result.track_roi_index_matrix, np.array([[7]]))
    assert result.summary()["n_tracks"] == 1
