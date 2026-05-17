import numpy as np
import numpy.testing as npt

from bayescatrack.core import _bridge_impl
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _dilate_mask_stack,
    _pairwise_dilated_iou_matrix,
)


def test_pairwise_dilated_iou_matches_reference_with_overlapping_pixels():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 3, 3] = True
    reference[1, 3, 4] = True
    measurement[0, 3, 5] = True
    measurement[1, 6, 6] = True

    radius = 2
    actual = _pairwise_dilated_iou_matrix(reference, measurement, radius=radius)
    expected = _bridge_impl._pairwise_iou_matrix(
        _dilate_mask_stack(reference, radius=radius),
        _dilate_mask_stack(measurement, radius=radius),
    )

    npt.assert_allclose(actual, expected)
