from __future__ import annotations

import numpy as np
from bayescatrack.experiments.oracle_affine_registration_qa import (
    _fit_affine_xy,
    _warp_masks_by_affine_xy,
)


def test_fit_affine_xy_recovers_known_mapping():
    target_xy = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
        ]
    )
    expected = np.asarray([[2.0, 0.5, 3.0], [-0.25, 1.5, 4.0]])
    source_xy = np.column_stack((target_xy, np.ones(target_xy.shape[0]))) @ expected.T

    fit = _fit_affine_xy(source_xy, target_xy)

    np.testing.assert_allclose(fit.matrix_xy, expected, atol=1.0e-12)
    assert fit.rank == 3
    assert fit.rms_residual < 1.0e-12


def test_warp_masks_by_affine_xy_applies_translation():
    masks = np.zeros((1, 5, 6), dtype=bool)
    masks[0, 2, 3] = True
    affine = np.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])

    warped = _warp_masks_by_affine_xy(masks, affine, (5, 6))

    expected = np.zeros_like(masks)
    expected[0, 3, 2] = True
    np.testing.assert_array_equal(warped, expected)
