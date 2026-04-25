import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData
from bayescatrack.registration import warp_image_into_reference_frame, warp_roi_masks_into_reference_frame


def test_registration_warp_helpers_preserve_identity_transform():
    image = np.zeros((3, 3), dtype=float)
    image[1, 1] = 1.0
    identity = np.eye(2)
    offset = np.zeros(2)

    warped = warp_image_into_reference_frame(image, identity, offset, output_shape=(3, 3))
    npt.assert_allclose(warped, image)

    masks = np.zeros((1, 3, 3), dtype=bool)
    masks[0, 1, 1] = True
    warped_masks = warp_roi_masks_into_reference_frame(masks, identity, offset, output_shape=(3, 3))
    npt.assert_allclose(warped_masks, masks.astype(float))


def test_calcium_plane_copy_preserves_metadata_after_mask_replacement():
    plane = CalciumPlaneData(roi_masks=np.ones((1, 2, 2), dtype=bool), roi_indices=np.array([5]))
    replacement = np.zeros((1, 2, 2), dtype=bool)
    replacement[0, 0, 0] = True

    copied = plane.with_replaced_masks(replacement, source="registered")

    npt.assert_array_equal(copied.roi_indices, np.array([5]))
    assert copied.source == "registered"
