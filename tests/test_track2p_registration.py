from bayescatrack.track2p_registration import (
    build_registered_subject_association_bundles,
    register_consecutive_session_measurement_planes,
    register_plane_pair,
)
import numpy as np


def test_track2p_registration_public_functions_are_importable():
    assert callable(register_plane_pair)
    assert callable(register_consecutive_session_measurement_planes)
    assert callable(build_registered_subject_association_bundles)


def test_register_plane_pair_none_uses_masks_without_track2p_backend(make_track2p_session):
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    session = make_track2p_session("2024-05-01_a", masks)

    registered = register_plane_pair(
        session.plane_data,
        session.plane_data,
        transform_type="none",
    )

    assert registered is session.plane_data
