from bayescatrack.track2p_registration import (
    build_registered_subject_association_bundles,
    register_consecutive_session_measurement_planes,
    register_plane_pair,
)


def test_track2p_registration_public_functions_are_importable():
    assert callable(register_plane_pair)
    assert callable(register_consecutive_session_measurement_planes)
    assert callable(build_registered_subject_association_bundles)
