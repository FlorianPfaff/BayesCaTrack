from bayescatrack import fov_registration


def test_fov_registration_module_exposes_expected_public_api():
    for name in (
        "FovRegisteredSessionPairBundle",
        "FovTranslationRegistration",
        "apply_integer_image_translation",
        "apply_integer_roi_mask_translation",
        "build_fov_registered_consecutive_session_association_bundles",
        "build_fov_registered_session_pair_association_bundle",
        "estimate_integer_fov_shift",
        "register_measurement_plane_by_fov_translation",
    ):
        assert hasattr(fov_registration, name)
