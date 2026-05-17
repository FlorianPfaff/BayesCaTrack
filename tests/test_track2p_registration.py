import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.fov_registration import apply_integer_image_translation
from bayescatrack.track2p_registration import (
    build_registered_subject_association_bundles,
    register_consecutive_session_measurement_planes,
    register_plane_pair,
)


def test_track2p_registration_public_functions_are_importable():
    assert callable(register_plane_pair)
    assert callable(register_consecutive_session_measurement_planes)
    assert callable(build_registered_subject_association_bundles)


def test_register_plane_pair_none_uses_masks_without_track2p_backend(
    make_track2p_session,
):
    masks: np.ndarray = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    session = make_track2p_session("2024-05-01_a", masks)

    registered = register_plane_pair(
        session.plane_data,
        session.plane_data,
        transform_type="none",
    )

    assert registered is session.plane_data


def test_register_plane_pair_uses_explicit_fov_translation(
    make_track2p_session,
):
    reference_masks: np.ndarray = np.zeros((1, 6, 6), dtype=bool)
    reference_masks[0, 1:4, 2:5] = True
    moving_masks = np.zeros_like(reference_masks)
    moving_masks[0] = apply_integer_image_translation(
        reference_masks[0], np.array([1, -1])
    )
    reference_fov = reference_masks.sum(axis=0, dtype=float)
    moving_fov = moving_masks.sum(axis=0, dtype=float)
    reference = make_track2p_session("2024-05-01_a", reference_masks, fov=reference_fov)
    moving = make_track2p_session("2024-05-02_a", moving_masks, fov=moving_fov)

    registered = register_plane_pair(
        reference.plane_data,
        moving.plane_data,
        transform_type="fov-translation",
    )

    npt.assert_array_equal(registered.roi_masks, reference_masks)
    assert registered.source == "raw_npy_fov_registered"
    assert registered.ops is not None
    assert registered.ops["registration_backend"] == "fov-translation"
    assert registered.ops["registration_transform_type"] == "fov-translation"
    assert registered.ops["registration_backend_reason"] == (
        "explicit transform_type='fov-translation'"
    )
    npt.assert_array_equal(
        registered.ops["fov_registration_measurement_to_reference_shift_yx"],
        np.array([-1, 1]),
    )


def test_register_plane_pair_affine_rigid_do_not_fall_back_to_fov_translation(
    make_track2p_session,
    monkeypatch,
):
    import bayescatrack.track2p_registration as registration

    masks: np.ndarray = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 1:4, 2:5] = True
    fov = masks.sum(axis=0, dtype=float)
    reference = make_track2p_session("2024-05-01_a", masks, fov=fov)
    moving = make_track2p_session("2024-05-02_a", masks.copy(), fov=fov.copy())

    def _missing_backend():
        raise ImportError("missing test backend")

    monkeypatch.setattr(
        registration, "_load_track2p_registration_backend", _missing_backend
    )

    with pytest.raises(ImportError, match="missing test backend"):
        register_plane_pair(
            reference.plane_data,
            moving.plane_data,
            transform_type="rigid",
        )
