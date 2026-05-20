from __future__ import annotations

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.registration_selection import select_registration_transform
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES


def _plane(fov: np.ndarray, *, source: str = "synthetic") -> CalciumPlaneData:
    masks = np.zeros((2, *fov.shape), dtype=bool)
    masks[0, 2:5, 2:5] = True
    masks[1, -6:-3, -6:-3] = True
    return CalciumPlaneData(
        roi_masks=masks,
        fov=np.asarray(fov, dtype=float),
        source=source,
    )


def test_auto_transform_is_registered_choice() -> None:
    assert "auto" in REGISTRATION_TRANSFORM_TYPES


def test_auto_registration_selects_fov_translation_when_it_improves_fov(
    monkeypatch,
) -> None:
    reference_fov = np.zeros((24, 24), dtype=float)
    reference_fov[3:8, 4:10] = 1.0
    reference_fov[15:19, 15:21] = 0.75
    moving_fov = np.flipud(reference_fov)
    reference = _plane(reference_fov, source="reference")
    moving = _plane(moving_fov, source="moving")

    def fake_candidate_registered_plane(
        reference_plane: CalciumPlaneData,
        moving_plane: CalciumPlaneData,
        *,
        transform_type: str,
    ) -> CalciumPlaneData:
        if transform_type == "none":
            return moving_plane.with_replaced_masks(
                moving_plane.roi_masks,
                fov=moving_plane.fov,
                source="none",
                ops={
                    "registration_transform_type": "none",
                    "registration_backend": "none",
                },
            )
        assert transform_type == "fov-translation"
        return moving_plane.with_replaced_masks(
            moving_plane.roi_masks,
            fov=reference_plane.fov,
            source="fov_translation",
            ops={
                "registration_transform_type": "fov-translation",
                "registration_backend": "fov-translation",
            },
        )

    monkeypatch.setattr(
        "bayescatrack.registration_selection._candidate_registered_plane",
        fake_candidate_registered_plane,
    )

    result = select_registration_transform(
        reference,
        moving,
        candidate_transforms=("none", "fov-translation"),
        min_fov_correlation_gain=0.01,
    )

    assert result.selected_transform_type == "fov-translation"
    assert result.registered_plane.ops is not None
    assert (
        result.registered_plane.ops["registration_auto_selected_transform"]
        == "fov-translation"
    )
    assert result.registered_plane.ops["registration_auto_fov_correlation"] > 0.99


def test_auto_registration_prefers_none_without_enough_gain(monkeypatch) -> None:
    reference_fov = np.zeros((16, 16), dtype=float)
    reference_fov[4:9, 4:9] = 1.0
    reference = _plane(reference_fov, source="reference")
    moving = _plane(reference_fov.copy(), source="moving")

    def fake_candidate_registered_plane(
        reference_plane: CalciumPlaneData,
        moving_plane: CalciumPlaneData,
        *,
        transform_type: str,
    ) -> CalciumPlaneData:
        return moving_plane.with_replaced_masks(
            moving_plane.roi_masks,
            fov=reference_plane.fov,
            source=transform_type,
            ops={
                "registration_transform_type": transform_type,
                "registration_backend": transform_type,
            },
        )

    monkeypatch.setattr(
        "bayescatrack.registration_selection._candidate_registered_plane",
        fake_candidate_registered_plane,
    )

    result = select_registration_transform(
        reference,
        moving,
        candidate_transforms=("none", "fov-translation"),
        min_fov_correlation_gain=0.02,
    )

    assert result.selected_transform_type == "none"
    assert result.registered_plane.ops is not None
    assert result.registered_plane.ops["registration_auto_selected_transform"] == "none"
