"""Regression tests for explicit registration-backend policy."""

from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack import track2p_registration
from bayescatrack.track2p_registration import register_plane_pair


def _plane(source: str) -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    fov = np.asarray(masks[0], dtype=float)
    return CalciumPlaneData(roi_masks=masks, fov=fov, source=source)


def test_affine_registration_requires_explicit_fallback_when_track2p_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_backend() -> tuple[object, object]:
        raise ImportError("missing elastix; pass allow_fov_affine_fallback=True")

    monkeypatch.setattr(
        track2p_registration, "_load_track2p_registration_backend", missing_backend
    )

    with pytest.raises(ImportError, match="allow_fov_affine_fallback=True"):
        register_plane_pair(_plane("reference"), _plane("moving"), transform_type="affine")


def test_affine_registration_can_use_explicit_fov_affine_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_backend() -> tuple[object, object]:
        raise ImportError("missing elastix")

    def fake_fov_affine(
        reference_plane: CalciumPlaneData,
        moving_plane: CalciumPlaneData,
        *,
        transform_type: str,
        reason: str,
    ) -> CalciumPlaneData:
        del reference_plane
        return moving_plane.with_replaced_masks(
            moving_plane.roi_masks,
            fov=moving_plane.fov,
            source=moving_plane.source,
            ops={
                "registration_backend": "fov-affine",
                "registration_transform_type": transform_type,
                "registration_backend_reason": reason,
            },
        )

    monkeypatch.setattr(
        track2p_registration, "_load_track2p_registration_backend", missing_backend
    )
    monkeypatch.setattr(
        track2p_registration, "_fov_affine_registered_plane", fake_fov_affine
    )

    registered = register_plane_pair(
        _plane("reference"),
        _plane("moving"),
        transform_type="affine",
        allow_fov_affine_fallback=True,
    )

    assert registered.ops is not None
    assert registered.ops["registration_backend"] == "fov-affine"
    assert registered.ops["registration_transform_type"] == "affine"
    assert "allow_fov_affine_fallback=True" in registered.ops["registration_backend_reason"]


def test_none_registration_records_backend_metadata() -> None:
    registered = register_plane_pair(_plane("reference"), _plane("moving"), transform_type="none")

    assert registered.ops is not None
    assert registered.ops["registration_backend"] == "none"
    assert registered.ops["registration_transform_type"] == "none"
