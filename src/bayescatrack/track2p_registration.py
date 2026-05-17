"""Track2p-backed registration helpers for BayesCaTrack."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from bayescatrack import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_consecutive_session_association_bundles,
    load_track2p_subject,
)

RegistrationTransform = Literal["affine", "rigid", "fov-translation", "none"]
REGISTRATION_TRANSFORM_TYPES: tuple[RegistrationTransform, ...] = (
    "affine",
    "rigid",
    "fov-translation",
    "none",
)


def _load_subject_sessions(
    subject_dir: str | Path,
    *,
    plane_name: str,
    input_format: str,
    include_behavior: bool,
    suite2p_kwargs: Mapping[str, Any],
) -> list[Track2pSession]:
    load_kwargs = {
        **suite2p_kwargs,
        "plane_name": plane_name,
        "input_format": input_format,
        "include_behavior": include_behavior,
    }
    return load_track2p_subject(subject_dir, **load_kwargs)


def _load_track2p_registration_backend() -> tuple[Any, Any]:
    try:
        from track2p.register.elastix import (  # type: ignore[import-not-found]
            itk_reg_all_roi,
            reg_img_elastix,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Track2p-compatible affine/rigid registration requires the 'track2p' "
            "package and its ITK/elastix stack. Install that backend for "
            "transform_type='affine' or transform_type='rigid', or request "
            "transform_type='fov-translation' explicitly to use BayesCaTrack's "
            "integer FOV phase-correlation fallback."
        ) from exc
    return reg_img_elastix, itk_reg_all_roi


def _coerce_registered_roi_masks(
    registered_roi_masks: Any, *, n_rois: int, image_shape: tuple[int, int]
) -> np.ndarray:
    registered_roi_masks = np.asarray(registered_roi_masks)
    if registered_roi_masks.shape == (*image_shape, n_rois):
        return np.moveaxis(registered_roi_masks > 0, -1, 0)
    if registered_roi_masks.shape == (n_rois, *image_shape):
        return registered_roi_masks > 0
    raise ValueError(
        "Registered ROI masks must have shape (height, width, n_roi) or (n_roi, height, width)."
    )


def _fov_translation_registered_plane(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str = "fov-translation",
    reason: str = "explicit transform_type='fov-translation'",
) -> CalciumPlaneData:
    from bayescatrack.fov_registration import (
        register_measurement_plane_by_fov_translation,
    )

    registered_plane = register_measurement_plane_by_fov_translation(
        reference_plane,
        moving_plane,
    ).registered_measurement_plane
    return _with_registration_backend_metadata(
        registered_plane,
        backend="fov-translation",
        transform_type=transform_type,
        reason=reason,
    )


def register_plane_pair(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: RegistrationTransform | str = "affine",
) -> CalciumPlaneData:
    if transform_type not in REGISTRATION_TRANSFORM_TYPES:
        valid_types = ", ".join(repr(value) for value in REGISTRATION_TRANSFORM_TYPES)
        raise ValueError(f"transform_type must be one of {valid_types}")
    if transform_type == "none":
        if reference_plane.image_shape != moving_plane.image_shape:
            raise ValueError("transform_type='none' requires matching image shapes")
        return moving_plane
    if reference_plane.fov is None or moving_plane.fov is None:
        raise ValueError("Both planes must provide FOV images for registration.")
    if transform_type == "fov-translation":
        return _fov_translation_registered_plane(reference_plane, moving_plane)

    reg_img_elastix, itk_reg_all_roi = _load_track2p_registration_backend()
    registered_fov, reg_params = reg_img_elastix(
        np.asarray(reference_plane.fov),
        np.asarray(moving_plane.fov),
        SimpleNamespace(transform_type=transform_type),
    )
    moving_support_masks_hw_n = np.moveaxis(
        np.asarray(moving_plane.roi_masks) > 0, 0, -1
    )
    registered_support_masks = _coerce_registered_roi_masks(
        itk_reg_all_roi(moving_support_masks_hw_n, reg_params),
        n_rois=moving_plane.n_rois,
        image_shape=reference_plane.image_shape,
    )
    return moving_plane.with_replaced_masks(
        registered_support_masks,
        fov=np.asarray(registered_fov),
        source=f"{moving_plane.source}_registered",
        ops=_registration_backend_ops(
            moving_plane.ops,
            backend="track2p-elastix",
            transform_type=transform_type,
            reason="track2p.register.elastix import succeeded",
        ),
    )


def _registration_backend_ops(
    source_ops: Mapping[str, Any] | None,
    *,
    backend: str,
    transform_type: str,
    reason: str,
) -> dict[str, Any]:
    ops = {} if source_ops is None else dict(source_ops)
    ops.update(
        {
            "registration_backend": backend,
            "registration_transform_type": transform_type,
            "registration_backend_reason": reason,
        }
    )
    return ops


def _with_registration_backend_metadata(
    plane: CalciumPlaneData,
    *,
    backend: str,
    transform_type: str,
    reason: str,
) -> CalciumPlaneData:
    return plane.with_replaced_masks(
        plane.roi_masks,
        fov=plane.fov,
        source=plane.source,
        ops=_registration_backend_ops(
            plane.ops,
            backend=backend,
            transform_type=transform_type,
            reason=reason,
        ),
    )


def register_consecutive_session_measurement_planes(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: RegistrationTransform | str = "affine",
) -> list[CalciumPlaneData]:
    sessions = list(sessions)
    if len(sessions) < 2:
        return []
    return [
        register_plane_pair(
            sessions[i].plane_data,
            sessions[i + 1].plane_data,
            transform_type=transform_type,
        )
        for i in range(len(sessions) - 1)
    ]


def build_registered_subject_association_bundles(  # pylint: disable=too-many-arguments
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    transform_type: RegistrationTransform | str = "affine",
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
    **suite2p_kwargs: Any,
) -> list[SessionAssociationBundle]:
    sessions = _load_subject_sessions(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        suite2p_kwargs=suite2p_kwargs,
    )
    registered_measurement_planes = register_consecutive_session_measurement_planes(
        sessions, transform_type=transform_type
    )

    association_kwargs: dict[str, Any] = {"order": order}
    association_kwargs["weighted_centroids"] = weighted_centroids
    association_kwargs["velocity_variance"] = velocity_variance
    association_kwargs["regularization"] = regularization
    association_kwargs["pairwise_cost_kwargs"] = pairwise_cost_kwargs
    association_kwargs["return_pairwise_components"] = return_pairwise_components

    return build_consecutive_session_association_bundles(
        sessions,
        measurement_planes_in_reference_frames=registered_measurement_planes,
        **association_kwargs,
    )
