"""Track2p-backed registration helpers for BayesCaTrack."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from bayescatrack import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_consecutive_session_association_bundles,
    load_track2p_subject,
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
        from track2p.register.elastix import itk_reg_all_roi, reg_img_elastix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Track2p-compatible registration requires the 'track2p' package and its ITK/elastix stack."
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


def register_plane_pair(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str = "affine",
) -> CalciumPlaneData:
    if transform_type not in {"affine", "rigid"}:
        raise ValueError("transform_type must be either 'affine' or 'rigid'")
    if reference_plane.fov is None or moving_plane.fov is None:
        raise ValueError("Both planes must provide FOV images for registration.")

    reg_img_elastix, itk_reg_all_roi = _load_track2p_registration_backend()
    registered_fov, reg_params = reg_img_elastix(
        np.asarray(reference_plane.fov),
        np.asarray(moving_plane.fov),
        SimpleNamespace(transform_type=transform_type),
    )
    moving_support_masks_hw_n = np.moveaxis(np.asarray(moving_plane.roi_masks) > 0, 0, -1)
    registered_support_masks = _coerce_registered_roi_masks(
        itk_reg_all_roi(moving_support_masks_hw_n, reg_params),
        n_rois=moving_plane.n_rois,
        image_shape=reference_plane.image_shape,
    )
    return moving_plane.with_replaced_masks(
        registered_support_masks,
        fov=np.asarray(registered_fov),
        source=f"{moving_plane.source}_registered",
    )


def register_consecutive_session_measurement_planes(
    sessions: Sequence[Track2pSession], *, transform_type: str = "affine"
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
    transform_type: str = "affine",
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
