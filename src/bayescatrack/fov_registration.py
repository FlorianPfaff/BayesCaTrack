from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_consecutive_session_association_bundles,
    build_session_pair_association_bundle,
)


@dataclass(frozen=True)
class FovTranslationRegistration:
    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    measurement_to_reference_shift_yx: np.ndarray
    reference_to_measurement_shift_yx: np.ndarray
    peak_correlation: float


@dataclass(frozen=True)
class FovRegisteredSessionPairBundle:
    registration: FovTranslationRegistration
    association_bundle: SessionAssociationBundle


@dataclass(frozen=True)
class _FovAssociationOptions:
    subtract_mean: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1e-6
    pairwise_cost_kwargs: Mapping[str, Any] | None = None
    return_pairwise_components: bool = True

    def registration_kwargs(self) -> dict[str, Any]:
        return {"subtract_mean": self.subtract_mean}

    def association_kwargs(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "weighted_centroids": self.weighted_centroids,
            "velocity_variance": self.velocity_variance,
            "regularization": self.regularization,
            "pairwise_cost_kwargs": self.pairwise_cost_kwargs,
            "return_pairwise_components": self.return_pairwise_components,
        }


_FOV_ASSOCIATION_OPTION_DEFAULTS: dict[str, Any] = {
    "subtract_mean": True,
    "order": "xy",
    "weighted_centroids": False,
    "velocity_variance": 25.0,
    "regularization": 1e-6,
    "pairwise_cost_kwargs": None,
    "return_pairwise_components": True,
}


def _fov_association_options_from_kwargs(
    bundle_kwargs: Mapping[str, Any],
) -> _FovAssociationOptions:
    unexpected_names = sorted(
        set(bundle_kwargs).difference(_FOV_ASSOCIATION_OPTION_DEFAULTS)
    )
    if unexpected_names:
        joined_names = ", ".join(unexpected_names)
        raise TypeError(f"Unexpected FOV registration option(s): {joined_names}")
    return _FovAssociationOptions(
        **{
            **_FOV_ASSOCIATION_OPTION_DEFAULTS,
            **bundle_kwargs,
        }
    )


def estimate_integer_fov_shift(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, float]:
    """Return the integer shift to apply to ``measurement_fov`` to align it with ``reference_fov``."""

    reference = np.asarray(reference_fov, dtype=float)
    measurement = np.asarray(measurement_fov, dtype=float)
    if reference.ndim != 2 or measurement.ndim != 2:
        raise ValueError("reference_fov and measurement_fov must both be 2-D arrays")
    if reference.shape != measurement.shape:
        raise ValueError("reference_fov and measurement_fov must have identical shape")
    if subtract_mean:
        reference = reference - float(np.mean(reference))
        measurement = measurement - float(np.mean(measurement))
    cross_power = np.fft.fftn(reference) * np.conj(np.fft.fftn(measurement))
    magnitude = np.abs(cross_power)
    magnitude[magnitude == 0.0] = 1.0
    correlation = np.abs(np.fft.ifftn(cross_power / magnitude))
    peak_index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift_yx = np.asarray(peak_index, dtype=int)
    for axis, size in enumerate(reference.shape):
        if shift_yx[axis] > size // 2:
            shift_yx[axis] -= size
    return shift_yx, float(correlation[peak_index])


def _overlap_slices(size: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, size - shift), slice(shift, size)
    return slice(-shift, size), slice(0, size + shift)


def apply_integer_image_translation(
    image: np.ndarray,
    shift_yx: Sequence[int] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    fill_value: float | bool = 0.0,
) -> np.ndarray:
    """Translate one 2-D image with zero padding."""

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")
    shift_y, shift_x = (int(v) for v in np.asarray(shift_yx, dtype=int))
    if output_shape is None:
        output_shape = image.shape
    result = np.full(output_shape, fill_value, dtype=image.dtype)
    src_y, dst_y = _overlap_slices(image.shape[0], shift_y)
    src_x, dst_x = _overlap_slices(image.shape[1], shift_x)
    result[dst_y, dst_x] = image[src_y, src_x]
    return result


def apply_integer_roi_mask_translation(
    roi_masks: np.ndarray,
    shift_yx: Sequence[int] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Translate an ROI-mask stack with zero padding."""

    roi_masks = np.asarray(roi_masks)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if output_shape is None:
        output_shape = (int(roi_masks.shape[1]), int(roi_masks.shape[2]))
    translated = np.zeros((roi_masks.shape[0], *output_shape), dtype=roi_masks.dtype)
    fill_value = False if roi_masks.dtype == np.bool_ else 0.0
    for roi_index, mask in enumerate(roi_masks):
        translated[roi_index] = apply_integer_image_translation(
            mask,
            shift_yx,
            output_shape=output_shape,
            fill_value=fill_value,
        )
    return translated


def register_measurement_plane_by_fov_translation(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    subtract_mean: bool = True,
) -> FovTranslationRegistration:
    """Align ``measurement_plane`` to ``reference_plane`` using FOV phase correlation."""

    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError(
            "Both planes must provide fov images for FOV-based registration"
        )
    shift_yx, peak_correlation = estimate_integer_fov_shift(
        reference_plane.fov,
        measurement_plane.fov,
        subtract_mean=subtract_mean,
    )
    registered_masks = apply_integer_roi_mask_translation(
        measurement_plane.roi_masks,
        shift_yx,
        output_shape=reference_plane.image_shape,
    )
    registered_fov = apply_integer_image_translation(
        measurement_plane.fov,
        shift_yx,
        output_shape=reference_plane.image_shape,
        fill_value=0.0,
    )
    registration_ops = (
        {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    )
    registration_ops.update(
        {
            "fov_registration_method": "phase_correlation_translation",
            "fov_registration_measurement_to_reference_shift_yx": shift_yx.astype(int),
            "fov_registration_reference_to_measurement_shift_yx": (-shift_yx).astype(
                int
            ),
            "fov_registration_peak_correlation": peak_correlation,
        }
    )
    registered_plane = measurement_plane.with_replaced_masks(
        registered_masks,
        fov=registered_fov,
        source=f"{measurement_plane.source}_fov_registered",
        ops=registration_ops,
    )
    return FovTranslationRegistration(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        measurement_to_reference_shift_yx=shift_yx.astype(int),
        reference_to_measurement_shift_yx=(-shift_yx).astype(int),
        peak_correlation=peak_correlation,
    )


# pylint: disable=too-many-arguments
def build_fov_registered_session_pair_association_bundle(
    reference_session: Track2pSession,
    measurement_session: Track2pSession,
    *,
    subtract_mean: bool = True,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
) -> FovRegisteredSessionPairBundle:
    """Register the later session by FOV, then build the standard association bundle."""

    options = _fov_association_options_from_kwargs(
        {
            "subtract_mean": subtract_mean,
            "order": order,
            "weighted_centroids": weighted_centroids,
            "velocity_variance": velocity_variance,
            "regularization": regularization,
            "pairwise_cost_kwargs": pairwise_cost_kwargs,
            "return_pairwise_components": return_pairwise_components,
        }
    )
    registration = register_measurement_plane_by_fov_translation(
        reference_session.plane_data,
        measurement_session.plane_data,
        **options.registration_kwargs(),
    )
    association_bundle = build_session_pair_association_bundle(
        reference_session,
        measurement_session,
        measurement_plane_in_reference_frame=registration.registered_measurement_plane,
        **options.association_kwargs(),
    )
    return FovRegisteredSessionPairBundle(
        registration=registration,
        association_bundle=association_bundle,
    )


def build_fov_registered_consecutive_session_association_bundles(
    sessions: Sequence[Track2pSession],
    **bundle_kwargs: Any,
) -> list[FovRegisteredSessionPairBundle]:
    """Build FOV-registered association bundles for all consecutive session pairs."""

    options = _fov_association_options_from_kwargs(bundle_kwargs)
    session_list = list(sessions)
    registrations = [
        register_measurement_plane_by_fov_translation(
            session_list[pair_index].plane_data,
            session_list[pair_index + 1].plane_data,
            **options.registration_kwargs(),
        )
        for pair_index in range(len(session_list) - 1)
    ]
    association_bundles = build_consecutive_session_association_bundles(
        session_list,
        measurement_planes_in_reference_frames=[
            registration.registered_measurement_plane for registration in registrations
        ],
        **options.association_kwargs(),
    )
    return [
        FovRegisteredSessionPairBundle(
            registration=registration,
            association_bundle=association_bundle,
        )
        for registration, association_bundle in zip(registrations, association_bundles)
    ]
