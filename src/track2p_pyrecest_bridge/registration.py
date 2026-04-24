"""Registration-aware tracking utilities for ``track2p_pyrecest_bridge``.

The core bridge already knows how to:

* load Track2p / Suite2p ROI masks,
* build ROI-aware pairwise association costs, and
* build PyRecEst-ready association bundles.

This module adds the longitudinal tracking step of expressing a later session in
an earlier session's coordinate frame before association. It estimates a
session-to-session transform with PyRecEst point-set registration, warps ROI
masks into the reference frame, and then delegates to the existing bridge for
cost-matrix and association-bundle construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypedDict, Unpack

import numpy as np

from . import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_session_pair_association_bundle,
)

RegistrationModel = Literal["translation", "rigid", "affine"]


class _RegistrationKwargs(TypedDict, total=False):
    order: str
    weighted_centroids: bool
    registration_model: RegistrationModel
    registration_max_cost: float | None
    registration_max_iterations: int
    registration_tolerance: float
    min_matches: int | None
    allow_reflection: bool
    binarize_registered_masks: bool
    registered_mask_threshold: float


class _AssociationBundleKwargs(TypedDict, total=False):
    order: str
    weighted_centroids: bool
    velocity_variance: float
    regularization: float
    pairwise_cost_kwargs: Mapping[str, Any] | None
    return_pairwise_components: bool


class _RegisteredSessionPairKwargs(TypedDict, total=False):
    order: str
    weighted_centroids: bool
    velocity_variance: float
    regularization: float
    registration_model: RegistrationModel
    registration_max_cost: float | None
    registration_max_iterations: int
    registration_tolerance: float
    min_matches: int | None
    allow_reflection: bool
    pairwise_cost_kwargs: Mapping[str, Any] | None
    return_pairwise_components: bool
    binarize_registered_masks: bool
    registered_mask_threshold: float


_DEFAULT_REGISTERED_SESSION_PAIR_KWARGS: _RegisteredSessionPairKwargs = {
    "order": "xy",
    "weighted_centroids": False,
    "velocity_variance": 25.0,
    "regularization": 1e-6,
    "registration_model": "affine",
    "registration_max_cost": None,
    "registration_max_iterations": 25,
    "registration_tolerance": 1e-8,
    "min_matches": None,
    "allow_reflection": False,
    "pairwise_cost_kwargs": None,
    "return_pairwise_components": True,
    "binarize_registered_masks": False,
    "registered_mask_threshold": 0.5,
}


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class PlaneRegistrationBundle:
    """Result of registering one measurement plane to one reference plane."""

    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    order: str
    requested_model: RegistrationModel
    effective_model: RegistrationModel
    reference_to_measurement_matrix: np.ndarray
    reference_to_measurement_offset: np.ndarray
    measurement_to_reference_matrix: np.ndarray
    measurement_to_reference_offset: np.ndarray
    pyrecest_registration_result: Any


@dataclass(frozen=True)
class RegisteredSessionPairBundle:
    """Association bundle plus the registration that produced it."""

    plane_registration: PlaneRegistrationBundle
    association_bundle: SessionAssociationBundle


@dataclass(frozen=True)
class RegisteredConsecutiveBundles:
    """Sequence of registration-aware bundles for consecutive sessions."""

    bundles: list[RegisteredSessionPairBundle]


def _validate_order(order: str) -> str:
    if order not in {"xy", "yx"}:
        raise ValueError("order must be either 'xy' or 'yx'.")
    return order


def _equivalent_roi_diameter(plane: CalciumPlaneData) -> float:
    areas = np.asarray(plane.roi_areas(weighted=False), dtype=float)
    positive_areas = areas[areas > 0.0]
    if positive_areas.size == 0:
        return 1.0
    reference_area = float(np.median(positive_areas))
    return float(max(1.0, np.sqrt(4.0 * reference_area / np.pi)))


def _estimate_default_registration_max_cost(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
) -> float:
    pooled_diameter = max(
        _equivalent_roi_diameter(reference_plane),
        _equivalent_roi_diameter(measurement_plane),
    )
    return float(4.0 * pooled_diameter)


def _minimum_required_matches(model: RegistrationModel, *, dim: int = 2) -> int:
    if model == "translation":
        return 1
    if model == "rigid":
        return max(2, dim)
    if model == "affine":
        return dim + 1
    raise ValueError(f"Unsupported registration model: {model}")


def _choose_effective_model(
    requested_model: RegistrationModel,
    *,
    n_reference: int,
    n_measurement: int,
    dim: int = 2,
) -> RegistrationModel:
    available_matches = min(n_reference, n_measurement)
    for candidate in (requested_model, "rigid", "translation"):
        if available_matches >= _minimum_required_matches(candidate, dim=dim):
            return candidate
    raise ValueError(
        "At least one ROI is required in both sessions for registration-aware tracking."
    )


def _extract_centroid_points(
    plane: CalciumPlaneData,
    *,
    order: str,
    weighted_centroids: bool,
) -> np.ndarray:
    return np.asarray(
        plane.centroids(order=order, weighted=weighted_centroids).T,
        dtype=float,
    )


def _invert_affine(
    matrix: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    offset = np.asarray(offset, dtype=float).reshape(-1)
    inverse_matrix = np.linalg.inv(matrix)
    inverse_offset = -inverse_matrix @ offset
    return inverse_matrix, inverse_offset


def _registration_kwargs(bundle_kwargs: Mapping[str, Any]) -> _RegistrationKwargs:
    return {
        "order": bundle_kwargs["order"],
        "weighted_centroids": bundle_kwargs["weighted_centroids"],
        "registration_model": bundle_kwargs["registration_model"],
        "registration_max_cost": bundle_kwargs["registration_max_cost"],
        "registration_max_iterations": bundle_kwargs["registration_max_iterations"],
        "registration_tolerance": bundle_kwargs["registration_tolerance"],
        "min_matches": bundle_kwargs["min_matches"],
        "allow_reflection": bundle_kwargs["allow_reflection"],
        "binarize_registered_masks": bundle_kwargs["binarize_registered_masks"],
        "registered_mask_threshold": bundle_kwargs["registered_mask_threshold"],
    }


def _association_bundle_kwargs(bundle_kwargs: Mapping[str, Any]) -> _AssociationBundleKwargs:
    return {
        "order": bundle_kwargs["order"],
        "weighted_centroids": bundle_kwargs["weighted_centroids"],
        "velocity_variance": bundle_kwargs["velocity_variance"],
        "regularization": bundle_kwargs["regularization"],
        "pairwise_cost_kwargs": bundle_kwargs["pairwise_cost_kwargs"],
        "return_pairwise_components": bundle_kwargs["return_pairwise_components"],
    }


# pylint: disable=too-many-arguments
def _build_registration_ops(
    measurement_plane: CalciumPlaneData,
    *,
    registration_model: RegistrationModel,
    effective_model: RegistrationModel,
    order: str,
    reference_to_measurement_matrix: np.ndarray,
    reference_to_measurement_offset: np.ndarray,
    measurement_to_reference_matrix: np.ndarray,
    measurement_to_reference_offset: np.ndarray,
    registration_max_cost: float,
) -> dict[str, Any]:
    registration_ops = {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    registration_ops.update(
        {
            "pyrecest_registration_model_requested": registration_model,
            "pyrecest_registration_model_effective": effective_model,
            "pyrecest_registration_order": order,
            "pyrecest_reference_to_measurement_matrix": reference_to_measurement_matrix,
            "pyrecest_reference_to_measurement_offset": reference_to_measurement_offset,
            "pyrecest_measurement_to_reference_matrix": measurement_to_reference_matrix,
            "pyrecest_measurement_to_reference_offset": measurement_to_reference_offset,
            "pyrecest_registration_max_cost": registration_max_cost,
        }
    )
    return registration_ops


def _grid_points(image_shape: tuple[int, int], *, order: str) -> np.ndarray:
    rows, cols = np.indices(image_shape, dtype=float)
    if order == "xy":
        return np.stack([cols.ravel(), rows.ravel()], axis=1)
    return np.stack([rows.ravel(), cols.ravel()], axis=1)


def _split_sampling_coordinates(
    flat_points: np.ndarray,
    image_shape: tuple[int, int],
    *,
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    if order == "xy":
        x_coords = flat_points[:, 0].reshape(image_shape)
        y_coords = flat_points[:, 1].reshape(image_shape)
    else:
        y_coords = flat_points[:, 0].reshape(image_shape)
        x_coords = flat_points[:, 1].reshape(image_shape)
    return x_coords, y_coords


def _bilinear_sample(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width).")
    if x_coords.shape != y_coords.shape:
        raise ValueError("x_coords and y_coords must have the same shape.")

    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    dx = x_coords - x0
    dy = y_coords - y0

    result = np.zeros_like(x_coords, dtype=float)
    for x_offset, x_weight in ((0, 1.0 - dx), (1, dx)):
        for y_offset, y_weight in ((0, 1.0 - dy), (1, dy)):
            xi = x0 + x_offset
            yi = y0 + y_offset
            valid = (
                (0 <= xi)
                & (xi < image.shape[1])
                & (0 <= yi)
                & (yi < image.shape[0])
            )
            if not np.any(valid):
                continue
            result[valid] += (
                x_weight[valid] * y_weight[valid] * image[yi[valid], xi[valid]]
            )
    return result


def warp_image_into_reference_frame(
    image: np.ndarray,
    reference_to_measurement_matrix: np.ndarray,
    reference_to_measurement_offset: np.ndarray,
    *,
    output_shape: tuple[int, int],
    order: str = "xy",
) -> np.ndarray:
    """Warp a 2-D image into the reference frame by inverse sampling.

    Parameters
    ----------
    image
        Source image in the measurement-session frame.
    reference_to_measurement_matrix, reference_to_measurement_offset
        Transform that maps reference-frame coordinates to measurement-frame
        coordinates. This is exactly the transform returned by
        ``joint_registration_assignment`` when called with
        ``reference_points`` and ``moving_points``.
    output_shape
        Spatial shape ``(height, width)`` of the reference frame.
    order
        Coordinate order used by the registration stage.
    """

    order = _validate_order(order)
    flat_reference_points = _grid_points(output_shape, order=order)
    flat_measurement_points = (
        np.asarray(reference_to_measurement_matrix, dtype=float) @ flat_reference_points.T
    ).T + np.asarray(reference_to_measurement_offset, dtype=float).reshape(1, -1)
    x_coords, y_coords = _split_sampling_coordinates(
        flat_measurement_points,
        output_shape,
        order=order,
    )
    return _bilinear_sample(np.asarray(image, dtype=float), x_coords, y_coords)


# pylint: disable=too-many-arguments
def warp_roi_masks_into_reference_frame(
    roi_masks: np.ndarray,
    reference_to_measurement_matrix: np.ndarray,
    reference_to_measurement_offset: np.ndarray,
    *,
    output_shape: tuple[int, int],
    order: str = "xy",
    binarize: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Warp a stack of ROI masks into the reference frame."""

    order = _validate_order(order)
    roi_masks = np.asarray(roi_masks, dtype=float)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width).")
    warped_masks = np.zeros((roi_masks.shape[0], *output_shape), dtype=float)
    for roi_index, mask in enumerate(roi_masks):
        warped_masks[roi_index] = warp_image_into_reference_frame(
            mask,
            reference_to_measurement_matrix,
            reference_to_measurement_offset,
            output_shape=output_shape,
            order=order,
        )
    if binarize:
        return warped_masks >= threshold
    return warped_masks


# pylint: disable=too-many-arguments,too-many-locals
def register_measurement_plane_to_reference(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    order: str = "xy",
    weighted_centroids: bool = False,
    registration_model: RegistrationModel = "affine",
    registration_max_cost: float | None = None,
    registration_max_iterations: int = 25,
    registration_tolerance: float = 1e-8,
    min_matches: int | None = None,
    allow_reflection: bool = False,
    binarize_registered_masks: bool = False,
    registered_mask_threshold: float = 0.5,
) -> PlaneRegistrationBundle:
    """Register one measurement plane into one reference plane.

    The estimated transform is based on PyRecEst point-set registration over ROI
    centroids. The returned plane contains measurement-session ROI masks resampled
    into the reference frame so the existing ROI-aware pairwise costs in
    ``track2p_pyrecest_bridge`` can be applied directly.
    """

    order = _validate_order(order)
    reference_points = _extract_centroid_points(
        reference_plane,
        order=order,
        weighted_centroids=weighted_centroids,
    )
    measurement_points = _extract_centroid_points(
        measurement_plane,
        order=order,
        weighted_centroids=weighted_centroids,
    )
    effective_model = _choose_effective_model(
        registration_model,
        n_reference=reference_points.shape[0],
        n_measurement=measurement_points.shape[0],
        dim=reference_points.shape[1],
    )
    if registration_max_cost is None:
        registration_max_cost = _estimate_default_registration_max_cost(
            reference_plane,
            measurement_plane,
        )

    try:
        from pyrecest.utils.point_set_registration import (  # type: ignore[import-untyped]
            joint_registration_assignment,
        )
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "PyRecEst point-set registration support is required for "
            "register_measurement_plane_to_reference()."
        ) from exc

    registration_result = joint_registration_assignment(
        reference_points,
        measurement_points,
        model=effective_model,
        max_cost=float(registration_max_cost),
        max_iterations=registration_max_iterations,
        tolerance=registration_tolerance,
        min_matches=min_matches,
        allow_reflection=allow_reflection,
    )
    reference_to_measurement_matrix = np.asarray(
        registration_result.transform.matrix,
        dtype=float,
    )
    reference_to_measurement_offset = np.asarray(
        registration_result.transform.offset,
        dtype=float,
    ).reshape(-1)
    measurement_to_reference_matrix, measurement_to_reference_offset = _invert_affine(
        reference_to_measurement_matrix,
        reference_to_measurement_offset,
    )

    warped_masks = warp_roi_masks_into_reference_frame(
        measurement_plane.roi_masks,
        reference_to_measurement_matrix,
        reference_to_measurement_offset,
        output_shape=reference_plane.image_shape,
        order=order,
        binarize=binarize_registered_masks,
        threshold=registered_mask_threshold,
    )
    warped_fov = None
    if measurement_plane.fov is not None:
        warped_fov = warp_image_into_reference_frame(
            measurement_plane.fov,
            reference_to_measurement_matrix,
            reference_to_measurement_offset,
            output_shape=reference_plane.image_shape,
            order=order,
        )

    registration_ops = _build_registration_ops(
        measurement_plane,
        registration_model=registration_model,
        effective_model=effective_model,
        order=order,
        reference_to_measurement_matrix=reference_to_measurement_matrix,
        reference_to_measurement_offset=reference_to_measurement_offset,
        measurement_to_reference_matrix=measurement_to_reference_matrix,
        measurement_to_reference_offset=measurement_to_reference_offset,
        registration_max_cost=float(registration_max_cost),
    )

    registered_plane = measurement_plane.with_replaced_masks(
        warped_masks,
        fov=warped_fov,
        source=f"{measurement_plane.source}_registered_{effective_model}",
        ops=registration_ops,
    )
    return PlaneRegistrationBundle(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        order=order,
        requested_model=registration_model,
        effective_model=effective_model,
        reference_to_measurement_matrix=reference_to_measurement_matrix,
        reference_to_measurement_offset=reference_to_measurement_offset,
        measurement_to_reference_matrix=measurement_to_reference_matrix,
        measurement_to_reference_offset=measurement_to_reference_offset,
        pyrecest_registration_result=registration_result,
    )


# pylint: disable=too-many-arguments
def build_registered_session_pair_association_bundle(
    reference_session: Track2pSession,
    measurement_session: Track2pSession,
    *,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    registration_model: RegistrationModel = "affine",
    registration_max_cost: float | None = None,
    registration_max_iterations: int = 25,
    registration_tolerance: float = 1e-8,
    min_matches: int | None = None,
    allow_reflection: bool = False,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
    binarize_registered_masks: bool = False,
    registered_mask_threshold: float = 0.5,
) -> RegisteredSessionPairBundle:
    """Register the later session, then build the standard association bundle."""

    bundle_kwargs: _RegisteredSessionPairKwargs = {
        "order": order,
        "weighted_centroids": weighted_centroids,
        "velocity_variance": velocity_variance,
        "regularization": regularization,
        "registration_model": registration_model,
        "registration_max_cost": registration_max_cost,
        "registration_max_iterations": registration_max_iterations,
        "registration_tolerance": registration_tolerance,
        "min_matches": min_matches,
        "allow_reflection": allow_reflection,
        "pairwise_cost_kwargs": pairwise_cost_kwargs,
        "return_pairwise_components": return_pairwise_components,
        "binarize_registered_masks": binarize_registered_masks,
        "registered_mask_threshold": registered_mask_threshold,
    }
    registration_kwargs = _registration_kwargs(bundle_kwargs)
    association_kwargs = _association_bundle_kwargs(bundle_kwargs)

    plane_registration = register_measurement_plane_to_reference(
        reference_session.plane_data,
        measurement_session.plane_data,
        **registration_kwargs,
    )
    association_bundle = build_session_pair_association_bundle(
        reference_session,
        measurement_session,
        measurement_plane_in_reference_frame=plane_registration.registered_measurement_plane,
        **association_kwargs,
    )
    return RegisteredSessionPairBundle(
        plane_registration=plane_registration,
        association_bundle=association_bundle,
    )


def build_registered_consecutive_session_association_bundles(
    sessions: list[Track2pSession] | tuple[Track2pSession, ...],
    **bundle_kwargs: Unpack[_RegisteredSessionPairKwargs],
) -> RegisteredConsecutiveBundles:
    """Build one registration-aware association bundle for each consecutive pair."""

    effective_bundle_kwargs: _RegisteredSessionPairKwargs = {
        **_DEFAULT_REGISTERED_SESSION_PAIR_KWARGS,
        **bundle_kwargs,
    }
    consecutive_bundles: list[RegisteredSessionPairBundle] = []
    sessions = list(sessions)
    for pair_index in range(len(sessions) - 1):
        consecutive_bundles.append(
            build_registered_session_pair_association_bundle(
                sessions[pair_index],
                sessions[pair_index + 1],
                **effective_bundle_kwargs,
            )
        )
    return RegisteredConsecutiveBundles(bundles=consecutive_bundles)
