"""Automatic registration-model selection for Track2p-style session pairs.

The selector is deliberately conservative: it only accepts a more flexible
registration when it improves FOV agreement while preserving most ROI support.
This makes it suitable for benchmark sweeps where a single hard-coded transform
can over-warp some session pairs and under-warp others.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData

DEFAULT_AUTO_REGISTRATION_CANDIDATES: tuple[str, ...] = (
    "none",
    "fov-translation",
    "affine",
    "rigid",
)

_DEFAULT_COMPLEXITY_PENALTY: Mapping[str, float] = {
    "none": 0.0,
    "fov-translation": 0.01,
    "fov-affine": 0.025,
    "affine": 0.035,
    "rigid": 0.03,
}


@dataclass(frozen=True)
class RegistrationCandidateDiagnostics:
    """Quality summary for one candidate registration transform."""

    transform_type: str
    accepted: bool
    score: float
    fov_correlation: float
    empty_roi_fraction: float
    retained_mask_area_fraction: float
    effective_transform_type: str | None = None
    backend: str | None = None
    reason: str | None = None
    failure: str | None = None


@dataclass(frozen=True)
class RegistrationSelectionResult:
    """Selected registration plane and all diagnostics used to select it."""

    registered_plane: CalciumPlaneData
    selected_transform_type: str
    diagnostics: tuple[RegistrationCandidateDiagnostics, ...]

    @property
    def selected_diagnostics(self) -> RegistrationCandidateDiagnostics:
        for diagnostics in self.diagnostics:
            if diagnostics.transform_type == self.selected_transform_type:
                return diagnostics
        raise RuntimeError("selected transform is missing from diagnostics")


# pylint: disable=too-many-arguments,too-many-locals
def select_registration_transform(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    candidate_transforms: Sequence[str] = DEFAULT_AUTO_REGISTRATION_CANDIDATES,
    min_fov_correlation_gain: float = 0.02,
    max_empty_roi_fraction: float = 0.1,
    min_retained_mask_area_fraction: float = 0.5,
    empty_roi_penalty: float = 0.75,
    retained_area_penalty: float = 0.5,
    complexity_penalty: Mapping[str, float] | None = None,
) -> RegistrationSelectionResult:
    """Select a registration transform for one session pair.

    Each candidate is applied to ``moving_plane`` and scored against
    ``reference_plane``. Candidates with excessive empty registered ROIs or too
    much lost ROI support are rejected. Among accepted candidates, the selector
    maximizes a small FOV-correlation objective with a complexity penalty. If
    the identity/no-registration candidate is valid, a non-identity candidate
    must improve FOV correlation by ``min_fov_correlation_gain`` to be selected.
    """

    if min_fov_correlation_gain < 0.0:
        raise ValueError("min_fov_correlation_gain must be non-negative")
    if not 0.0 <= max_empty_roi_fraction <= 1.0:
        raise ValueError("max_empty_roi_fraction must be between 0 and 1")
    if min_retained_mask_area_fraction < 0.0:
        raise ValueError("min_retained_mask_area_fraction must be non-negative")
    if empty_roi_penalty < 0.0 or retained_area_penalty < 0.0:
        raise ValueError("selection penalties must be non-negative")

    penalties = dict(_DEFAULT_COMPLEXITY_PENALTY)
    if complexity_penalty is not None:
        penalties.update(
            {key: float(value) for key, value in complexity_penalty.items()}
        )

    diagnostics: list[RegistrationCandidateDiagnostics] = []
    registered_planes: dict[str, CalciumPlaneData] = {}
    for transform_type in _unique_candidate_transforms(candidate_transforms):
        try:
            registered_plane = _candidate_registered_plane(
                reference_plane,
                moving_plane,
                transform_type=transform_type,
            )
            candidate_diagnostics = _diagnose_candidate(
                reference_plane,
                moving_plane,
                registered_plane,
                requested_transform_type=transform_type,
                complexity_penalty=float(penalties.get(transform_type, 0.04)),
                max_empty_roi_fraction=max_empty_roi_fraction,
                min_retained_mask_area_fraction=min_retained_mask_area_fraction,
                empty_roi_penalty=empty_roi_penalty,
                retained_area_penalty=retained_area_penalty,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            diagnostics.append(
                RegistrationCandidateDiagnostics(
                    transform_type=transform_type,
                    accepted=False,
                    score=float("-inf"),
                    fov_correlation=float("nan"),
                    empty_roi_fraction=1.0,
                    retained_mask_area_fraction=0.0,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        registered_planes[transform_type] = registered_plane
        diagnostics.append(candidate_diagnostics)

    accepted = [candidate for candidate in diagnostics if candidate.accepted]
    if not accepted:
        failures = "; ".join(
            f"{candidate.transform_type}: {candidate.failure or 'rejected'}"
            for candidate in diagnostics
        )
        raise ValueError(
            f"No automatic registration candidate was acceptable: {failures}"
        )

    selected = max(accepted, key=lambda candidate: candidate.score)
    identity = next(
        (candidate for candidate in accepted if candidate.transform_type == "none"),
        None,
    )
    if (
        identity is not None
        and selected.transform_type != "none"
        and selected.fov_correlation
        < identity.fov_correlation + float(min_fov_correlation_gain)
    ):
        selected = identity

    selected_plane = _with_auto_registration_metadata(
        registered_planes[selected.transform_type],
        selected=selected,
        diagnostics=diagnostics,
        min_fov_correlation_gain=float(min_fov_correlation_gain),
        max_empty_roi_fraction=float(max_empty_roi_fraction),
        min_retained_mask_area_fraction=float(min_retained_mask_area_fraction),
    )
    return RegistrationSelectionResult(
        registered_plane=selected_plane,
        selected_transform_type=selected.transform_type,
        diagnostics=tuple(diagnostics),
    )


def _unique_candidate_transforms(
    candidate_transforms: Sequence[str],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for transform_type in candidate_transforms:
        transform_type = str(transform_type)
        if transform_type == "auto":
            raise ValueError(
                "'auto' must not be nested inside auto-registration candidates"
            )
        if transform_type not in candidates:
            candidates.append(transform_type)
    if not candidates:
        raise ValueError("At least one registration candidate is required")
    return tuple(candidates)


def _candidate_registered_plane(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str,
) -> CalciumPlaneData:
    if transform_type == "none":
        if reference_plane.image_shape != moving_plane.image_shape:
            raise ValueError("transform_type='none' requires matching image shapes")
        return _identity_registered_plane(moving_plane)

    from bayescatrack.track2p_registration import register_plane_pair

    return register_plane_pair(
        reference_plane,
        moving_plane,
        transform_type=transform_type,
    )


def _identity_registered_plane(moving_plane: CalciumPlaneData) -> CalciumPlaneData:
    ops = {} if moving_plane.ops is None else dict(moving_plane.ops)
    ops.update(
        {
            "registration_backend": "none",
            "registration_transform_type": "none",
            "registration_backend_reason": (
                "automatic selection evaluated identity/no registration"
            ),
        }
    )
    return moving_plane.with_replaced_masks(
        moving_plane.roi_masks,
        fov=moving_plane.fov,
        source=f"{moving_plane.source}_unregistered",
        ops=ops,
    )


# pylint: disable=too-many-arguments
def _diagnose_candidate(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    registered_plane: CalciumPlaneData,
    *,
    requested_transform_type: str,
    complexity_penalty: float,
    max_empty_roi_fraction: float,
    min_retained_mask_area_fraction: float,
    empty_roi_penalty: float,
    retained_area_penalty: float,
) -> RegistrationCandidateDiagnostics:
    fov_correlation = _fov_correlation(reference_plane.fov, registered_plane.fov)
    empty_roi_fraction = _empty_roi_fraction(registered_plane.roi_masks)
    retained_area_fraction = _retained_mask_area_fraction(
        moving_plane.roi_masks,
        registered_plane.roi_masks,
    )
    retained_shortfall = max(
        0.0,
        min_retained_mask_area_fraction - retained_area_fraction,
    )
    score = (
        fov_correlation
        - complexity_penalty
        - empty_roi_penalty * empty_roi_fraction
        - retained_area_penalty * retained_shortfall
    )
    accepted = (
        np.isfinite(fov_correlation)
        and empty_roi_fraction <= max_empty_roi_fraction
        and retained_area_fraction >= min_retained_mask_area_fraction
    )
    ops = registered_plane.ops or {}
    reason = None
    if not accepted:
        reason = _rejection_reason(
            fov_correlation=fov_correlation,
            empty_roi_fraction=empty_roi_fraction,
            retained_area_fraction=retained_area_fraction,
            max_empty_roi_fraction=max_empty_roi_fraction,
            min_retained_mask_area_fraction=min_retained_mask_area_fraction,
        )
    return RegistrationCandidateDiagnostics(
        transform_type=requested_transform_type,
        accepted=accepted,
        score=float(score),
        fov_correlation=float(fov_correlation),
        empty_roi_fraction=float(empty_roi_fraction),
        retained_mask_area_fraction=float(retained_area_fraction),
        effective_transform_type=_maybe_str(ops.get("registration_transform_type")),
        backend=_maybe_str(ops.get("registration_backend")),
        reason=reason,
    )


def _rejection_reason(
    *,
    fov_correlation: float,
    empty_roi_fraction: float,
    retained_area_fraction: float,
    max_empty_roi_fraction: float,
    min_retained_mask_area_fraction: float,
) -> str:
    if not np.isfinite(fov_correlation):
        return "non-finite FOV correlation"
    if empty_roi_fraction > max_empty_roi_fraction:
        return "too many registered ROIs became empty"
    if retained_area_fraction < min_retained_mask_area_fraction:
        return "too much registered ROI support was lost"
    return "candidate rejected"


def _fov_correlation(
    reference_fov: np.ndarray | None,
    registered_fov: np.ndarray | None,
) -> float:
    if reference_fov is None or registered_fov is None:
        return float("nan")
    reference = np.asarray(reference_fov, dtype=float)
    registered = np.asarray(registered_fov, dtype=float)
    if reference.ndim != 2 or registered.ndim != 2:
        return float("nan")
    if reference.shape != registered.shape:
        common_shape = (
            max(int(reference.shape[0]), int(registered.shape[0])),
            max(int(reference.shape[1]), int(registered.shape[1])),
        )
        reference = _pad_to_shape(reference, common_shape)
        registered = _pad_to_shape(registered, common_shape)
    reference = reference - float(np.mean(reference))
    registered = registered - float(np.mean(registered))
    denominator = float(
        np.linalg.norm(reference.ravel()) * np.linalg.norm(registered.ravel())
    )
    if denominator <= 0.0:
        return 0.0
    normalized_dot = np.dot(reference.ravel(), registered.ravel()) / denominator
    return float(np.clip(normalized_dot, -1.0, 1.0))


def _empty_roi_fraction(roi_masks: np.ndarray) -> float:
    masks = np.asarray(roi_masks)
    if masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if masks.shape[0] == 0:
        return 0.0
    support = np.asarray(masks > 0.0).reshape(masks.shape[0], -1)
    return float(np.mean(np.sum(support, axis=1) == 0))


def _retained_mask_area_fraction(
    source_roi_masks: np.ndarray,
    registered_roi_masks: np.ndarray,
) -> float:
    source_area = float(np.sum(np.asarray(source_roi_masks) > 0.0))
    registered_area = float(np.sum(np.asarray(registered_roi_masks) > 0.0))
    if source_area <= 0.0:
        return 1.0 if registered_area <= 0.0 else 0.0
    return float(registered_area / source_area)


def _pad_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape == shape:
        return image
    output = np.zeros(shape, dtype=float)
    output[: image.shape[0], : image.shape[1]] = image
    return output


def _with_auto_registration_metadata(
    plane: CalciumPlaneData,
    *,
    selected: RegistrationCandidateDiagnostics,
    diagnostics: Sequence[RegistrationCandidateDiagnostics],
    min_fov_correlation_gain: float,
    max_empty_roi_fraction: float,
    min_retained_mask_area_fraction: float,
) -> CalciumPlaneData:
    ops = {} if plane.ops is None else dict(plane.ops)
    ops.update(
        {
            "registration_auto_selected_transform": selected.transform_type,
            "registration_auto_selected_effective_transform": (
                selected.effective_transform_type
            ),
            "registration_auto_selected_backend": selected.backend,
            "registration_auto_score": float(selected.score),
            "registration_auto_fov_correlation": float(selected.fov_correlation),
            "registration_auto_empty_roi_fraction": float(selected.empty_roi_fraction),
            "registration_auto_retained_mask_area_fraction": float(
                selected.retained_mask_area_fraction
            ),
            "registration_auto_min_fov_correlation_gain": float(
                min_fov_correlation_gain
            ),
            "registration_auto_max_empty_roi_fraction": float(max_empty_roi_fraction),
            "registration_auto_min_retained_mask_area_fraction": float(
                min_retained_mask_area_fraction
            ),
            "registration_auto_candidate_diagnostics": [
                asdict(candidate) for candidate in diagnostics
            ],
        }
    )
    return plane.with_replaced_masks(
        plane.roi_masks,
        fov=plane.fov,
        source=f"{plane.source}_auto_selected_{selected.transform_type}",
        ops=ops,
    )


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
