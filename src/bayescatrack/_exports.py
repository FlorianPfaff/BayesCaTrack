"""Shared export groups and re-export helpers for BayesCaTrack modules."""

# pylint: disable=duplicate-code

from __future__ import annotations

from types import ModuleType
from typing import Any


def _public_names(*names: str) -> tuple[str, ...]:
    """Return export names without validating them in this helper module."""

    return names


BRIDGE_PUBLIC_NAMES = _public_names(
    "CalciumPlaneData",
    "SessionAssociationBundle",
    "Track2pSession",
    "build_consecutive_session_association_bundles",
    "build_session_pair_association_bundle",
    "export_subject_to_npz",
    "find_track2p_session_dirs",
    "load_raw_npy_plane",
    "load_suite2p_plane",
    "load_track2p_subject",
    "main",
    "summarize_subject",
)

ASSOCIATION_PUBLIC_NAMES = BRIDGE_PUBLIC_NAMES[:5]

TRACK2P_DATASET_PUBLIC_NAMES = _public_names(
    "Track2pSession",
    "export_subject_to_npz",
    "find_track2p_session_dirs",
    "load_raw_npy_plane",
    "load_track2p_subject",
    "summarize_subject",
)

IO_PUBLIC_NAMES = (*TRACK2P_DATASET_PUBLIC_NAMES, "load_suite2p_plane")

REFERENCE_PUBLIC_NAMES = _public_names(
    "Track2pReference",
    "load_aligned_subject_reference",
    "load_track2p_reference",
    "pairs_from_label_vectors",
    "score_label_vectors_against_reference",
    "score_pairwise_matches",
)

REGISTRATION_PUBLIC_NAMES = _public_names(
    "PlaneRegistrationBundle",
    "RegisteredConsecutiveBundles",
    "RegisteredSessionPairBundle",
    "RegistrationModel",
    "build_registered_consecutive_session_association_bundles",
    "build_registered_session_pair_association_bundle",
    "register_measurement_plane_to_reference",
    "warp_image_into_reference_frame",
    "warp_roi_masks_into_reference_frame",
)

TRACK2P_REGISTRATION_PUBLIC_NAMES = _public_names(
    "build_registered_subject_association_bundles",
    "register_consecutive_session_measurement_planes",
    "register_plane_pair",
)


def reexport(
    source: ModuleType,
    target_globals: dict[str, Any],
    names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Copy named attributes from ``source`` into a target module namespace."""

    export_names = tuple(getattr(source, "__all__", ())) if names is None else tuple(names)
    target_globals.update({name: getattr(source, name) for name in export_names})
    return export_names
