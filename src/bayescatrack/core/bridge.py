"""Public bridge exports for BayesCaTrack core."""

# pylint: disable=duplicate-code

from .._exports import BRIDGE_PUBLIC_NAMES
from . import _bridge_impl
from . import _mahalanobis as _mahalanobis

_mahalanobis.install_mahalanobis_pairwise_features(_bridge_impl.CalciumPlaneData)

CalciumPlaneData = _bridge_impl.CalciumPlaneData
SessionAssociationBundle = _bridge_impl.SessionAssociationBundle
Track2pSession = _bridge_impl.Track2pSession
build_consecutive_session_association_bundles = (
    _bridge_impl.build_consecutive_session_association_bundles
)
build_session_pair_association_bundle = (
    _bridge_impl.build_session_pair_association_bundle
)
export_subject_to_npz = _bridge_impl.export_subject_to_npz
find_track2p_session_dirs = _bridge_impl.find_track2p_session_dirs
load_raw_npy_plane = _bridge_impl.load_raw_npy_plane
load_suite2p_plane = _bridge_impl.load_suite2p_plane
load_track2p_subject = _bridge_impl.load_track2p_subject
main = _bridge_impl.main
summarize_subject = _bridge_impl.summarize_subject

__all__ = tuple(name for name in BRIDGE_PUBLIC_NAMES if name in globals())
