"""BayesCaTrack public package API."""

# pylint: disable=duplicate-code

from . import cli as _cli
from .core import bridge as _bridge
from .soft_overlap_costs import (
    install_soft_overlap_costs as _install_soft_overlap_costs,
)


CalciumPlaneData = _bridge.CalciumPlaneData
SessionAssociationBundle = _bridge.SessionAssociationBundle
Track2pSession = _bridge.Track2pSession
build_consecutive_session_association_bundles = (
    _bridge.build_consecutive_session_association_bundles
)
build_session_pair_association_bundle = _bridge.build_session_pair_association_bundle
export_subject_to_npz = _bridge.export_subject_to_npz
find_track2p_session_dirs = _bridge.find_track2p_session_dirs
load_raw_npy_plane = _bridge.load_raw_npy_plane
load_suite2p_plane = _bridge.load_suite2p_plane
load_track2p_subject = _bridge.load_track2p_subject
main = _cli.main
summarize_subject = _bridge.summarize_subject

_install_soft_overlap_costs()

__all__ = tuple(dict.fromkeys((*_bridge.__all__, "main")))
