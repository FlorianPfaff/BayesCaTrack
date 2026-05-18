"""Association helpers for BayesCaTrack."""

from .._exports import ASSOCIATION_PUBLIC_NAMES, reexport
from ..core import bridge as _bridge
from . import (
    _calibrated_mahalanobis_bundle_patch as _calibrated_mahalanobis_bundle_patch,
)
from . import (
    _calibrated_roi_stat_feature_patch as _calibrated_roi_stat_feature_patch,
)

_PATCH_MODULES = (
    _calibrated_mahalanobis_bundle_patch,
    _calibrated_roi_stat_feature_patch,
)

__all__ = reexport(_bridge, globals(), ASSOCIATION_PUBLIC_NAMES)
