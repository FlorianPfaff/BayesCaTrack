"""Compatibility patch for calibrated local-evidence association features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import calibrated_costs as _calibrated_costs

LOCAL_EVIDENCE_ASSOCIATION_FEATURES = (
    "one_minus_weighted_dice",
    "one_minus_overlap_min_fraction",
    "weighted_dice_cost",
    "overlap_fraction_cost",
    "containment_asymmetry_cost",
    "distance_transform_cost",
    "image_patch_cost",
    "neighbor_constellation_cost",
    "centroid_rank_cost",
)

_LOCAL_EVIDENCE_COST_FEATURES = {
    feature_name
    for feature_name in LOCAL_EVIDENCE_ASSOCIATION_FEATURES
    if not feature_name.startswith("one_minus_")
}
_ORIGINAL_FEATURE_TRANSFORMS_ATTR = (
    "_bayescatrack_original_feature_transforms_for_local_evidence"
)

if not hasattr(_calibrated_costs, _ORIGINAL_FEATURE_TRANSFORMS_ATTR):
    setattr(
        _calibrated_costs,
        _ORIGINAL_FEATURE_TRANSFORMS_ATTR,
        _calibrated_costs._feature_transforms_for,  # pylint: disable=protected-access
    )

_ORIGINAL_FEATURE_TRANSFORMS_FOR = getattr(
    _calibrated_costs,
    _ORIGINAL_FEATURE_TRANSFORMS_ATTR,
)


def _feature_transforms_for(
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Return calibrated feature transforms including local-evidence terms."""

    transforms = dict(_ORIGINAL_FEATURE_TRANSFORMS_FOR(feature_names))
    for feature_name in feature_names:
        if feature_name == "one_minus_weighted_dice":
            transforms[feature_name] = _optional_one_minus_component_transform(
                "weighted_dice_similarity"
            )
        elif feature_name == "one_minus_overlap_min_fraction":
            transforms[feature_name] = _optional_one_minus_component_transform(
                "overlap_min_fraction"
            )
        elif feature_name in _LOCAL_EVIDENCE_COST_FEATURES:
            transforms[feature_name] = _optional_zero_component_transform(feature_name)
    return transforms


def _optional_zero_component_transform(component_name: str) -> Any:
    def transform(pairwise_components: Mapping[str, Any]) -> np.ndarray:
        if component_name not in pairwise_components:
            return _calibrated_costs._zero_like_pairwise_component(  # pylint: disable=protected-access
                pairwise_components
            )
        return _calibrated_costs._finite_component(  # pylint: disable=protected-access
            pairwise_components, component_name
        )

    return transform


def _optional_one_minus_component_transform(component_name: str) -> Any:
    def transform(pairwise_components: Mapping[str, Any]) -> np.ndarray:
        if component_name not in pairwise_components:
            return _calibrated_costs._zero_like_pairwise_component(  # pylint: disable=protected-access
                pairwise_components
            )
        return 1.0 - np.clip(
            _calibrated_costs._finite_component(  # pylint: disable=protected-access
                pairwise_components, component_name
            ),
            0.0,
            1.0,
        )

    return transform


_calibrated_costs.LOCAL_EVIDENCE_ASSOCIATION_FEATURES = (
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES
)
_calibrated_costs._feature_transforms_for = _feature_transforms_for  # pylint: disable=protected-access
