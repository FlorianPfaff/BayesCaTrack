"""Compatibility patch for calibrated split Suite2p ROI-stat features."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bayescatrack.core._roi_stat_features import SPLIT_ROI_STAT_FEATURES

from . import calibrated_costs as _calibrated_costs

_ORIGINAL_DEFAULT_ASSOCIATION_FEATURES = _calibrated_costs.DEFAULT_ASSOCIATION_FEATURES
_ORIGINAL_FEATURE_TRANSFORMS_FOR_ATTR = "_bayescatrack_original_feature_transforms_for"

_patched_default_association_features: list[str] = []
for _feature_name in _ORIGINAL_DEFAULT_ASSOCIATION_FEATURES:
    if _feature_name == "roi_feature_cost":
        continue
    if _feature_name == "cell_probability_cost":
        _patched_default_association_features.extend(SPLIT_ROI_STAT_FEATURES)
    _patched_default_association_features.append(_feature_name)
DEFAULT_ASSOCIATION_FEATURES: tuple[str, ...] = tuple(
    _patched_default_association_features
)

if not hasattr(_calibrated_costs, _ORIGINAL_FEATURE_TRANSFORMS_FOR_ATTR):
    setattr(
        _calibrated_costs,
        _ORIGINAL_FEATURE_TRANSFORMS_FOR_ATTR,
        _calibrated_costs._feature_transforms_for,  # pylint: disable=protected-access
    )

_ORIGINAL_FEATURE_TRANSFORMS_FOR: Callable[[Any], dict[str, Any]] = getattr(
    _calibrated_costs,
    _ORIGINAL_FEATURE_TRANSFORMS_FOR_ATTR,
)


def _feature_transforms_for(feature_names: Any) -> dict[str, Any]:
    transforms = _ORIGINAL_FEATURE_TRANSFORMS_FOR(feature_names)
    for feature_name in tuple(feature_names):
        if feature_name in SPLIT_ROI_STAT_FEATURES or feature_name == (
            "mahalanobis_centroid_distance"
        ):
            transforms.setdefault(
                feature_name,
                _calibrated_costs._optional_zero_component_transform(  # pylint: disable=protected-access
                    feature_name
                ),
            )
    return transforms


def _replace_default_feature_tuple(function: Any) -> None:
    defaults = getattr(function, "__defaults__", None)
    if defaults:
        function.__defaults__ = tuple(
            (
                DEFAULT_ASSOCIATION_FEATURES
                if value == _ORIGINAL_DEFAULT_ASSOCIATION_FEATURES
                else value
            )
            for value in defaults
        )
    keyword_defaults = getattr(function, "__kwdefaults__", None)
    if keyword_defaults:
        function.__kwdefaults__ = {
            key: (
                DEFAULT_ASSOCIATION_FEATURES
                if value == _ORIGINAL_DEFAULT_ASSOCIATION_FEATURES
                else value
            )
            for key, value in keyword_defaults.items()
        }


def _patch_dataclass_default(class_object: Any, field_name: str) -> None:
    fields = getattr(class_object, "__dataclass_fields__", {})
    field = fields.get(field_name)
    if field is not None and field.default == _ORIGINAL_DEFAULT_ASSOCIATION_FEATURES:
        field.default = DEFAULT_ASSOCIATION_FEATURES
    _replace_default_feature_tuple(class_object.__init__)


setattr(_calibrated_costs, "SPLIT_ROI_STAT_FEATURES", SPLIT_ROI_STAT_FEATURES)
setattr(_calibrated_costs, "DEFAULT_ASSOCIATION_FEATURES", DEFAULT_ASSOCIATION_FEATURES)
# pylint: disable-next=protected-access
_calibrated_costs._feature_transforms_for = _feature_transforms_for

_patch_dataclass_default(_calibrated_costs.CalibratedAssociationModel, "feature_names")
_patch_dataclass_default(_calibrated_costs.ReferenceTrainingOptions, "feature_names")
_replace_default_feature_tuple(_calibrated_costs.pairwise_feature_schema)
_replace_default_feature_tuple(_calibrated_costs.pairwise_feature_tensor)
_replace_default_feature_tuple(_calibrated_costs.fit_logistic_association_model)
