"""Split Suite2p ROI-stat feature extension for Track2p ROI association."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_ROI_STAT_FEATURES_INSTALLED_ATTR = "_bayescatrack_roi_stat_features_installed"

SPLIT_ROI_STAT_FEATURES = (
    "abs_log_radius_ratio",
    "abs_aspect_ratio_diff",
    "abs_compact_diff",
    "abs_footprint_diff",
    "abs_skew_diff",
    "abs_std_diff",
    "abs_log_npix_ratio",
    "abs_npix_norm_diff",
    "missing_radius_indicator",
    "missing_stat_indicator",
)

_SPLIT_ROI_FEATURE_COMPONENT_SPECS = (
    ("radius", "abs_log_radius_ratio", "log_ratio"),
    ("aspect_ratio", "abs_aspect_ratio_diff", "scaled_absdiff"),
    ("compact", "abs_compact_diff", "scaled_absdiff"),
    ("footprint", "abs_footprint_diff", "scaled_absdiff"),
    ("skew", "abs_skew_diff", "scaled_absdiff"),
    ("std", "abs_std_diff", "scaled_absdiff"),
    ("npix", "abs_log_npix_ratio", "log_ratio"),
    ("npix_norm", "abs_npix_norm_diff", "scaled_absdiff"),
)


def install_split_roi_stat_pairwise_features(calcium_plane_cls: type[Any]) -> None:
    """Install split Suite2p ROI-stat components on ``CalciumPlaneData``.

    The core bridge exposes the legacy scalar ``roi_feature_cost`` by averaging
    all available Suite2p ROI statistics. This extension keeps that scalar for
    backward compatibility but additionally exposes one pairwise component per
    raw ROI statistic plus explicit missingness indicators for raw-NPY or
    partially populated Suite2p inputs.
    """

    if getattr(calcium_plane_cls, _ROI_STAT_FEATURES_INSTALLED_ATTR, False):
        return

    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    # pylint: disable=too-many-arguments,too-many-locals
    def build_pairwise_cost_matrix(
        self: Any,
        other: Any,
        *args: Any,
        feature_names: Sequence[str] | None = None,
        similarity_epsilon: float = 1.0e-6,
        return_components: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build a ROI-aware cost matrix with split ROI-stat diagnostics."""

        if similarity_epsilon <= 0.0:
            raise ValueError("similarity_epsilon must be strictly positive")

        result = original_build_pairwise_cost_matrix(
            self,
            other,
            *args,
            feature_names=_normalize_roi_feature_names(feature_names),
            similarity_epsilon=similarity_epsilon,
            return_components=True,
            **kwargs,
        )
        total_cost, components = result
        components = dict(components)
        components.update(
            _pairwise_roi_feature_components(
                self,
                other,
                scale_epsilon=similarity_epsilon,
                value_epsilon=similarity_epsilon,
            )
        )

        if return_components:
            return total_cost, components
        return total_cost

    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _ROI_STAT_FEATURES_INSTALLED_ATTR, True)


def _pairwise_roi_feature_components(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    scale_epsilon: float = 1.0e-6,
    value_epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Return separate pairwise Suite2p ROI-stat feature planes."""

    if scale_epsilon <= 0.0:
        raise ValueError("scale_epsilon must be strictly positive")
    if value_epsilon <= 0.0:
        raise ValueError("value_epsilon must be strictly positive")

    component_shape = (reference_plane.n_rois, measurement_plane.n_rois)
    components: dict[str, np.ndarray] = {}

    if reference_plane.n_rois == 0 or measurement_plane.n_rois == 0:
        empty = np.zeros(component_shape, dtype=float)
        for _, component_name, _ in _SPLIT_ROI_FEATURE_COMPONENT_SPECS:
            components[component_name] = empty.copy()
        components["missing_radius_indicator"] = empty.copy()
        components["missing_stat_indicator"] = empty.copy()
        return components

    missing_radius = np.zeros(component_shape, dtype=bool)
    missing_any_stat = np.zeros(component_shape, dtype=bool)

    for raw_feature_name, component_name, transform_name in (
        _SPLIT_ROI_FEATURE_COMPONENT_SPECS
    ):
        reference_values = _roi_feature_vector(reference_plane, raw_feature_name)
        measurement_values = _roi_feature_vector(measurement_plane, raw_feature_name)

        if transform_name == "log_ratio":
            component, missing = _pairwise_abs_log_ratio(
                reference_values,
                measurement_values,
                value_epsilon=value_epsilon,
            )
        elif transform_name == "scaled_absdiff":
            component, missing = _pairwise_scaled_absdiff(
                reference_values,
                measurement_values,
                scale_epsilon=scale_epsilon,
            )
        else:  # pragma: no cover - guarded by static spec table above
            raise RuntimeError(f"Unknown ROI-feature transform {transform_name!r}")

        components[component_name] = component
        missing_any_stat |= missing
        if raw_feature_name == "radius":
            missing_radius = missing

    components["missing_radius_indicator"] = missing_radius.astype(float)
    components["missing_stat_indicator"] = missing_any_stat.astype(float)
    return components


def _roi_feature_vector(plane: Any, feature_name: str) -> np.ndarray:
    """Return one scalar ROI feature per ROI, using NaN for unavailable values."""

    if feature_name not in plane.roi_features:
        return np.full((plane.n_rois,), np.nan, dtype=float)

    values = np.asarray(plane.roi_features[feature_name], dtype=float)
    values = values.reshape(plane.n_rois, -1)
    if values.shape[1] == 0:
        return np.full((plane.n_rois,), np.nan, dtype=float)
    return values[:, 0]


def _pairwise_scaled_absdiff(
    reference_values: np.ndarray,
    measurement_values: np.ndarray,
    *,
    scale_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(reference_values)[:, None] & np.isfinite(measurement_values)[
        None, :
    ]
    diff = np.zeros(valid.shape, dtype=float)
    if np.any(valid):
        scale = _roi_feature_scale(
            reference_values,
            measurement_values,
            scale_epsilon=scale_epsilon,
        )
        raw_diff = np.abs(reference_values[:, None] - measurement_values[None, :])
        diff[valid] = raw_diff[valid] / scale
    return np.nan_to_num(diff, nan=0.0, posinf=1.0e6, neginf=0.0), ~valid


def _pairwise_abs_log_ratio(
    reference_values: np.ndarray,
    measurement_values: np.ndarray,
    *,
    value_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_reference = np.isfinite(reference_values) & (reference_values > 0.0)
    valid_measurement = np.isfinite(measurement_values) & (measurement_values > 0.0)
    valid = valid_reference[:, None] & valid_measurement[None, :]
    log_ratio = np.zeros(valid.shape, dtype=float)
    if np.any(valid):
        raw_log_ratio = np.abs(
            np.log(
                np.maximum(reference_values[:, None], value_epsilon)
                / np.maximum(measurement_values[None, :], value_epsilon)
            )
        )
        log_ratio[valid] = raw_log_ratio[valid]
    return np.nan_to_num(log_ratio, nan=0.0, posinf=1.0e6, neginf=0.0), ~valid


def _roi_feature_scale(
    reference_values: np.ndarray,
    measurement_values: np.ndarray,
    *,
    scale_epsilon: float,
) -> float:
    pooled_values = np.concatenate(
        [reference_values.reshape(-1), measurement_values.reshape(-1)]
    )
    pooled_values = pooled_values[np.isfinite(pooled_values)]
    if pooled_values.size == 0:
        return 1.0
    scale = float(np.std(pooled_values))
    if not np.isfinite(scale) or scale < scale_epsilon:
        return 1.0
    return scale


def _normalize_roi_feature_names(
    feature_names: Sequence[str] | None,
) -> list[str] | None:
    """Map split-component feature names back to raw Suite2p stat names."""

    if feature_names is None:
        return None

    component_to_raw = {
        component_name: raw_feature_name
        for raw_feature_name, component_name, _ in _SPLIT_ROI_FEATURE_COMPONENT_SPECS
    }
    normalized: list[str] = []
    for feature_name in feature_names:
        raw_feature_name = component_to_raw.get(feature_name, feature_name)
        if raw_feature_name not in normalized:
            normalized.append(raw_feature_name)
    return normalized
