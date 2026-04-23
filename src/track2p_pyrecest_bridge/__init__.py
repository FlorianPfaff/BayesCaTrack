"""Standalone Track2p/Suite2p bridge for PyRecEst.

This package provides a Track2p/Suite2p bridge for PyRecEst. It focuses on the
calcium-imaging formats used by Track2p and turns them into PyRecEst-friendly state
representations without adding neuroscience-specific code to PyRecEst itself.

Supported inputs
----------------
* Suite2p folders (``suite2p/planeX``)
* Track2p raw NPY folders (``data_npy/planeX``)

Typical usage
-------------
Inspect a subject directory::

    python -m track2p_pyrecest_bridge summary /path/to/jm039 --plane plane0

Export per-session measurements and state moments::

    python -m track2p_pyrecest_bridge export /path/to/jm039 /tmp/jm039_plane0.npz

Use from Python::

    from track2p_pyrecest_bridge import load_track2p_subject

    sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0")
    filters = sessions[0].plane_data.to_pyrecest_kalman_filters()

The exported states follow the constant-velocity layout ``[pos_1, vel_1, pos_2, vel_2]``
with coordinate order controlled by ``order``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

_SESSION_NAME_PATTERN = re.compile(r"^(?P<session_date>\d{4}-\d{2}-\d{2})(?:_.+)?$")


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class CalciumPlaneData:
    """ROI-level representation of one imaging plane from one session."""

    roi_masks: np.ndarray
    traces: np.ndarray | None = None
    fov: np.ndarray | None = None
    spike_traces: np.ndarray | None = None
    neuropil_traces: np.ndarray | None = None
    cell_probabilities: np.ndarray | None = None
    roi_indices: np.ndarray | None = None
    roi_features: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "unknown"
    plane_name: str | None = None
    ops: dict[str, Any] | None = None

    # pylint: disable=too-many-branches
    def __post_init__(self) -> None:
        roi_masks = np.asarray(self.roi_masks)
        if roi_masks.ndim != 3:
            raise ValueError("roi_masks must have shape (n_roi, height, width)")
        object.__setattr__(self, "roi_masks", roi_masks)

        n_rois = roi_masks.shape[0]

        for field_name in ("traces", "spike_traces", "neuropil_traces"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = np.asarray(value)
            if value.ndim != 2:
                raise ValueError(f"{field_name} must have shape (n_roi, n_timepoints)")
            if value.shape[0] != n_rois:
                raise ValueError(
                    f"{field_name} must have first dimension equal to the number of ROIs"
                )
            object.__setattr__(self, field_name, value)

        if self.fov is not None:
            fov = np.asarray(self.fov)
            if fov.ndim != 2:
                raise ValueError("fov must have shape (height, width)")
            if fov.shape != roi_masks.shape[1:]:
                raise ValueError("fov spatial shape must match the mask spatial shape")
            object.__setattr__(self, "fov", fov)

        if self.cell_probabilities is not None:
            probabilities = np.asarray(self.cell_probabilities, dtype=float)
            if probabilities.shape != (n_rois,):
                raise ValueError("cell_probabilities must have shape (n_roi,)")
            object.__setattr__(self, "cell_probabilities", probabilities)

        if self.roi_indices is not None:
            roi_indices = np.asarray(self.roi_indices, dtype=int)
            if roi_indices.shape != (n_rois,):
                raise ValueError("roi_indices must have shape (n_roi,)")
            object.__setattr__(self, "roi_indices", roi_indices)

        sanitized_features: dict[str, np.ndarray] = {}
        for key, value in self.roi_features.items():
            array_value = np.asarray(value)
            if array_value.ndim == 0:
                raise ValueError(f"ROI feature '{key}' must be at least one-dimensional")
            if array_value.shape[0] != n_rois:
                raise ValueError(
                    f"ROI feature '{key}' must have first dimension equal to n_roi"
                )
            sanitized_features[key] = array_value
        object.__setattr__(self, "roi_features", sanitized_features)

    @property
    def n_rois(self) -> int:
        return int(self.roi_masks.shape[0])

    @property
    def image_shape(self) -> tuple[int, int]:
        return int(self.roi_masks.shape[1]), int(self.roi_masks.shape[2])

    def with_replaced_masks(
        self,
        roi_masks: np.ndarray,
        *,
        fov: np.ndarray | None = None,
        source: str | None = None,
        plane_name: str | None = None,
        ops: dict[str, Any] | None = None,
    ) -> "CalciumPlaneData":
        """Return a copy with new ROI masks but preserved per-ROI metadata.

        This is useful when ROIs from a later session have been transformed into
        the coordinate frame of an earlier session, e.g. after sequential image
        registration. The transformed plane can then be passed to
        :meth:`build_pairwise_cost_matrix` or the association-bundle helpers so
        both the association costs and measurement updates operate in a common
        coordinate frame.
        """

        roi_masks = np.asarray(roi_masks)
        if roi_masks.shape[0] != self.n_rois:
            raise ValueError(
                "roi_masks must preserve the number of ROIs when replacing masks"
            )

        return CalciumPlaneData(
            roi_masks=roi_masks,
            traces=self.traces,
            fov=self.fov if fov is None else fov,
            spike_traces=self.spike_traces,
            neuropil_traces=self.neuropil_traces,
            cell_probabilities=self.cell_probabilities,
            roi_indices=self.roi_indices,
            roi_features=self.roi_features,
            source=self.source if source is None else source,
            plane_name=self.plane_name if plane_name is None else plane_name,
            ops=self.ops if ops is None else ops,
        )

    def roi_areas(self, *, weighted: bool = False) -> np.ndarray:
        """Return per-ROI areas as a ``(n_roi,)`` vector."""

        if self.n_rois == 0:
            return np.zeros((0,), dtype=float)

        flat_masks = np.asarray(self.roi_masks, dtype=float).reshape(self.n_rois, -1)
        if weighted:
            return np.sum(flat_masks, axis=1)
        return np.sum(flat_masks > 0.0, axis=1, dtype=float)

    def pairwise_centroid_distances(
        self,
        other: "CalciumPlaneData",
        *,
        order: str = "xy",
        weighted: bool = False,
    ) -> np.ndarray:
        """Return Euclidean distances between all ROI centroids.

        The output has shape ``(self.n_rois, other.n_rois)``.
        """

        order = _validate_coordinate_order(order)
        if self.n_rois == 0 or other.n_rois == 0:
            return np.zeros((self.n_rois, other.n_rois), dtype=float)

        centroids_self = self.centroids(order=order, weighted=weighted).T
        centroids_other = other.centroids(order=order, weighted=weighted).T
        diffs = centroids_self[:, None, :] - centroids_other[None, :, :]
        return np.linalg.norm(diffs, axis=2)

    # pylint: disable=too-many-arguments,too-many-locals
    def build_pairwise_cost_matrix(
        self,
        other: "CalciumPlaneData",
        *,
        order: str = "xy",
        weighted_centroids: bool = False,
        centroid_weight: float = 1.0,
        centroid_scale: float | None = None,
        max_centroid_distance: float | None = None,
        iou_weight: float = 6.0,
        mask_cosine_weight: float = 2.0,
        area_weight: float = 0.5,
        roi_feature_weight: float = 0.25,
        feature_names: Sequence[str] | None = None,
        cell_probability_weight: float = 0.0,
        large_cost: float = 1.0e6,
        similarity_epsilon: float = 1.0e-6,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:  # pylint: disable=too-many-arguments,too-many-locals
        """Build a soft ROI-aware association cost matrix.

        This method is the critical bridge between longitudinal calcium-imaging
        data and PyRecEst multitarget trackers. Unlike the centroid-only export,
        it preserves the cues that actually separate neurons across sessions:
        registered ROI overlap, mask similarity, soma size changes, and optional
        Suite2p ROI features.

        Parameters
        ----------
        other
            The candidate measurement plane. For longitudinal tracking this
            should usually be a version of the later session transformed into
            the reference frame of ``self`` using :meth:`with_replaced_masks`.
        order, weighted_centroids
            Forwarded to centroid/covariance computations.
        centroid_weight, iou_weight, mask_cosine_weight, area_weight,
        roi_feature_weight, cell_probability_weight
            Non-negative weights of the corresponding cost terms.
        centroid_scale
            Characteristic spatial scale in pixels. If omitted, it is estimated
            from the pooled ROI areas as an equivalent-cell diameter.
        max_centroid_distance
            Optional hard gate in pixels. Pairs beyond the threshold are set to
            ``large_cost``.
        large_cost
            Finite penalty assigned to pairs excluded by the hard gate.
        similarity_epsilon
            Small positive constant preventing ``log(0)``.
        return_components
            If ``True``, also return a dictionary of intermediate matrices for
            diagnostics and ablations.
        """

        if similarity_epsilon <= 0.0:
            raise ValueError("similarity_epsilon must be strictly positive")
        if large_cost <= 0.0:
            raise ValueError("large_cost must be strictly positive")
        for weight_name, weight_value in {
            "centroid_weight": centroid_weight,
            "iou_weight": iou_weight,
            "mask_cosine_weight": mask_cosine_weight,
            "area_weight": area_weight,
            "roi_feature_weight": roi_feature_weight,
            "cell_probability_weight": cell_probability_weight,
        }.items():
            if weight_value < 0.0:
                raise ValueError(f"{weight_name} must be non-negative")

        centroid_distances = self.pairwise_centroid_distances(
            other,
            order=order,
            weighted=weighted_centroids,
        )
        centroid_scale = _estimate_default_centroid_scale(
            self,
            other,
            centroid_scale=centroid_scale,
        )
        centroid_cost = (centroid_distances / centroid_scale) ** 2

        iou_matrix = _pairwise_iou_matrix(self.roi_masks, other.roi_masks)
        iou_cost = -np.log(np.clip(iou_matrix, similarity_epsilon, 1.0))

        mask_cosine_similarity = _pairwise_mask_cosine_similarity(
            self.roi_masks,
            other.roi_masks,
            similarity_epsilon=similarity_epsilon,
        )
        mask_cosine_cost = 1.0 - np.clip(mask_cosine_similarity, 0.0, 1.0)

        areas_self = self.roi_areas(weighted=False)
        areas_other = other.roi_areas(weighted=False)
        area_ratio_cost = np.abs(
            np.log(
                np.maximum(areas_self[:, None], similarity_epsilon)
                / np.maximum(areas_other[None, :], similarity_epsilon)
            )
        )

        roi_feature_cost = _pairwise_roi_feature_distance(
            self,
            other,
            feature_names=feature_names,
        )

        if self.cell_probabilities is not None and other.cell_probabilities is not None:
            probabilities_self = np.clip(self.cell_probabilities, similarity_epsilon, 1.0)
            probabilities_other = np.clip(other.cell_probabilities, similarity_epsilon, 1.0)
            cell_probability_cost = -0.5 * (
                np.log(probabilities_self[:, None]) + np.log(probabilities_other[None, :])
            )
        else:
            cell_probability_cost = np.zeros((self.n_rois, other.n_rois), dtype=float)

        total_cost = (
            centroid_weight * centroid_cost
            + iou_weight * iou_cost
            + mask_cosine_weight * mask_cosine_cost
            + area_weight * area_ratio_cost
            + roi_feature_weight * roi_feature_cost
            + cell_probability_weight * cell_probability_cost
        )

        if max_centroid_distance is not None:
            if max_centroid_distance <= 0.0:
                raise ValueError("max_centroid_distance must be strictly positive")
            gated = centroid_distances > max_centroid_distance
            total_cost = np.where(gated, large_cost, total_cost)
        else:
            gated = np.zeros_like(total_cost, dtype=bool)

        total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)

        if not return_components:
            return total_cost

        components = {
            "pairwise_cost_matrix": total_cost,
            "centroid_distance": centroid_distances,
            "centroid_cost": centroid_cost,
            "iou": iou_matrix,
            "iou_cost": iou_cost,
            "mask_cosine_similarity": mask_cosine_similarity,
            "mask_cosine_cost": mask_cosine_cost,
            "area_ratio_cost": area_ratio_cost,
            "roi_feature_cost": roi_feature_cost,
            "cell_probability_cost": cell_probability_cost,
            "gated": gated.astype(bool),
        }
        return total_cost, components

    def centroids(self, order: str = "xy", weighted: bool = False) -> np.ndarray:
        """Return ROI centroids as a ``(2, n_roi)`` measurement matrix."""

        order = _validate_coordinate_order(order)
        if self.n_rois == 0:
            return np.zeros((2, 0), dtype=float)

        coords = np.zeros((2, self.n_rois), dtype=float)
        for roi_index, mask in enumerate(self.roi_masks):
            row_coords, col_coords = np.nonzero(mask)
            if row_coords.size == 0:
                raise ValueError(f"ROI {roi_index} has an empty mask")

            if weighted:
                weights = np.asarray(mask[row_coords, col_coords], dtype=float)
            else:
                weights = np.ones(row_coords.shape[0], dtype=float)

            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(f"ROI {roi_index} has non-positive total mask weight")

            centroid_y = float(np.dot(row_coords, weights) / weight_sum)
            centroid_x = float(np.dot(col_coords, weights) / weight_sum)

            if order == "xy":
                coords[:, roi_index] = np.array([centroid_x, centroid_y])
            else:
                coords[:, roi_index] = np.array([centroid_y, centroid_x])

        return coords

    def position_covariances(
        self,
        order: str = "xy",
        weighted: bool = False,
        regularization: float = 1e-6,
    ) -> np.ndarray:
        """Return per-ROI spatial covariance matrices with shape ``(2, 2, n_roi)``."""

        order = _validate_coordinate_order(order)
        if regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.n_rois == 0:
            return np.zeros((2, 2, 0), dtype=float)

        covariances = np.zeros((2, 2, self.n_rois), dtype=float)
        centroids = self.centroids(order=order, weighted=weighted)

        for roi_index, mask in enumerate(self.roi_masks):
            row_coords, col_coords = np.nonzero(mask)
            if weighted:
                weights = np.asarray(mask[row_coords, col_coords], dtype=float)
            else:
                weights = np.ones(row_coords.shape[0], dtype=float)

            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(f"ROI {roi_index} has non-positive total mask weight")

            if order == "xy":
                samples = np.vstack((col_coords, row_coords)).astype(float)
            else:
                samples = np.vstack((row_coords, col_coords)).astype(float)

            centered = samples - centroids[:, roi_index][:, None]
            covariance = (centered * weights[None, :]) @ centered.T / weight_sum
            if regularization > 0.0:
                covariance = covariance + regularization * np.eye(2)
            covariances[:, :, roi_index] = covariance

        return covariances

    def to_measurement_matrix(self, order: str = "xy", weighted: bool = False) -> np.ndarray:
        return self.centroids(order=order, weighted=weighted)

    def to_constant_velocity_state_moments(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed ROI positions into constant-velocity state moments.

        Returns
        -------
        means, covariances
            ``means`` has shape ``(4, n_roi)`` and ``covariances`` has shape
            ``(4, 4, n_roi)``.
        """

        if velocity_variance < 0.0:
            raise ValueError("velocity_variance must be non-negative")

        means_2d = self.centroids(order=order, weighted=weighted)
        covariances_2d = self.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )

        means = np.zeros((4, self.n_rois), dtype=float)
        covariances = np.zeros((4, 4, self.n_rois), dtype=float)

        for roi_index in range(self.n_rois):
            means[:, roi_index] = np.array(
                [means_2d[0, roi_index], 0.0, means_2d[1, roi_index], 0.0],
                dtype=float,
            )
            covariances[:, :, roi_index] = np.array(
                [
                    [covariances_2d[0, 0, roi_index], 0.0, covariances_2d[0, 1, roi_index], 0.0],
                    [0.0, velocity_variance, 0.0, 0.0],
                    [covariances_2d[1, 0, roi_index], 0.0, covariances_2d[1, 1, roi_index], 0.0],
                    [0.0, 0.0, 0.0, velocity_variance],
                ],
                dtype=float,
            )

        return means, covariances

    def to_pyrecest_gaussian_distributions(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> list[Any]:
        """Return one PyRecEst ``GaussianDistribution`` per ROI.

        Import is delayed so the script remains usable for inspection/export even when
        PyRecEst is not installed in the current environment.
        """

        try:
            from pyrecest.distributions import GaussianDistribution
        except ImportError as exc:  # pragma: no cover - exercised in real runtime only
            raise ImportError(
                "PyRecEst is required for to_pyrecest_gaussian_distributions()."
            ) from exc

        means, covariances = self.to_constant_velocity_state_moments(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        return [
            GaussianDistribution(means[:, roi_index], covariances[:, :, roi_index])
            for roi_index in range(self.n_rois)
        ]

    def to_pyrecest_kalman_filters(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> list[Any]:
        """Return one PyRecEst ``KalmanFilter`` per ROI."""

        try:
            from pyrecest.filters.kalman_filter import KalmanFilter
        except ImportError as exc:  # pragma: no cover - exercised in real runtime only
            raise ImportError(
                "PyRecEst is required for to_pyrecest_kalman_filters()."
            ) from exc

        gaussians = self.to_pyrecest_gaussian_distributions(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        return [KalmanFilter(gaussian) for gaussian in gaussians]

    def to_export_dict(
        self,
        *,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
        include_masks: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return plain NumPy arrays suitable for NPZ export."""

        means, covariances = self.to_constant_velocity_state_moments(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        export = {
            "measurements": self.to_measurement_matrix(order=order, weighted=weighted),
            "state_means": means,
            "state_covariances": covariances,
            "roi_indices": np.asarray(self.roi_indices if self.roi_indices is not None else np.arange(self.n_rois), dtype=int),
        }
        if self.traces is not None:
            export["traces"] = self.traces
        if self.spike_traces is not None:
            export["spike_traces"] = self.spike_traces
        if self.neuropil_traces is not None:
            export["neuropil_traces"] = self.neuropil_traces
        if self.cell_probabilities is not None:
            export["cell_probabilities"] = self.cell_probabilities
        if self.fov is not None:
            export["fov"] = self.fov
        if include_masks:
            export["roi_masks"] = self.roi_masks
        for key, value in self.roi_features.items():
            export[f"feature__{key}"] = value
        return export


@dataclass(frozen=True)
class Track2pSession:
    """One recording session from a Track2p-style subject directory."""

    session_dir: Path
    session_name: str
    session_date: date | None
    plane_data: CalciumPlaneData
    motion_energy: np.ndarray | None = None


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SessionAssociationBundle:
    """PyRecEst-ready association inputs for one reference/measurement pair.

    The state layout matches :meth:`CalciumPlaneData.to_constant_velocity_state_moments`,
    i.e. ``[pos_1, vel_1, pos_2, vel_2]`` and the measurement matrix is the
    corresponding linear position extractor.
    """

    reference_session_name: str
    measurement_session_name: str
    reference_state_means: np.ndarray
    reference_state_covariances: np.ndarray
    measurements: np.ndarray
    measurement_covariances: np.ndarray
    measurement_matrix: np.ndarray
    pairwise_cost_matrix: np.ndarray
    reference_roi_indices: np.ndarray
    measurement_roi_indices: np.ndarray
    pairwise_components: dict[str, np.ndarray] = field(default_factory=dict)

    def to_pyrecest_update_kwargs(self) -> dict[str, np.ndarray]:
        """Return keyword arguments for ``tracker.update_linear(...)``."""

        return {
            "measurements": self.measurements,
            "measurement_matrix": self.measurement_matrix,
            "covMatsMeas": self.measurement_covariances,
            "pairwise_cost_matrix": self.pairwise_cost_matrix,
        }

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def load_suite2p_plane(
    plane_dir: str | Path,
    *,
    include_non_cells: bool = False,
    cell_probability_threshold: float = 0.5,
    weighted_masks: bool = False,
    exclude_overlapping_pixels: bool = True,
    load_traces: bool = True,
    load_spike_traces: bool = True,
    load_neuropil_traces: bool = False,
) -> CalciumPlaneData:
    """Load one Suite2p plane folder into a :class:`CalciumPlaneData` instance."""

    if not 0.0 <= cell_probability_threshold <= 1.0:
        raise ValueError("cell_probability_threshold must be between 0 and 1")

    plane_dir = Path(plane_dir)
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    if stat.ndim != 1:
        raise ValueError("Suite2p stat.npy must be a one-dimensional object array")

    iscell_path = plane_dir / "iscell.npy"
    iscell = np.load(iscell_path, allow_pickle=True) if iscell_path.exists() else None

    ops_path = plane_dir / "ops.npy"
    ops = None
    fov = None
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        mean_image = ops.get("meanImg")
        if mean_image is not None:
            fov = np.asarray(mean_image)

    image_shape = _infer_image_shape(stat, ops)

    selected_indices: list[int] = []
    roi_masks: list[np.ndarray] = []
    cell_probabilities: list[float] = []
    feature_names = (
        "radius",
        "aspect_ratio",
        "compact",
        "footprint",
        "skew",
        "std",
        "npix",
        "npix_norm",
    )
    collected_features: dict[str, list[float]] = {name: [] for name in feature_names}

    for roi_index, roi_stat in enumerate(stat):
        keep_roi = True
        probability = np.nan
        if iscell is not None:
            probability = (
                float(iscell[roi_index, 1])
                if np.ndim(iscell) == 2 and iscell.shape[1] > 1
                else float(iscell[roi_index])
            )
            is_cell = bool(iscell[roi_index, 0]) if np.ndim(iscell) == 2 else bool(iscell[roi_index])
            if not include_non_cells:
                keep_roi = is_cell and probability >= cell_probability_threshold

        if not keep_roi:
            continue

        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        lam = np.asarray(roi_stat.get("lam", np.ones_like(ypix)), dtype=float)

        if exclude_overlapping_pixels and "overlap" in roi_stat:
            overlap = np.asarray(roi_stat["overlap"], dtype=bool)
            if overlap.shape == ypix.shape:
                valid = ~overlap
                ypix = ypix[valid]
                xpix = xpix[valid]
                lam = lam[valid]

        if ypix.size == 0:
            continue

        mask_dtype = float if weighted_masks else bool
        mask = np.zeros(image_shape, dtype=mask_dtype)
        if weighted_masks:
            mask[ypix, xpix] = lam
        else:
            mask[ypix, xpix] = True

        selected_indices.append(roi_index)
        roi_masks.append(mask)
        cell_probabilities.append(probability)
        for feature_name in feature_names:
            collected_features[feature_name].append(float(roi_stat.get(feature_name, np.nan)))

    roi_mask_array = _stack_or_empty_masks(roi_masks, image_shape, weighted_masks)
    selected_indices_array = np.asarray(selected_indices, dtype=int)
    probability_array = (
        np.asarray(cell_probabilities, dtype=float)
        if roi_masks
        else np.zeros((0,), dtype=float)
    )
    feature_arrays = {
        key: np.asarray(value, dtype=float)
        for key, value in collected_features.items()
        if value
    }

    traces = None
    if load_traces and (plane_dir / "F.npy").exists():
        traces = np.load(plane_dir / "F.npy")
        traces = traces[selected_indices_array]

    spike_traces = None
    if load_spike_traces and (plane_dir / "spks.npy").exists():
        spike_traces = np.load(plane_dir / "spks.npy")
        spike_traces = spike_traces[selected_indices_array]

    neuropil_traces = None
    if load_neuropil_traces and (plane_dir / "Fneu.npy").exists():
        neuropil_traces = np.load(plane_dir / "Fneu.npy")
        neuropil_traces = neuropil_traces[selected_indices_array]

    return CalciumPlaneData(
        roi_masks=roi_mask_array,
        traces=traces,
        fov=fov,
        spike_traces=spike_traces,
        neuropil_traces=neuropil_traces,
        cell_probabilities=probability_array,
        roi_indices=selected_indices_array,
        roi_features=feature_arrays,
        source="suite2p",
        plane_name=plane_dir.name,
        ops=ops,
    )


def load_raw_npy_plane(plane_dir: str | Path) -> CalciumPlaneData:
    """Load one Track2p ``data_npy/planeX`` folder."""

    plane_dir = Path(plane_dir)
    roi_masks = np.load(plane_dir / "rois.npy")
    traces = np.load(plane_dir / "F.npy")
    fov = np.load(plane_dir / "fov.npy")

    if roi_masks.ndim != 3:
        raise ValueError("rois.npy must have shape (n_roi, height, width)")
    if traces.ndim != 2:
        raise ValueError("F.npy must have shape (n_roi, n_timepoints)")
    if traces.shape[0] != roi_masks.shape[0]:
        raise ValueError("F.npy and rois.npy must contain the same number of ROIs")
    if fov.shape != roi_masks.shape[1:]:
        raise ValueError("fov.npy spatial shape must match rois.npy")

    return CalciumPlaneData(
        roi_masks=np.asarray(roi_masks),
        traces=np.asarray(traces),
        fov=np.asarray(fov),
        roi_indices=np.arange(roi_masks.shape[0], dtype=int),
        source="raw_npy",
        plane_name=plane_dir.name,
    )


def find_track2p_session_dirs(subject_dir: str | Path) -> list[Path]:
    """Return Track2p-style session folders sorted chronologically."""

    subject_dir = Path(subject_dir)
    candidate_dirs = [path for path in subject_dir.iterdir() if path.is_dir()]
    recognized_dirs: list[tuple[date | None, str, Path]] = []
    for candidate in candidate_dirs:
        match = _SESSION_NAME_PATTERN.match(candidate.name)
        session_date = (
            date.fromisoformat(match.group("session_date")) if match is not None else None
        )
        if session_date is None and not ((candidate / "suite2p").exists() or (candidate / "data_npy").exists()):
            continue
        recognized_dirs.append((session_date, candidate.name, candidate))

    recognized_dirs.sort(key=lambda item: (item[0] is None, item[0], item[1]))
    return [path for _, _, path in recognized_dirs]


def load_track2p_subject(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    **suite2p_kwargs: Any,
) -> list[Track2pSession]:
    """Load all sessions of one Track2p-style subject folder."""

    if input_format not in {"auto", "suite2p", "npy"}:
        raise ValueError("input_format must be 'auto', 'suite2p', or 'npy'")

    subject_dir = Path(subject_dir)
    sessions: list[Track2pSession] = []
    for session_dir in find_track2p_session_dirs(subject_dir):
        suite2p_plane_dir = session_dir / "suite2p" / plane_name
        npy_plane_dir = session_dir / "data_npy" / plane_name

        plane_data: CalciumPlaneData | None = None
        if input_format in {"auto", "suite2p"} and suite2p_plane_dir.exists():
            plane_data = load_suite2p_plane(suite2p_plane_dir, **suite2p_kwargs)
        elif input_format in {"auto", "npy"} and npy_plane_dir.exists():
            plane_data = load_raw_npy_plane(npy_plane_dir)

        if plane_data is None:
            if input_format == "auto":
                continue
            raise FileNotFoundError(
                f"Could not find {input_format} data for session '{session_dir.name}' and plane '{plane_name}'"
            )

        motion_energy = None
        if include_behavior:
            motion_energy_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if motion_energy_path.exists():
                motion_energy = np.load(motion_energy_path)

        match = _SESSION_NAME_PATTERN.match(session_dir.name)
        session_date = (
            date.fromisoformat(match.group("session_date")) if match is not None else None
        )
        sessions.append(
            Track2pSession(
                session_dir=session_dir,
                session_name=session_dir.name,
                session_date=session_date,
                plane_data=plane_data,
                motion_energy=motion_energy,
            )
        )

    return sessions

# pylint: disable=too-many-arguments,too-many-locals
def build_session_pair_association_bundle(
    reference_session: Track2pSession,
    measurement_session: Track2pSession,
    *,
    measurement_plane_in_reference_frame: CalciumPlaneData | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
) -> SessionAssociationBundle:
    """Build PyRecEst-ready inputs for one consecutive-session association step.

    Parameters
    ----------
    reference_session, measurement_session
        Consecutive sessions to be linked.
    measurement_plane_in_reference_frame
        Optional replacement for ``measurement_session.plane_data`` that has
        been transformed into the coordinate frame of ``reference_session``.
        This should generally be used for longitudinal cell tracking, because
        ROI overlap and centroid proximity only become meaningful after the
        later session is expressed in the earlier session's frame.
    pairwise_cost_kwargs
        Forwarded to :meth:`CalciumPlaneData.build_pairwise_cost_matrix`.
    """

    association_plane = (
        measurement_session.plane_data
        if measurement_plane_in_reference_frame is None
        else measurement_plane_in_reference_frame
    )

    reference_state_means, reference_state_covariances = (
        reference_session.plane_data.to_constant_velocity_state_moments(
            order=order,
            weighted=weighted_centroids,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
    )
    measurements = association_plane.to_measurement_matrix(
        order=order,
        weighted=weighted_centroids,
    )
    measurement_covariances = association_plane.position_covariances(
        order=order,
        weighted=weighted_centroids,
        regularization=regularization,
    )
    measurement_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=float,
    )

    pairwise_cost_kwargs = dict(pairwise_cost_kwargs or {})
    if return_pairwise_components:
        pairwise_cost_kwargs["return_components"] = True
        pairwise_cost_matrix, pairwise_components = (
            reference_session.plane_data.build_pairwise_cost_matrix(
                association_plane,
                order=order,
                weighted_centroids=weighted_centroids,
                **pairwise_cost_kwargs,
            )
        )
    else:
        pairwise_cost_matrix = reference_session.plane_data.build_pairwise_cost_matrix(
            association_plane,
            order=order,
            weighted_centroids=weighted_centroids,
            **pairwise_cost_kwargs,
        )
        pairwise_components = {}

    reference_roi_indices = (
        np.asarray(reference_session.plane_data.roi_indices, dtype=int)
        if reference_session.plane_data.roi_indices is not None
        else np.arange(reference_session.plane_data.n_rois, dtype=int)
    )
    measurement_roi_indices = (
        np.asarray(association_plane.roi_indices, dtype=int)
        if association_plane.roi_indices is not None
        else np.arange(association_plane.n_rois, dtype=int)
    )

    return SessionAssociationBundle(
        reference_session_name=reference_session.session_name,
        measurement_session_name=measurement_session.session_name,
        reference_state_means=reference_state_means,
        reference_state_covariances=reference_state_covariances,
        measurements=measurements,
        measurement_covariances=measurement_covariances,
        measurement_matrix=measurement_matrix,
        pairwise_cost_matrix=pairwise_cost_matrix,
        reference_roi_indices=reference_roi_indices,
        measurement_roi_indices=measurement_roi_indices,
        pairwise_components=pairwise_components,
    )

# pylint: disable=too-many-arguments
def build_consecutive_session_association_bundles(
    sessions: Sequence[Track2pSession],
    *,
    measurement_planes_in_reference_frames: Sequence[CalciumPlaneData | None] | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
) -> list[SessionAssociationBundle]:
    """Build one :class:`SessionAssociationBundle` for each consecutive session pair."""

    sessions = list(sessions)
    if len(sessions) < 2:
        return []

    if measurement_planes_in_reference_frames is None:
        measurement_planes_in_reference_frames = [None] * (len(sessions) - 1)
    elif len(measurement_planes_in_reference_frames) != len(sessions) - 1:
        raise ValueError(
            "measurement_planes_in_reference_frames must have length len(sessions) - 1"
        )

    bundles: list[SessionAssociationBundle] = []
    for pair_index, measurement_plane_in_reference_frame in enumerate(
        measurement_planes_in_reference_frames
    ):
        bundles.append(
            build_session_pair_association_bundle(
                sessions[pair_index],
                sessions[pair_index + 1],
                measurement_plane_in_reference_frame=measurement_plane_in_reference_frame,
                order=order,
                weighted_centroids=weighted_centroids,
                velocity_variance=velocity_variance,
                regularization=regularization,
                pairwise_cost_kwargs=pairwise_cost_kwargs,
                return_pairwise_components=return_pairwise_components,
            )
        )

    return bundles

# pylint: disable=too-many-arguments,too-many-locals
def export_subject_to_npz(
    subject_dir: str | Path,
    output_path: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    include_masks: bool = False,
    order: str = "xy",
    weighted: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    validate_pyrecest: bool = False,
    **suite2p_kwargs: Any,
) -> dict[str, Any]:
    """Export one subject into a single NPZ archive.

    The archive contains one block of per-session arrays keyed as
    ``session_{index}__<name>`` plus summary metadata.
    """

    sessions = load_track2p_subject(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        **suite2p_kwargs,
    )

    payload: dict[str, np.ndarray] = {
        "session_names": np.asarray([session.session_name for session in sessions], dtype=object),
        "session_dates": np.asarray([
            session.session_date.isoformat() if session.session_date is not None else ""
            for session in sessions
        ], dtype=object),
        "plane_name": np.asarray(plane_name, dtype=object),
        "input_format": np.asarray(input_format, dtype=object),
    }

    summary_sessions: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        plane_data = session.plane_data
        export = plane_data.to_export_dict(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
            include_masks=include_masks,
        )
        for key, value in export.items():
            payload[f"session_{session_index}__{key}"] = value
        if session.motion_energy is not None:
            payload[f"session_{session_index}__motion_energy"] = session.motion_energy

        if validate_pyrecest:
            # Force lazy imports and object construction without storing fragile pickles.
            _ = plane_data.to_pyrecest_gaussian_distributions(
                order=order,
                weighted=weighted,
                velocity_variance=velocity_variance,
                regularization=regularization,
            )

        summary_sessions.append(
            {
                "session_name": session.session_name,
                "session_date": session.session_date.isoformat() if session.session_date else None,
                "source": plane_data.source,
                "n_rois": plane_data.n_rois,
                "image_shape": list(plane_data.image_shape),
                "has_traces": plane_data.traces is not None,
                "has_fov": plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    return {
        "subject_dir": str(Path(subject_dir)),
        "output_path": str(output_path),
        "n_sessions": len(sessions),
        "plane_name": plane_name,
        "input_format": input_format,
        "sessions": summary_sessions,
    }


def summarize_subject(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    **suite2p_kwargs: Any,
) -> dict[str, Any]:
    """Return JSON-serializable summary for one subject."""

    sessions = load_track2p_subject(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        **suite2p_kwargs,
    )
    return {
        "subject_dir": str(Path(subject_dir)),
        "plane_name": plane_name,
        "input_format": input_format,
        "n_sessions": len(sessions),
        "sessions": [
            {
                "session_name": session.session_name,
                "session_date": session.session_date.isoformat() if session.session_date else None,
                "source": session.plane_data.source,
                "n_rois": session.plane_data.n_rois,
                "image_shape": list(session.plane_data.image_shape),
                "trace_shape": list(session.plane_data.traces.shape) if session.plane_data.traces is not None else None,
                "has_fov": session.plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
            for session in sessions
        ],
    }


def _stack_or_empty_masks(
    roi_masks: list[np.ndarray],
    image_shape: tuple[int, int],
    weighted_masks: bool,
) -> np.ndarray:
    if roi_masks:
        return np.stack(roi_masks, axis=0)
    mask_dtype = float if weighted_masks else bool
    return np.zeros((0, image_shape[0], image_shape[1]), dtype=mask_dtype)


def _infer_image_shape(stat: np.ndarray, ops: dict[str, Any] | None) -> tuple[int, int]:
    if ops is not None and "Ly" in ops and "Lx" in ops:
        return int(ops["Ly"]), int(ops["Lx"])
    if len(stat) == 0:
        raise ValueError("Cannot infer image shape from an empty stat.npy without ops.npy")
    max_y = 0
    max_x = 0
    for roi_stat in stat:
        max_y = max(max_y, int(np.max(roi_stat["ypix"])))
        max_x = max(max_x, int(np.max(roi_stat["xpix"])))
    return max_y + 1, max_x + 1


def _validate_coordinate_order(order: str) -> str:
    if order not in {"xy", "yx"}:
        raise ValueError("order must be either 'xy' or 'yx'")
    return order


def _ensure_finite_cost_matrix(cost_matrix: np.ndarray, *, large_cost: float) -> np.ndarray:
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    sanitized = np.array(cost_matrix, dtype=float, copy=True)
    invalid = ~np.isfinite(sanitized)
    if np.any(invalid):
        sanitized[invalid] = large_cost
    sanitized[sanitized < 0.0] = 0.0
    return sanitized


def _estimate_default_centroid_scale(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    centroid_scale: float | None = None,
) -> float:
    if centroid_scale is not None:
        if centroid_scale <= 0.0:
            raise ValueError("centroid_scale must be strictly positive")
        return float(centroid_scale)

    pooled_areas = np.concatenate(
        [
            reference_plane.roi_areas(weighted=False),
            measurement_plane.roi_areas(weighted=False),
        ]
    )
    pooled_areas = pooled_areas[np.isfinite(pooled_areas)]
    if pooled_areas.size == 0:
        return 1.0
    equivalent_diameter = 2.0 * np.sqrt(np.median(pooled_areas) / np.pi)
    return float(max(equivalent_diameter, 1.0))


def _pairwise_iou_matrix(reference_masks: np.ndarray, measurement_masks: np.ndarray) -> np.ndarray:
    reference_support = np.asarray(reference_masks).reshape(reference_masks.shape[0], -1) > 0
    measurement_support = np.asarray(measurement_masks).reshape(measurement_masks.shape[0], -1) > 0
    if reference_support.shape[0] == 0 or measurement_support.shape[0] == 0:
        return np.zeros((reference_support.shape[0], measurement_support.shape[0]), dtype=float)

    intersections = reference_support.astype(float) @ measurement_support.astype(float).T
    areas_reference = np.sum(reference_support, axis=1, dtype=float)
    areas_measurement = np.sum(measurement_support, axis=1, dtype=float)
    unions = areas_reference[:, None] + areas_measurement[None, :] - intersections
    iou = np.zeros_like(intersections, dtype=float)
    valid = unions > 0.0
    iou[valid] = intersections[valid] / unions[valid]
    return iou


def _pairwise_mask_cosine_similarity(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    similarity_epsilon: float,
) -> np.ndarray:
    flat_reference = np.asarray(reference_masks, dtype=float).reshape(reference_masks.shape[0], -1)
    flat_measurement = np.asarray(measurement_masks, dtype=float).reshape(measurement_masks.shape[0], -1)
    if flat_reference.shape[0] == 0 or flat_measurement.shape[0] == 0:
        return np.zeros((flat_reference.shape[0], flat_measurement.shape[0]), dtype=float)

    numerator = flat_reference @ flat_measurement.T
    denom_reference = np.linalg.norm(flat_reference, axis=1)
    denom_measurement = np.linalg.norm(flat_measurement, axis=1)
    denominator = np.maximum(
        denom_reference[:, None] * denom_measurement[None, :],
        similarity_epsilon,
    )
    return numerator / denominator

# pylint: disable=too-many-locals
def _pairwise_roi_feature_distance(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    feature_names: Sequence[str] | None = None,
    scale_epsilon: float = 1.0e-6,
) -> np.ndarray:
    if reference_plane.n_rois == 0 or measurement_plane.n_rois == 0:
        return np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)

    if feature_names is None:
        feature_names = sorted(
            set(reference_plane.roi_features).intersection(measurement_plane.roi_features)
        )
    else:
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name in reference_plane.roi_features
            and feature_name in measurement_plane.roi_features
        ]

    if not feature_names:
        return np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)

    feature_distance = np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)
    used_feature_dims = 0

    for feature_name in feature_names:
        reference_feature = np.asarray(reference_plane.roi_features[feature_name], dtype=float)
        measurement_feature = np.asarray(measurement_plane.roi_features[feature_name], dtype=float)

        reference_feature = reference_feature.reshape(reference_plane.n_rois, -1)
        measurement_feature = measurement_feature.reshape(measurement_plane.n_rois, -1)
        pooled_feature = np.concatenate(
            [reference_feature.reshape(-1), measurement_feature.reshape(-1)]
        )
        pooled_feature = pooled_feature[np.isfinite(pooled_feature)]
        if pooled_feature.size == 0:
            continue
        feature_scale = float(np.std(pooled_feature))
        if feature_scale < scale_epsilon:
            feature_scale = 1.0

        for dim_index in range(reference_feature.shape[1]):
            reference_values = reference_feature[:, dim_index]
            measurement_values = measurement_feature[:, dim_index]
            valid = np.isfinite(reference_values)[:, None] & np.isfinite(measurement_values)[None, :]
            diff = np.zeros_like(feature_distance)
            diff[valid] = (
                np.abs(reference_values[:, None] - measurement_values[None, :])[valid]
                / feature_scale
            )
            feature_distance += diff
            used_feature_dims += 1

    if used_feature_dims == 0:
        return np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)
    return feature_distance / used_feature_dims


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Track2p/Suite2p loader that exports PyRecEst-ready state moments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("subject_dir", type=Path, help="Track2p-style subject directory")
    common.add_argument("--plane", dest="plane_name", default="plane0", help="Plane subdirectory to load")
    common.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format to load",
    )
    common.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load motion_energy_glob.npy when present",
    )
    common.add_argument(
        "--include-non-cells",
        action="store_true",
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    common.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=0.5,
        help="Suite2p iscell probability threshold",
    )
    common.add_argument(
        "--weighted-masks",
        action="store_true",
        help="Reconstruct Suite2p masks using lam weights",
    )
    common.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )

    summary_parser = subparsers.add_parser("summary", parents=[common], help="Print JSON summary")
    summary_parser.set_defaults(_handler=_handle_summary)

    export_parser = subparsers.add_parser("export", parents=[common], help="Export an NPZ bundle")
    export_parser.add_argument("output_path", type=Path, help="Destination .npz file")
    export_parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Include ROI masks in the export archive",
    )
    export_parser.add_argument(
        "--order",
        default="xy",
        choices=("xy", "yx"),
        help="Coordinate order in exported measurement/state arrays",
    )
    export_parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use weighted centroids/covariances when masks contain weights",
    )
    export_parser.add_argument(
        "--velocity-variance",
        type=float,
        default=25.0,
        help="Velocity variance for the constant-velocity embedding",
    )
    export_parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Small diagonal regularization added to 2D ROI covariances",
    )
    export_parser.add_argument(
        "--validate-pyrecest",
        action="store_true",
        help="Instantiate PyRecEst GaussianDistribution objects during export",
    )
    export_parser.set_defaults(_handler=_handle_export)

    return parser


def _suite2p_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "include_non_cells": args.include_non_cells,
        "cell_probability_threshold": args.cell_probability_threshold,
        "weighted_masks": args.weighted_masks,
        "exclude_overlapping_pixels": args.exclude_overlapping_pixels,
    }


def _handle_summary(args: argparse.Namespace) -> int:
    summary = summarize_subject(
        args.subject_dir,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        **_suite2p_kwargs_from_args(args),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    summary = export_subject_to_npz(
        args.subject_dir,
        args.output_path,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        include_masks=args.include_masks,
        order=args.order,
        weighted=args.weighted,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        validate_pyrecest=args.validate_pyrecest,
        **_suite2p_kwargs_from_args(args),
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler")
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
