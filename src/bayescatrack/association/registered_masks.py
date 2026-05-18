"""Utilities for registered ROI masks used by association costs."""

from __future__ import annotations

import numpy as np
from bayescatrack.core.bridge import CalciumPlaneData


def replace_empty_registered_masks(
    plane: CalciumPlaneData,
) -> tuple[CalciumPlaneData, np.ndarray]:
    roi_masks = np.asarray(plane.roi_masks)
    nonzero_mask = np.any(roi_masks != 0, axis=(1, 2))
    empty_registered_rois = ~nonzero_mask
    if not np.any(empty_registered_rois):
        return plane, empty_registered_rois

    replacement_masks = np.array(roi_masks, copy=True)
    fill_value = True if replacement_masks.dtype == np.bool_ else 1
    flat_masks = replacement_masks.reshape(replacement_masks.shape[0], -1)
    occupied_pixels = np.any(flat_masks != 0, axis=0)
    available_pixels = np.flatnonzero(~occupied_pixels)
    empty_count = int(np.count_nonzero(empty_registered_rois))
    if available_pixels.size == 0:
        available_pixels = np.arange(flat_masks.shape[1], dtype=int)
    if available_pixels.size < empty_count:
        available_pixels = np.resize(available_pixels, empty_count)
    else:
        available_pixels = available_pixels[:empty_count]
    for roi_index, pixel_index in zip(
        np.flatnonzero(empty_registered_rois),
        available_pixels,
        strict=False,
    ):
        flat_masks[roi_index, pixel_index] = fill_value
    return (
        plane.with_replaced_masks(
            replacement_masks,
            fov=plane.fov,
            source=plane.source,
            ops=plane.ops,
        ),
        empty_registered_rois,
    )
