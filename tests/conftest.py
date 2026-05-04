from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.testing as npt
import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _make_track2p_session(
    session_name: str,
    roi_masks: np.ndarray,
    *,
    fov: np.ndarray | None = None,
) -> Any:
    from bayescatrack import CalciumPlaneData, Track2pSession

    roi_masks = np.asarray(roi_masks)
    fov_array = (
        np.ones(roi_masks.shape[1:], dtype=float)
        if fov is None
        else np.asarray(fov, dtype=float)
    )
    return Track2pSession(
        session_dir=Path(session_name),
        session_name=session_name,
        session_date=date.fromisoformat(session_name.split("_")[0]),
        plane_data=CalciumPlaneData(
            roi_masks=roi_masks,
            fov=fov_array,
            roi_indices=np.arange(roi_masks.shape[0], dtype=int),
            source="raw_npy",
            plane_name="plane0",
        ),
    )


@pytest.fixture
def make_track2p_session() -> Callable[..., Any]:
    return _make_track2p_session


def _assert_diagonal_association(
    association_bundle: Any,
    *,
    iou_atol: float = 0.0,
) -> None:
    pairwise_cost_matrix = association_bundle.pairwise_cost_matrix
    assert pairwise_cost_matrix.shape == (2, 2)
    assert pairwise_cost_matrix[0, 0] < pairwise_cost_matrix[0, 1]
    assert pairwise_cost_matrix[1, 1] < pairwise_cost_matrix[1, 0]
    npt.assert_allclose(
        np.diag(association_bundle.pairwise_components["iou"]),
        np.ones(2),
        atol=iou_atol,
    )


@pytest.fixture
def assert_diagonal_association() -> Callable[..., None]:
    return _assert_diagonal_association


@pytest.fixture
def registered_pair_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_masks = np.zeros((2, 4, 6), dtype=bool)
    ref_masks[0, 0:2, 0:2] = True
    ref_masks[1, 1:3, 3:5] = True

    mov_masks_raw = np.zeros_like(ref_masks)
    mov_masks_raw[0, 0:2, 1:3] = True
    mov_masks_raw[1, 1:3, 4:6] = True

    mov_masks_registered = np.zeros_like(ref_masks)
    mov_masks_registered[0, 0:2, 0:2] = True
    mov_masks_registered[1, 1:3, 3:5] = True

    return ref_masks, mov_masks_raw, mov_masks_registered


@pytest.fixture
def write_raw_npy_session():
    def _write_raw_npy_session(
        subject_dir: Path,
        session_name: str,
        roi_masks: np.ndarray,
        *,
        offset: float,
        plane_name: str = "plane0",
    ) -> Path:
        plane_dir = subject_dir / session_name / "data_npy" / plane_name
        plane_dir.mkdir(parents=True, exist_ok=True)
        np.save(plane_dir / "rois.npy", roi_masks)
        np.save(
            plane_dir / "F.npy",
            np.array([[offset, offset + 1], [offset + 2, offset + 3]], dtype=float),
        )
        np.save(
            plane_dir / "fov.npy", np.full(roi_masks.shape[1:], offset, dtype=float)
        )
        return plane_dir

    return _write_raw_npy_session
