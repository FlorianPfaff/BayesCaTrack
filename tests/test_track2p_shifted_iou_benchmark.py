"""Tests for shifted-IoU benchmark sweep option handling."""

from argparse import Namespace
from pathlib import Path

import pytest

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_shifted_iou_benchmark import (
    ShiftedIouSetting,
    _shifted_pairwise_cost_kwargs,
    _sweep_settings,
)


def _args(**overrides):
    values = {
        "shifted_iou_radius": 4,
        "shifted_iou_radii": None,
        "shifted_iou_additive_weight": 0.0,
        "shifted_iou_additive_weights": None,
        "shifted_mask_cosine_weight": 0.0,
        "shifted_mask_cosine_weights": None,
        "shifted_iou_shift_penalty_weight": 0.0,
        "shifted_iou_shift_penalty_weights": None,
        "shifted_iou_shift_penalty_scale": None,
        "shifted_iou_shift_penalty_scales": None,
        "costs": None,
        "transform_types": None,
        "weighted_mask_states": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _config() -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(data=Path("."), method="global-assignment")


def test_shifted_iou_sweep_defaults_to_single_legacy_setting():
    settings = _sweep_settings(
        _args(
            shifted_iou_radius=3,
            shifted_iou_additive_weight=0.2,
            shifted_mask_cosine_weight=0.1,
            shifted_iou_shift_penalty_weight=0.25,
            shifted_iou_shift_penalty_scale=2.0,
        ),
        _config(),
    )

    assert settings == (
        ShiftedIouSetting(
            cost="registered-iou",
            radius=3,
            additive_weight=0.2,
            mask_cosine_weight=0.1,
            shift_penalty_weight=0.25,
            shift_penalty_scale=2.0,
            transform_type="affine",
            weighted_masks=False,
            sweep_index=1,
            sweep_count=1,
        ),
    )


def test_shifted_iou_sweep_builds_cartesian_product_with_indices():
    settings = _sweep_settings(
        _args(
            shifted_iou_radii="1,2",
            shifted_iou_shift_penalty_weights="0,0.25",
            shifted_iou_shift_penalty_scales="none,2",
        ),
        _config(),
    )

    assert len(settings) == 8
    assert [setting.sweep_index for setting in settings] == list(range(1, 9))
    assert {setting.sweep_count for setting in settings} == {8}
    assert {(setting.radius, setting.shift_penalty_weight) for setting in settings} == {
        (1, 0.0),
        (1, 0.25),
        (2, 0.0),
        (2, 0.25),
    }
    assert {setting.shift_penalty_scale for setting in settings} == {None, 2.0}


def test_shifted_pairwise_cost_kwargs_overrides_stale_shift_scale():
    kwargs = _shifted_pairwise_cost_kwargs(
        {"large_cost": 42.0, "shifted_iou_shift_penalty_scale": 3.0},
        radius=0,
        additive_weight=0.5,
        mask_cosine_weight=0.25,
        shift_penalty_weight=0.0,
        shift_penalty_scale=None,
    )

    assert kwargs["large_cost"] == 42.0
    assert kwargs["shifted_iou_radius"] == 0
    assert kwargs["use_shifted_iou_for_iou_cost"] is False
    assert kwargs["shifted_iou_weight"] == 0.5
    assert kwargs["shifted_mask_cosine_weight"] == 0.25
    assert "shifted_iou_shift_penalty_scale" not in kwargs


def test_shifted_iou_sweep_rejects_empty_tokens():
    with pytest.raises(ValueError, match="--shifted-iou-radii"):
        _sweep_settings(_args(shifted_iou_radii="1,,2"), _config())


def test_shifted_iou_sweep_rejects_nonpositive_explicit_scale():
    with pytest.raises(ValueError, match="--shifted-iou-shift-penalty-scales"):
        _sweep_settings(_args(shifted_iou_shift_penalty_scales="none,0"), _config())
