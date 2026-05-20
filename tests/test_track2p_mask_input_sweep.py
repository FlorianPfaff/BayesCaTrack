from __future__ import annotations

from pathlib import Path

import pytest
from bayescatrack.experiments import track2p_mask_input_sweep
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
)
from bayescatrack.experiments.track2p_mask_input_sweep import (
    MaskInputSweepConfig,
    _mask_input_settings,
    run_track2p_mask_input_sweep,
)

# pylint: disable=protected-access


def test_mask_input_settings_deduplicate_all_roi_thresholds():
    config = MaskInputSweepConfig(
        benchmark=Track2pBenchmarkConfig(
            data=Path("dataset"), method="global-assignment", progress=False
        ),
        include_non_cells=(False, True),
        cell_probability_thresholds=(0.25, 0.75),
        weighted_masks=(False, True),
        weighted_centroids=None,
        exclude_overlapping_pixels=(True,),
    )

    settings = _mask_input_settings(config)

    assert len(settings) == 6
    assert {setting.sweep_count for setting in settings} == {6}
    all_roi_settings = [setting for setting in settings if setting.include_non_cells]
    assert len(all_roi_settings) == 2
    assert {setting.cell_probability_threshold for setting in all_roi_settings} == {None}
    assert all(setting.weighted_centroids == setting.weighted_masks for setting in settings)


def test_mask_input_sweep_augments_rows_and_forwards_config(monkeypatch):
    seen_configs: list[Track2pBenchmarkConfig] = []

    def fake_run_track2p_benchmark(config: Track2pBenchmarkConfig):
        seen_configs.append(config)
        return [
            SubjectBenchmarkResult(
                subject="jm001",
                variant="Same costs + global assignment",
                method=config.method,
                scores={"pairwise_f1": 1.0, "complete_track_f1": 0.5},
                n_sessions=2,
                reference_source="ground_truth_csv",
            )
        ]

    monkeypatch.setattr(
        track2p_mask_input_sweep,
        "run_track2p_benchmark",
        fake_run_track2p_benchmark,
    )
    config = MaskInputSweepConfig(
        benchmark=Track2pBenchmarkConfig(
            data=Path("dataset"), method="global-assignment", progress=False
        ),
        include_non_cells=(False,),
        cell_probability_thresholds=(0.25,),
        weighted_masks=(True,),
        weighted_centroids=(True,),
        exclude_overlapping_pixels=(False,),
    )

    rows = [result.to_dict() for result in run_track2p_mask_input_sweep(config)]

    assert len(rows) == 1
    assert len(seen_configs) == 1
    assert seen_configs[0].include_non_cells is False
    assert seen_configs[0].cell_probability_threshold == pytest.approx(0.25)
    assert seen_configs[0].weighted_masks is True
    assert seen_configs[0].weighted_centroids is True
    assert seen_configs[0].exclude_overlapping_pixels is False
    assert rows[0]["input_variant"] == "iscell>=0.25/lam-masks/weighted-centroids/keep-overlap"
    assert rows[0]["cell_probability_threshold"] == pytest.approx(0.25)
    assert rows[0]["weighted_masks"] == "true"
