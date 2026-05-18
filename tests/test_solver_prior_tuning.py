from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from bayescatrack.experiments import solver_prior_tuning as tuning
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_solver_prior_tuning import (
    DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS,
    DEFAULT_SOLVER_PRIOR_END_COSTS,
    DEFAULT_SOLVER_PRIOR_GAP_PENALTIES,
    DEFAULT_SOLVER_PRIOR_START_COSTS,
    SolverPriorTuningOptions,
    _solver_prior_parameter_grid,
)


@dataclass(frozen=True)
class _Plane:
    n_rois: int = 1


@dataclass(frozen=True)
class _Session:
    plane_data: _Plane = _Plane()


@dataclass(frozen=True)
class _Subject:
    subject_name: str
    sessions: tuple[Any, ...]
    reference: Any


def _config(**kwargs):
    return Track2pBenchmarkConfig(
        data="/tmp/track2p",
        method="global-assignment",
        split="leave-one-subject-out",
        cost="calibrated",
        **kwargs,
    )


def test_solver_prior_grid_includes_current_config_values():
    config = _config(
        start_cost=1.25,
        end_cost=1.5,
        gap_penalty=0.3,
        cost_threshold=2.5,
    )

    grid = _solver_prior_parameter_grid(config, options=SolverPriorTuningOptions())
    starts = {candidate.start_cost for candidate in grid}
    ends = {candidate.end_cost for candidate in grid}
    gaps = {candidate.gap_penalty for candidate in grid}
    thresholds = {candidate.cost_threshold for candidate in grid}

    assert set(DEFAULT_SOLVER_PRIOR_START_COSTS) <= starts
    assert set(DEFAULT_SOLVER_PRIOR_END_COSTS) <= ends
    assert set(DEFAULT_SOLVER_PRIOR_GAP_PENALTIES) <= gaps
    assert set(DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS) <= thresholds
    assert 1.25 in starts
    assert 1.5 in ends
    assert 0.3 in gaps
    assert 2.5 in thresholds


def test_solver_prior_grid_uses_explicit_single_candidate():
    config = _config()
    grid = _solver_prior_parameter_grid(
        config,
        options=SolverPriorTuningOptions(
            objective="complete_track_f1",
            start_costs=(0.75,),
            end_costs=(0.8,),
            gap_penalties=(0.4,),
            cost_thresholds=(None,),
        ),
    )

    assert len(grid) == 1
    candidate = grid[0]
    assert candidate.start_cost == pytest.approx(0.75)
    assert candidate.end_cost == pytest.approx(0.8)
    assert candidate.gap_penalty == pytest.approx(0.4)
    assert candidate.cost_threshold is None


def test_solver_prior_grid_rejects_invalid_values():
    config = _config()

    with pytest.raises(ValueError, match="start costs"):
        _solver_prior_parameter_grid(
            config,
            options=SolverPriorTuningOptions(start_costs=(0.0,)),
        )

    with pytest.raises(ValueError, match="gap penalties"):
        _solver_prior_parameter_grid(
            config,
            options=SolverPriorTuningOptions(gap_penalties=(-0.1,)),
        )

    with pytest.raises(ValueError, match="threshold"):
        _solver_prior_parameter_grid(
            config,
            options=SolverPriorTuningOptions(cost_thresholds=(-1.0,)),
        )


def test_tune_solver_priors_selects_best_training_candidate(monkeypatch):
    config = Track2pBenchmarkConfig(
        data="missing",
        method="global-assignment",
        cost="registered-iou",
        progress=False,
    )
    subjects = (_Subject("train", (_Session(),), object()),)
    search = tuning.SolverPriorSearchConfig(
        start_costs=(1.0, 2.0),
        end_costs=(1.0,),
        gap_penalties=(0.0,),
        cost_thresholds=(None,),
        objective="pairwise_f1",
    )

    monkeypatch.setattr(
        tuning,
        "build_registered_pairwise_costs",
        lambda *_args, **_kwargs: {(0, 1): np.asarray([[0.0]], dtype=float)},
    )

    class FakeSolverResult:
        def __init__(self, start_cost):
            self.tracks = [start_cost]

    def fake_solver(_pairwise_costs, **kwargs):
        return FakeSolverResult(kwargs["start_cost"])

    monkeypatch.setattr(tuning, "_load_pyrecest_multisession_solver", lambda: fake_solver)
    monkeypatch.setattr(
        tuning,
        "tracks_to_suite2p_index_matrix",
        lambda tracks, _sessions: np.asarray([[float(tracks[0])]]),
    )

    def fake_score(predicted, _reference, *, config):
        del config
        success = float(np.asarray(predicted)[0, 0]) == 2.0
        if success:
            return {
                "pairwise_true_positives": 1,
                "pairwise_false_positives": 0,
                "pairwise_false_negatives": 0,
                "complete_track_true_positives": 1,
                "complete_track_false_positives": 0,
                "complete_track_false_negatives": 0,
            }
        return {
            "pairwise_true_positives": 0,
            "pairwise_false_positives": 1,
            "pairwise_false_negatives": 1,
            "complete_track_true_positives": 0,
            "complete_track_false_positives": 1,
            "complete_track_false_negatives": 1,
        }

    monkeypatch.setattr(tuning, "_score_prediction_against_reference", fake_score)

    result = tuning.tune_solver_priors(subjects, config=config, search=search)

    assert result.best_candidate.start_cost == pytest.approx(2.0)
    assert result.best_candidate.end_cost == pytest.approx(1.0)
    assert result.evaluated_candidates == 2
    assert result.best_scores["pairwise_f1"] == pytest.approx(1.0)
    assert result.config_with_best_priors(config).start_cost == pytest.approx(2.0)
    assert result.score_fields()["learned_cost_threshold"] == "none"


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "1,"])
def test_positive_list_parser_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        tuning.parse_positive_list(raw, name="--start-costs")


def test_threshold_parser_accepts_none():
    assert tuning.parse_threshold_list("none,2") == (None, 2.0)
