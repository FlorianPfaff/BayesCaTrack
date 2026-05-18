from __future__ import annotations

import pytest

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_solver_prior_tuning import (
    DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS,
    DEFAULT_SOLVER_PRIOR_END_COSTS,
    DEFAULT_SOLVER_PRIOR_GAP_PENALTIES,
    DEFAULT_SOLVER_PRIOR_START_COSTS,
    SolverPriorTuningOptions,
    _solver_prior_parameter_grid,
)


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
