from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import GlobalAssignmentRun
from bayescatrack.experiments import track2p_benchmark
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    assignment_prior_assignment_runs,
    assignment_prior_score_metadata,
    assignment_prior_settings_from_config,
    assignment_prior_variant_name,
    build_arg_parser,
)


def _global_assignment_config(**kwargs):
    return Track2pBenchmarkConfig(
        data=Path("data"),
        method="global-assignment",
        **kwargs,
    )


def test_assignment_prior_settings_form_cartesian_product():
    config = _global_assignment_config(
        start_cost=5.0,
        end_cost=7.0,
        gap_penalty=1.0,
        cost_threshold=6.0,
        sweep_start_costs=(3.0, 5.0),
        sweep_gap_penalties=(0.5, 1.0),
        sweep_cost_thresholds=(6.0, None),
    )

    settings = assignment_prior_settings_from_config(config)

    assert len(settings) == 8
    assert settings[0].start_cost == 3.0
    assert settings[0].end_cost == 7.0
    assert settings[0].gap_penalty == 0.5
    assert settings[0].cost_threshold == 6.0
    assert settings[-1].start_cost == 5.0
    assert settings[-1].cost_threshold is None


def test_assignment_prior_runs_reuse_base_assignment(monkeypatch):
    base_assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[{0: 0, 1: 0}]),
        pairwise_costs={(0, 1): np.asarray([[1.0]])},
        session_sizes=(1, 1),
        session_edges=((0, 1),),
    )
    solver_calls = []

    def fake_solve_from_pairwise_costs(
        pairwise_costs,
        *,
        session_sizes,
        session_edges=None,
        start_cost=5.0,
        end_cost=5.0,
        gap_penalty=1.0,
        cost_threshold=6.0,
    ):
        solver_calls.append((start_cost, end_cost, gap_penalty, cost_threshold))
        return GlobalAssignmentRun(
            result=SimpleNamespace(tracks=[{0: 0}]),
            pairwise_costs=dict(pairwise_costs),
            session_sizes=tuple(session_sizes),
            session_edges=tuple(session_edges or ()),
        )

    monkeypatch.setattr(
        track2p_benchmark,
        "solve_global_assignment_from_pairwise_costs",
        fake_solve_from_pairwise_costs,
    )
    config = _global_assignment_config(
        start_cost=5.0,
        sweep_start_costs=(5.0, 8.0),
    )

    runs = assignment_prior_assignment_runs(base_assignment, config)

    assert runs[0][1] is base_assignment
    assert solver_calls == [(8.0, 5.0, 1.0, 6.0)]


def test_assignment_prior_cli_parses_float_and_none_thresholds():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "data",
            "--method",
            "global-assignment",
            "--sweep-start-costs",
            "3,5",
            "--sweep-cost-thresholds",
            "6,none",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert tuple(config.sweep_start_costs) == (3.0, 5.0)
    assert tuple(config.sweep_cost_thresholds) == (6.0, None)


def test_assignment_prior_label_and_metadata_are_compact():
    config = _global_assignment_config(sweep_gap_penalties=(0.5,))
    setting = assignment_prior_settings_from_config(config)[0]

    assert assignment_prior_variant_name("variant", setting, config) == (
        "variant [start=5,end=5,gap=0.5,threshold=6]"
    )
    assert assignment_prior_score_metadata(setting) == {
        "assignment_start_cost": 5.0,
        "assignment_end_cost": 5.0,
        "assignment_gap_penalty": 0.5,
        "assignment_cost_threshold": 6.0,
    }
