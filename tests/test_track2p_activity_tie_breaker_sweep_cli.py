from __future__ import annotations

from bayescatrack.experiments import track2p_activity_tie_breaker_sweep

# pylint: disable=protected-access


def test_activity_tie_breaker_sweep_cli_builds_config():
    args = track2p_activity_tie_breaker_sweep.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--activity-tie-breaker-weights",
            "0,0.03,0.1",
            "--activity-tie-breaker-component",
            "spike_similarity_cost",
            "--activity-trace-source",
            "spike_traces",
            "--activity-event-threshold",
            "0.2",
            "--no-cost-threshold",
            "--no-progress",
        ]
    )

    config = track2p_activity_tie_breaker_sweep._config_from_args(args)

    assert config.benchmark.method == "global-assignment"
    assert config.benchmark.cost_threshold is None
    assert config.benchmark.progress is False
    assert config.activity_tie_breaker_weights == (0.0, 0.03, 0.1)
    assert config.activity_tie_breaker_component == "spike_similarity_cost"
    assert config.activity_trace_source == "spike_traces"
    assert config.activity_event_threshold == 0.2


def test_activity_tie_breaker_sweep_table_includes_activity_columns():
    table = track2p_activity_tie_breaker_sweep.format_activity_sweep_table(
        [
            {
                "subject": "jm038",
                "activity_tie_breaker_weight": 0.03,
                "activity_tie_breaker_component": "activity_tiebreaker_cost",
                "activity_trace_source": "auto",
                "cost_threshold": 6.0,
                "pairwise_f1": 0.5,
                "complete_track_f1": 0.25,
                "pairwise_precision": 0.6,
                "pairwise_recall": 0.4,
                "cost_median": 1.2,
                "cost_p90": 3.4,
                "cost_threshold_admitted_fraction": 0.7,
            }
        ]
    )

    assert "activity_tie_breaker_weight" in table
    assert "activity_tiebreaker_cost" in table
    assert "0.030" in table
