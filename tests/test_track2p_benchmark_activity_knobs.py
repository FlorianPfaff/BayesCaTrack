from __future__ import annotations

# pylint: disable=protected-access

from pathlib import Path

from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
)
from bayescatrack.experiments import track2p_benchmark as benchmark
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def test_track2p_benchmark_cli_parses_activity_and_higher_order_knobs(tmp_path):
    args = benchmark.build_arg_parser().parse_args(
        [
            "--data",
            str(tmp_path / "data"),
            "--method",
            "global-assignment",
            "--activity-tie-breaker-weight",
            "0.125",
            "--activity-tie-breaker-component",
            "spike_similarity_cost",
            "--activity-trace-source",
            "spike_traces",
            "--activity-event-threshold",
            "0.2",
            "--higher-order-triplet-weight",
            "0.4",
            "--higher-order-support-top-k",
            "5",
            "--higher-order-support-cost-cap",
            "3.5",
            "--higher-order-max-penalty",
            "1.25",
            "--higher-order-large-cost",
            "12345",
        ]
    )

    config = benchmark._config_from_args(args)

    assert config.activity_tie_breaker_weight == 0.125
    assert config.activity_tie_breaker_component == "spike_similarity_cost"
    assert config.activity_trace_source == "spike_traces"
    assert config.activity_event_threshold == 0.2
    assert config.higher_order_triplet_weight == 0.4
    assert config.higher_order_support_top_k == 5
    assert config.higher_order_support_cost_cap == 3.5
    assert config.higher_order_max_penalty == 1.25
    assert config.higher_order_large_cost == 12345.0


def test_solve_configured_global_assignment_forwards_activity_and_higher_order(
    monkeypatch,
    tmp_path,
):
    captured_kwargs = {}

    def fake_solver(_sessions, **kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "solve_global_assignment_for_sessions", fake_solver)
    config = benchmark.Track2pBenchmarkConfig(
        data=tmp_path,
        method="global-assignment",
        activity_tie_breaker_weight=0.05,
        activity_tie_breaker_component="fluorescence_similarity_cost",
        activity_trace_source="traces",
        activity_event_threshold=0.3,
        higher_order_triplet_weight=0.6,
        higher_order_support_top_k=4,
        higher_order_support_cost_cap=2.5,
        higher_order_max_penalty=1.5,
        higher_order_large_cost=99.0,
    )

    result = benchmark.solve_configured_global_assignment([], config)

    assert result is not None
    assert captured_kwargs["activity_tie_breaker_weight"] == 0.05
    assert (
        captured_kwargs["activity_tie_breaker_component"]
        == "fluorescence_similarity_cost"
    )
    assert captured_kwargs["activity_trace_source"] == "traces"
    assert captured_kwargs["activity_event_threshold"] == 0.3
    higher_order_config = captured_kwargs["higher_order_consistency_config"]
    assert isinstance(higher_order_config, HigherOrderConsistencyConfig)
    assert higher_order_config.triplet_weight == 0.6
    assert higher_order_config.support_top_k == 4
    assert higher_order_config.support_cost_cap == 2.5
    assert higher_order_config.max_penalty == 1.5
    assert higher_order_config.large_cost == 99.0


def test_manifest_accepts_activity_and_higher_order_knobs(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    manifest_path.write_text(
        """
        {
          "defaults": {
            "data": "data/jm_manifest",
            "method": "global-assignment",
            "activity_tie_breaker_weight": 0.075,
            "higher_order_triplet_weight": 0.25
          },
          "runs": [
            {
              "name": "activity-higher-order",
              "output": "results/activity.csv"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    manifest = load_benchmark_manifest(manifest_path)
    config = manifest.runs[0].config

    assert config.data == Path(tmp_path / "data" / "jm_manifest")
    assert config.activity_tie_breaker_weight == 0.075
    assert config.higher_order_triplet_weight == 0.25
