from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from bayescatrack.experiments import track2p_benchmark
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    format_benchmark_table,
    run_track2p_benchmark,
)


def _write_subject(subject_dir, write_raw_npy_session, *, write_reference=True):
    masks_a = np.zeros((2, 4, 4), dtype=bool)
    masks_a[0, 0:2, 0:2] = True
    masks_a[1, 2:4, 2:4] = True
    masks_b = masks_a.copy()
    masks_c = masks_a.copy()

    write_raw_npy_session(subject_dir, "2024-05-01_a", masks_a, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks_b, offset=10.0)
    write_raw_npy_session(subject_dir, "2024-05-03_a", masks_c, offset=20.0)

    if not write_reference:
        return

    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir()
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.array(
                [
                    str(subject_dir / "2024-05-01_a"),
                    str(subject_dir / "2024-05-02_a"),
                    str(subject_dir / "2024-05-03_a"),
                ],
                dtype=object,
            ),
            "vector_curation_plane_0": np.array([1.0, 1.0]),
        },
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        np.array([[0, 0, 0], [1, 1, 1]], dtype=object),
        allow_pickle=True,
    )


def _write_ground_truth_csv(
    subject_dir: Path, session_names: tuple[str, ...], rows: tuple[tuple[int, ...], ...]
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def _write_suite2p_session(
    subject_dir: Path, session_name: str, *, iscell: np.ndarray
) -> Path:
    plane_dir = subject_dir / session_name / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = np.array(
        [
            {
                "ypix": np.array([0, 0]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
            {
                "ypix": np.array([1, 1]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
            {
                "ypix": np.array([2, 2]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            },
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 4, "Lx": 4, "meanImg": np.zeros((4, 4), dtype=float)},
        allow_pickle=True,
    )
    np.save(plane_dir / "F.npy", np.arange(6, dtype=float).reshape(3, 2))
    return plane_dir


def _install_fake_multisession_assignment(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_assignment = types.ModuleType("pyrecest.utils.multisession_assignment")

    class Result:
        def __init__(self):
            self.tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 2: 1}]
            self.matched_edges = []
            self.total_cost = 0.0

    def solve_multisession_assignment(pairwise_costs, **kwargs):
        assert (0, 1) in pairwise_costs
        assert (0, 2) in pairwise_costs
        assert kwargs["session_sizes"] == (2, 2, 2)
        assert kwargs["gap_penalty"] == pytest.approx(1.0)
        return Result()

    fake_assignment.solve_multisession_assignment = solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "pyrecest.utils.multisession_assignment", fake_assignment
    )


class _FakeLosoResult:
    def __init__(self, variant: str):
        self.variant = variant

    def to_benchmark_results(self):
        return [
            SubjectBenchmarkResult(
                subject="heldout",
                variant=self.variant,
                method="global-assignment",
                scores={
                    "pairwise_f1": 1.0,
                    "complete_track_f1": 1.0,
                },
                n_sessions=2,
                reference_source="ground_truth_csv",
            )
        ]


def test_track2p_baseline_benchmark_scores_track2p_output_only_as_smoke_test(
    tmp_path, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session)

    with pytest.raises(ValueError, match="not independent manual ground truth"):
        run_track2p_benchmark(
            Track2pBenchmarkConfig(data=tmp_path, method="track2p-baseline")
        )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="track2p-baseline",
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Track2p default"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
    assert result["complete_tracks"] == 2
    assert "Track2p default" in format_benchmark_table([result])


def test_track2p_baseline_benchmark_scores_aligned_rows_without_track2p_output(
    tmp_path, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="track2p-baseline",
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    result = rows[0].to_dict()
    assert result["variant"] == "Track2p default"
    assert result["reference_source"] == "aligned_subject_rows"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)


def test_benchmark_uses_ground_truth_csv_reference(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        ((0, 0, 0), (1, 1, 1)),
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(data=tmp_path, method="track2p-baseline")
    )

    result = rows[0].to_dict()
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)


def test_track2p_benchmark_cli_can_emit_edge_ranking_diagnostics(
    tmp_path, monkeypatch, capsys
):
    seen = {}

    def fake_run_track2p_benchmark(config):
        seen["benchmark_config"] = config
        return []

    def fake_run_track2p_edge_ranking(
        config,
        output_path,
        *,
        summary_output_path=None,
        feature_names=(),
        similarity_features=(),
    ):
        seen["edge_config"] = config
        seen["edge_output_path"] = output_path
        seen["edge_summary_output_path"] = summary_output_path
        seen["edge_feature_names"] = tuple(feature_names)
        seen["edge_similarity_features"] = tuple(similarity_features)
        return 7, 3

    from bayescatrack.experiments import track2p_edge_ranking

    monkeypatch.setattr(
        track2p_benchmark, "run_track2p_benchmark", fake_run_track2p_benchmark
    )
    monkeypatch.setattr(
        track2p_edge_ranking,
        "run_track2p_edge_ranking",
        fake_run_track2p_edge_ranking,
    )

    edge_output = tmp_path / "edge_ranking.csv"
    summary_output = tmp_path / "edge_ranking_summary.csv"
    status = track2p_benchmark.main(
        [
            "--data",
            str(tmp_path),
            "--method",
            "global-assignment",
            "--edge-ranking-output",
            str(edge_output),
            "--edge-ranking-summary-output",
            str(summary_output),
            "--edge-ranking-feature",
            "pairwise_cost_matrix",
            "--edge-ranking-feature",
            "iou",
            "--edge-ranking-similarity-feature",
            "iou",
            "--no-progress",
        ]
    )

    assert status == 0
    assert seen["edge_config"] is seen["benchmark_config"]
    assert seen["edge_output_path"] == edge_output
    assert seen["edge_summary_output_path"] == summary_output
    assert seen["edge_feature_names"] == ("pairwise_cost_matrix", "iou")
    assert seen["edge_similarity_features"] == ("iou",)
    assert "Wrote 7 edge-ranking rows" in capsys.readouterr().err


def test_oracle_gt_links_reconstructs_complete_tracks(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm006"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        ((0, 1, 0), (1, 0, 1)),
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="oracle-gt-links",
            reference_kind="manual-gt",
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Oracle GT consecutive links"
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
    assert result["complete_tracks"] == 2


def test_oracle_gt_links_honors_nonzero_seed_session(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm007"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        (
            (0, 1, 0),
            (1, 0, 1),
            (0, -1, 1),
        ),
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="oracle-gt-links",
            reference_kind="manual-gt",
            seed_session=1,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["reference_seed_rois"] == 2
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
    assert result["complete_tracks"] == 2


def test_benchmark_recomputes_f1_from_counts_when_no_links_match(
    tmp_path, write_raw_npy_session
):
    subject_dir = tmp_path / "jm005"
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    write_raw_npy_session(subject_dir, "2024-05-01_a", masks, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks, offset=10.0)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((1, 0),),
    )

    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir()
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.array(
                [
                    str(subject_dir / "2024-05-01_a"),
                    str(subject_dir / "2024-05-02_a"),
                ],
                dtype=object,
            ),
            "vector_curation_plane_0": np.array([1.0]),
        },
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        np.array([[0, 1]], dtype=object),
        allow_pickle=True,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="track2p-baseline",
            restrict_to_reference_seed_rois=False,
        )
    )

    result = rows[0].to_dict()
    assert result["pairwise_true_positives"] == 0
    assert result["pairwise_false_positives"] == 1
    assert result["pairwise_false_negatives"] == 1
    assert result["pairwise_f1"] == pytest.approx(0.0)
    assert result["complete_track_f1"] == pytest.approx(0.0)


def test_ground_truth_csv_validation_catches_filtered_stat_rows(tmp_path):
    subject_dir = tmp_path / "jm003"
    iscell = np.array([[1.0, 0.95], [0.0, 0.1], [1.0, 0.9]], dtype=float)
    _write_suite2p_session(subject_dir, "2024-05-01_a", iscell=iscell)
    _write_suite2p_session(subject_dir, "2024-05-02_a", iscell=iscell)
    _write_ground_truth_csv(
        subject_dir, ("2024-05-01_a", "2024-05-02_a"), ((0, 0), (1, 1))
    )

    config = Track2pBenchmarkConfig(
        data=subject_dir, method="track2p-baseline", input_format="suite2p"
    )
    with pytest.raises(ValueError, match="--include-non-cells"):
        run_track2p_benchmark(config)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="track2p-baseline",
            input_format="suite2p",
            include_non_cells=True,
        )
    )

    result = rows[0].to_dict()
    assert result["reference_source"] == "ground_truth_csv"
    assert result["pairwise_recall"] == pytest.approx(1.0)
    assert result["pairwise_precision"] == pytest.approx(1.0)
    assert result["dropped_prediction_tracks"] == 1


def test_ground_truth_scoring_filters_predictions_to_reference_seed_rois(tmp_path):
    subject_dir = tmp_path / "jm004"
    iscell = np.ones((3, 2), dtype=float)
    _write_suite2p_session(subject_dir, "2024-05-01_a", iscell=iscell)
    _write_suite2p_session(subject_dir, "2024-05-02_a", iscell=iscell)
    _write_ground_truth_csv(
        subject_dir, ("2024-05-01_a", "2024-05-02_a"), ((0, 0), (1, 1))
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="track2p-baseline",
            input_format="suite2p",
            include_non_cells=True,
        )
    )

    result = rows[0].to_dict()
    assert result["reference_seed_rois"] == 2
    assert result["evaluated_prediction_tracks"] == 2
    assert result["dropped_prediction_tracks"] == 1
    assert result["pairwise_precision"] == pytest.approx(1.0)


def test_global_assignment_benchmark_uses_skip_edges(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm002"
    _write_subject(subject_dir, write_raw_npy_session)
    _install_fake_multisession_assignment(monkeypatch)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(
        global_assignment,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="global-assignment",
            cost="registered-iou",
            max_gap=2,
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Same costs + global assignment"
    assert result["pairwise_f1"] == pytest.approx(2 / 3)
    assert result["complete_track_f1"] == pytest.approx(2 / 3)
    assert result["complete_tracks"] == 1


def test_loso_benchmark_dispatches_noncalibrated_solver_prior_tuning(monkeypatch):
    from bayescatrack.experiments import solver_prior_tuning

    captured = {}

    def fake_run_track2p_loso_solver_priors(config, *, search):
        captured["config"] = config
        captured["search"] = search
        return _FakeLosoResult("noncalibrated priors")

    monkeypatch.setattr(
        solver_prior_tuning,
        "run_track2p_loso_solver_priors",
        fake_run_track2p_loso_solver_priors,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=Path("unused"),
            method="global-assignment",
            split="leave-one-subject-out",
            cost="registered-iou",
            tune_solver_priors=True,
            solver_prior_objective="pairwise_f1",
            solver_prior_start_costs=(0.75,),
            solver_prior_end_costs=(0.8,),
            solver_prior_gap_penalties=(0.2,),
            solver_prior_cost_thresholds=(None,),
        )
    )

    assert rows[0].variant == "noncalibrated priors"
    assert captured["config"].cost == "registered-iou"
    assert captured["search"].objective == "pairwise_f1"
    assert captured["search"].start_costs == (0.75,)
    assert captured["search"].end_costs == (0.8,)
    assert captured["search"].gap_penalties == (0.2,)
    assert captured["search"].cost_thresholds == (None,)


def test_track2p_benchmark_parser_accepts_item6_ablation_knobs():
    from bayescatrack.experiments.track2p_benchmark import (
        _config_from_args,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(
        [
            "--data",
            "/tmp/track2p",
            "--method",
            "global-assignment",
            "--cost",
            "registered-soft-iou",
            "--activity-tie-breaker-weight",
            "0.03",
            "--activity-tie-breaker-component",
            "spike_similarity_cost",
            "--activity-trace-source",
            "spike_traces",
            "--activity-event-threshold",
            "0.2",
            "--higher-order-consistency-json",
            '{"triplet_weight": 0.4, "support_top_k": 6}',
        ]
    )

    config = _config_from_args(args)

    assert config.cost == "registered-soft-iou"
    assert config.activity_tie_breaker_weight == pytest.approx(0.03)
    assert config.activity_tie_breaker_component == "spike_similarity_cost"
    assert config.activity_trace_source == "spike_traces"
    assert config.activity_event_threshold == pytest.approx(0.2)
    assert config.higher_order_consistency_config == {
        "triplet_weight": 0.4,
        "support_top_k": 6,
    }


def test_solve_configured_global_assignment_passes_item6_knobs(monkeypatch):
    from bayescatrack.experiments import track2p_benchmark as benchmark

    captured = {}

    def _fake_solver(sessions, **kwargs):
        captured["sessions"] = sessions
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        benchmark, "solve_global_assignment_for_sessions", _fake_solver
    )
    config = Track2pBenchmarkConfig(
        data=Path("/tmp/track2p"),
        method="global-assignment",
        cost="registered-soft-iou",
        activity_tie_breaker_weight=0.03,
        activity_tie_breaker_component="spike_similarity_cost",
        activity_trace_source="spike_traces",
        activity_event_threshold=0.2,
        higher_order_consistency_config={
            "triplet_weight": 0.4,
            "support_top_k": 6,
        },
    )

    result = benchmark.solve_configured_global_assignment([], config)

    assert result is not None
    assert captured["sessions"] == []
    assert captured["cost"] == "registered-soft-iou"
    assert captured["activity_tie_breaker_weight"] == pytest.approx(0.03)
    assert captured["activity_tie_breaker_component"] == "spike_similarity_cost"
    assert captured["activity_trace_source"] == "spike_traces"
    assert captured["activity_event_threshold"] == pytest.approx(0.2)
    assert captured["higher_order_consistency_config"] == {
        "triplet_weight": 0.4,
        "support_top_k": 6,
    }


def test_configured_global_assignment_forwards_activity_tiebreaker(monkeypatch):
    from bayescatrack.experiments import track2p_benchmark as benchmark_module

    captured = {}
    sentinel = object()

    def fake_solve_global_assignment_for_sessions(sessions, **kwargs):
        captured["sessions"] = sessions
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        benchmark_module,
        "solve_global_assignment_for_sessions",
        fake_solve_global_assignment_for_sessions,
    )
    sessions = [object()]
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="global-assignment",
        activity_tie_breaker_weight=0.03,
        activity_tie_breaker_component="fluorescence_similarity_cost",
        activity_trace_source="traces",
        activity_event_threshold=0.2,
    )

    result = benchmark_module.solve_configured_global_assignment(sessions, config)
    assert result is sentinel

    assert captured["sessions"] is sessions
    assert captured["activity_tie_breaker_weight"] == pytest.approx(0.03)
    assert captured["activity_tie_breaker_component"] == "fluorescence_similarity_cost"


def test_global_assignment_benchmark_runs_transform_suite(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm008"
    _write_subject(subject_dir, write_raw_npy_session)
    _install_fake_multisession_assignment(monkeypatch)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    used_transforms: list[str] = []

    def fake_register_plane_pair(_reference, moving, **kwargs):
        used_transforms.append(kwargs["transform_type"])
        return moving

    monkeypatch.setattr(
        global_assignment,
        "register_plane_pair",
        fake_register_plane_pair,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="global-assignment",
            cost="registered-iou",
            max_gap=2,
            transform_suite=("fov-translation", "fov-affine"),
            allow_track2p_as_reference_for_smoke_test=True,
        )
    )

    assert len(rows) == 2
    results = [row.to_dict() for row in rows]
    assert [result["transform_type"] for result in results] == [
        "fov-translation",
        "fov-affine",
    ]
    assert [result["variant"] for result in results] == [
        "Same costs + global assignment [fov-translation]",
        "Same costs + global assignment [fov-affine]",
    ]
    assert {"fov-translation", "fov-affine"}.issubset(set(used_transforms))
    assert all(result["pairwise_f1"] == pytest.approx(2 / 3) for result in results)
    assert all(
        result["complete_track_f1"] == pytest.approx(2 / 3) for result in results
    )
    assert captured["activity_trace_source"] == "traces"
    assert captured["activity_event_threshold"] == pytest.approx(0.2)
    expected_variant = (
        "Same costs + global assignment + activity tie-breaker 0.03 "
        "(fluorescence_similarity_cost)"
    )
    assert benchmark_module._configured_variant_name(config) == expected_variant


def test_activity_tiebreaker_rejects_non_global_assignment_method():
    with pytest.raises(ValueError, match="method='global-assignment'"):
        run_track2p_benchmark(
            Track2pBenchmarkConfig(
                data=Path("."),
                method="track2p-baseline",
                activity_tie_breaker_weight=0.01,
            )
        )


def test_loso_benchmark_dispatches_calibrated_solver_prior_tuning(monkeypatch):
    from bayescatrack.experiments import track2p_solver_prior_tuning

    captured = {}

    def fake_run_track2p_loso_solver_prior_tuning(config, *, solver_prior_options):
        captured["config"] = config
        captured["options"] = solver_prior_options
        return _FakeLosoResult("calibrated priors")

    monkeypatch.setattr(
        track2p_solver_prior_tuning,
        "run_track2p_loso_solver_prior_tuning",
        fake_run_track2p_loso_solver_prior_tuning,
    )

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=Path("unused"),
            method="global-assignment",
            split="leave-one-subject-out",
            cost="calibrated",
            tune_solver_priors=True,
            solver_prior_objective="pairwise_f1",
            solver_prior_start_costs=(0.75,),
            solver_prior_end_costs=(0.8,),
            solver_prior_gap_penalties=(0.2,),
            solver_prior_cost_thresholds=(None,),
        )
    )

    assert rows[0].variant == "calibrated priors"
    assert captured["config"].cost == "calibrated"
    assert captured["options"].objective == "pairwise_f1"
    assert captured["options"].start_costs == (0.75,)
    assert captured["options"].end_costs == (0.8,)
    assert captured["options"].gap_penalties == (0.2,)
    assert captured["options"].cost_thresholds == (None,)
