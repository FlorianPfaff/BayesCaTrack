"""CLI plumbing for LOSO calibrated association-cost benchmarks."""

from __future__ import annotations

from pathlib import Path

from bayescatrack.experiments.track2p_benchmark import (
    _config_from_args,
    _csv_fieldnames,
    build_arg_parser,
)


def test_track2p_benchmark_cli_parses_calibration_knobs() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "/tmp/track2p",
            "--method",
            "global-assignment",
            "--split",
            "leave-one-subject-out",
            "--cost",
            "calibrated",
            "--calibration-feature-set",
            "default+local-evidence",
            "--calibration-sample-weight-strategy",
            "balanced",
            "--calibration-model-kwargs-json",
            '{"C": 0.25, "class_weight": null}',
            "--calibration-hard-negative-ratio",
            "7.5",
            "--calibration-hard-negative-top-k",
            "11",
            "--calibration-hard-negative-no-column-candidates",
            "--calibration-hard-negative-features",
            "centroid_distance, one_minus_iou",
            "--pairwise-cost-kwargs-json",
            '{"local_evidence_components": true}',
        ]
    )

    config = _config_from_args(args)

    assert config.data == Path("/tmp/track2p")
    assert config.calibration_feature_set == "default+local-evidence"
    assert config.calibration_sample_weight_strategy == "balanced"
    assert config.calibration_model_kwargs == {"C": 0.25, "class_weight": None}
    assert config.calibration_hard_negative_ratio == 7.5
    assert config.calibration_hard_negative_top_k == 11
    assert config.calibration_hard_negative_include_columns is False
    assert config.calibration_hard_negative_features == (
        "centroid_distance",
        "one_minus_iou",
    )
    assert config.pairwise_cost_kwargs == {"local_evidence_components": True}


def test_track2p_benchmark_cli_can_disable_hard_negative_top_k() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "/tmp/track2p",
            "--method",
            "global-assignment",
            "--split",
            "leave-one-subject-out",
            "--cost",
            "calibrated",
            "--no-calibration-hard-negative-top-k",
        ]
    )

    config = _config_from_args(args)

    assert config.calibration_hard_negative_top_k is None


def test_calibration_fields_are_prioritized_in_csv_output() -> None:
    fieldnames = _csv_fieldnames(
        [
            {
                "subject": "jm001",
                "variant": "calibrated",
                "calibration_feature_set": "default",
                "calibration_hard_negative_ratio": 4.0,
                "zzz_extra": 1,
            }
        ]
    )

    assert fieldnames.index("calibration_feature_set") < fieldnames.index("zzz_extra")
    assert fieldnames.index("calibration_hard_negative_ratio") < fieldnames.index(
        "zzz_extra"
    )
