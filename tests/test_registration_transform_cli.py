from __future__ import annotations

from bayescatrack.experiments import (
    oracle_affine_registration_qa,
    registration_qa_report,
    solver_prior_tuning,
    track2p_benchmark,
    track2p_calibration_export,
    track2p_cost_sweep,
)

# pylint: disable=protected-access


def test_track2p_benchmark_cli_accepts_fov_translation_transform():
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-type",
            "fov-translation",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.transform_type == "fov-translation"


def test_track2p_cost_sweep_cli_accepts_fov_translation_transform():
    args = track2p_cost_sweep.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--cost-scales",
            "1",
            "--cost-thresholds",
            "6",
            "--transform-type",
            "fov-translation",
        ]
    )

    config = track2p_cost_sweep._config_from_args(args)

    assert config.benchmark.transform_type == "fov-translation"


def test_solver_prior_loso_cli_accepts_fov_translation_transform():
    args = solver_prior_tuning.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--transform-type",
            "fov-translation",
            "--start-costs",
            "1",
            "--end-costs",
            "1",
            "--gap-penalties",
            "0",
            "--cost-thresholds",
            "none",
        ]
    )

    assert args.transform_type == "fov-translation"


def test_calibration_export_cli_accepts_fov_translation_transform():
    args = track2p_calibration_export.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--output",
            "calibration.csv",
            "--transform-type",
            "fov-translation",
        ]
    )

    assert args.transform_type == "fov-translation"


def test_registration_qa_cli_accepts_fov_translation_transform():
    args = registration_qa_report.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "fov-translation"]
    )

    config = registration_qa_report._config_from_args(args)

    assert config.transform_type == "fov-translation"


def test_oracle_affine_qa_cli_accepts_fov_translation_transform():
    args = oracle_affine_registration_qa.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "fov-translation"]
    )

    config = oracle_affine_registration_qa.OracleAffineQAConfig(
        registration=registration_qa_report._config_from_args(args)
    )

    assert config.registration.transform_type == "fov-translation"


def test_registration_qa_cli_accepts_gt_affine_oracle_transform():
    args = registration_qa_report.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "gt-affine-oracle"]
    )

    config = registration_qa_report._config_from_args(args)

    assert config.transform_type == "gt-affine-oracle"
