from __future__ import annotations

from bayescatrack.experiments import (
    oracle_affine_registration_qa,
    registration_qa_report,
    solver_prior_tuning,
    track2p_activity_tie_breaker_sweep,
    track2p_benchmark,
    track2p_calibration_export,
    track2p_configurable_loso_calibration,
    track2p_cost_sweep,
    track2p_monotone_loso_calibration,
)
from bayescatrack.experiments.registration_qa_report import (
    REGISTRATION_QA_TRANSFORM_TYPES,
)
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES

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


def test_track2p_benchmark_cli_accepts_all_registration_transforms():
    parser = track2p_benchmark.build_arg_parser()

    for transform_type in REGISTRATION_TRANSFORM_TYPES:
        args = parser.parse_args(
            [
                "--data",
                "dataset",
                "--method",
                "global-assignment",
                "--transform-type",
                transform_type,
            ]
        )
        config = track2p_benchmark._config_from_args(args)

        assert config.transform_type == transform_type


def test_track2p_benchmark_cli_accepts_transform_suite():
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-suite",
            "fov-translation",
            "fov-affine",
            "bspline",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.transform_suite == ("fov-translation", "fov-affine", "bspline")


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


def test_track2p_benchmark_family_clis_accept_nonrigid_transform():
    transform_type = "tps"

    benchmark_args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-type",
            transform_type,
        ]
    )
    cost_sweep_args = track2p_cost_sweep.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--cost-scales",
            "1",
            "--cost-thresholds",
            "6",
            "--transform-type",
            transform_type,
        ]
    )
    activity_args = track2p_activity_tie_breaker_sweep.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", transform_type]
    )
    calibration_args = track2p_calibration_export.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--output",
            "calibration.csv",
            "--transform-type",
            transform_type,
        ]
    )
    solver_prior_args = solver_prior_tuning.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--transform-type",
            transform_type,
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
    configurable_loso_args = (
        track2p_configurable_loso_calibration.build_arg_parser().parse_args(
            ["--data", "dataset", "--transform-type", transform_type]
        )
    )
    monotone_loso_args = (
        track2p_monotone_loso_calibration.build_arg_parser().parse_args(
            ["--data", "dataset", "--transform-type", transform_type]
        )
    )

    assert (
        track2p_benchmark._config_from_args(benchmark_args).transform_type
        == transform_type
    )
    assert track2p_cost_sweep._config_from_args(
        cost_sweep_args
    ).benchmark.transform_type == transform_type
    assert track2p_activity_tie_breaker_sweep._config_from_args(
        activity_args
    ).benchmark.transform_type == transform_type
    assert calibration_args.transform_type == transform_type
    assert solver_prior_args.transform_type == transform_type
    assert configurable_loso_args.transform_type == transform_type
    assert monotone_loso_args.transform_type == transform_type


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


def test_registration_qa_cli_accepts_registration_and_oracle_transforms():
    parser = registration_qa_report.build_arg_parser()

    for transform_type in REGISTRATION_QA_TRANSFORM_TYPES:
        args = parser.parse_args(
            ["--data", "dataset", "--transform-type", transform_type]
        )
        config = registration_qa_report._config_from_args(args)

        assert config.transform_type == transform_type
