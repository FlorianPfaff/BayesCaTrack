"""CLI coverage for the registered-soft-iou benchmark preset."""

from __future__ import annotations

from bayescatrack.experiments.track2p_benchmark import (
    _config_from_args,
    _variant_name,
    build_arg_parser,
)


def test_track2p_benchmark_cli_accepts_registered_soft_iou() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data",
            "dummy-dataset-root",
            "--method",
            "global-assignment",
            "--cost",
            "registered-soft-iou",
        ]
    )

    config = _config_from_args(args)

    assert config.cost == "registered-soft-iou"


def test_registered_soft_iou_has_distinct_variant_name() -> None:
    assert (
        _variant_name("registered-soft-iou")
        == "Registered soft-IoU + global assignment"
    )
