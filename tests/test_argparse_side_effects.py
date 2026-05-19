"""Regression tests for BayesCaTrack CLI argument handling."""

from __future__ import annotations

import argparse
import importlib

import pytest


def test_import_bayescatrack_does_not_patch_argparse_add_argument() -> None:
    """Importing the library must not alter unrelated argparse parsers."""

    importlib.import_module("bayescatrack")

    add_argument = argparse.ArgumentParser.add_argument
    assert not getattr(add_argument, "bayescatrack_registration_transform_patch", False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--transform-type", choices=("affine", "none"))

    with pytest.raises(SystemExit):
        parser.parse_args(["--transform-type", "bspline"])


def test_track2p_benchmark_parser_declares_bayescatrack_choices_locally() -> None:
    """BayesCaTrack CLIs still accept BayesCaTrack-specific choices explicitly."""

    from bayescatrack.experiments.track2p_benchmark import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--data",
            ".",
            "--method",
            "global-assignment",
            "--cost",
            "registered-soft-iou",
            "--transform-type",
            "bspline",
        ]
    )
    assert args.cost == "registered-soft-iou"
    assert args.transform_type == "bspline"
