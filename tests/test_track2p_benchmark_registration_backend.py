from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    GlobalAssignmentRun,
    build_registered_pairwise_costs,
    solve_global_assignment_for_sessions,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _assignment_registration_metadata,
    build_arg_parser,
)


def test_track2p_benchmark_default_registration_backend_is_stable() -> None:
    config = Track2pBenchmarkConfig(data=Path("."), method="global-assignment")
    assert config.transform_type == "fov-affine"

    parser = build_arg_parser()
    args = parser.parse_args(["--data", ".", "--method", "global-assignment"])
    assert args.transform_type == "fov-affine"
    explicit_args = parser.parse_args(
        [
            "--data",
            ".",
            "--method",
            "global-assignment",
            "--transform-type",
            "fov-affine",
        ]
    )
    assert explicit_args.transform_type == "fov-affine"
    assert (
        inspect.signature(build_registered_pairwise_costs)
        .parameters["transform_type"]
        .default
        == "fov-affine"
    )
    assert (
        inspect.signature(solve_global_assignment_for_sessions)
        .parameters["transform_type"]
        .default
        == "fov-affine"
    )


def test_assignment_registration_metadata_is_reported_in_result_rows() -> None:
    assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[]),
        pairwise_costs={(0, 1): np.zeros((1, 1))},
        session_sizes=(1, 1),
        session_edges=((0, 1),),
        registration_backends={(0, 1): "fov-affine"},
        registration_transform_types={(0, 1): "fov-affine"},
        registration_backend_reasons={(0, 1): "explicit transform_type='fov-affine'"},
    )

    assert _assignment_registration_metadata(assignment) == {
        "registration_backend": "fov-affine",
        "registration_transform_type": "fov-affine",
        "registration_backend_reason": "explicit transform_type='fov-affine'",
    }
