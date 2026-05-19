"""Tests for shifted-overlap benchmark ablation setup."""

from bayescatrack.experiments.track2p_shifted_overlap_ablation import (
    build_arg_parser,
    build_shifted_overlap_variants,
)


def test_shifted_overlap_ablation_variants_include_cosine_tie_breaker():
    variants = build_shifted_overlap_variants(
        (0, 2), shifted_cosine_weights=(0.5,)
    )

    assert [variant.name for variant in variants] == [
        "exact-iou",
        "shifted-iou-r2",
        "shifted-iou-r2-shifted-cosine-w0p5",
    ]
    cosine_variant = variants[-1]
    assert cosine_variant.shifted_iou_radius == 2
    assert cosine_variant.use_shifted_iou_for_iou_cost is True
    assert cosine_variant.shifted_mask_cosine_weight == 0.5
    assert cosine_variant.pairwise_cost_kwargs == {
        "shifted_iou_radius": 2,
        "use_shifted_iou_for_iou_cost": True,
        "shifted_mask_cosine_weight": 0.5,
    }


def test_shifted_overlap_ablation_parser_defaults_to_fov_affine_global_assignment():
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.method == "global-assignment"
    assert args.transform_type == "fov-affine"
    assert args.radii == "0,2,4,6,8"
