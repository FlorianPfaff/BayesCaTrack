"""Shared argparse choices for BayesCaTrack-owned command line parsers."""

ASSOCIATION_COST_CHOICES: tuple[str, ...] = (
    "registered-iou",
    "registered-soft-iou",
    "registered-shifted-iou",
    "roi-aware",
    "roi-aware-shifted",
    "calibrated",
)

REGISTRATION_TRANSFORM_CHOICES: tuple[str, ...] = (
    "affine",
    "rigid",
    "fov-translation",
    "fov-affine",
    "bspline",
    "b-spline",
    "thin-plate-spline",
    "tps",
    "landmark-tps",
    "local-affine-grid",
    "optical-flow",
    "none",
)

REGISTRATION_TRANSFORM_HELP = (
    "Track2p registration transform type; supports Track2p affine/rigid, "
    "BayesCaTrack FOV transforms fov-translation/fov-affine, and growth-aware "
    "transforms bspline, tps, local-affine-grid, and optical-flow"
)

ASSOCIATION_COST_HELP = (
    "Pairwise cost used by global assignment; supports registered-soft-iou and "
    "registered-shifted-iou/roi-aware-shifted for near-miss registered ROI overlap"
)
