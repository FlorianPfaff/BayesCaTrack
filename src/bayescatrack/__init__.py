"""BayesCaTrack public package API."""

# pylint: disable=duplicate-code

from . import cli as _cli
from .core import bridge as _bridge


def _install_registration_transform_argparse_patch() -> None:
    """Keep legacy CLI parsers aligned with the central registration transform set."""

    import argparse as _argparse

    current_add_argument = _argparse.ArgumentParser.add_argument
    patch_flag = "bayescatrack_registration_transform_patch"
    if getattr(current_add_argument, patch_flag, False):
        return

    transform_choices = (
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
        "gt-affine-oracle",
        "none",
    )

    def _bayescatrack_add_argument(self, *name_or_flags, **kwargs):
        if "--transform-type" in name_or_flags:
            choices = kwargs.get("choices")
            try:
                choices_tuple = tuple(choices) if choices is not None else ()
            except TypeError:
                choices_tuple = ()
            if choices_tuple:
                allowed = tuple(
                    value for value in transform_choices if value in choices_tuple or value != "gt-affine-oracle"
                )
                if "gt-affine-oracle" in choices_tuple:
                    allowed = tuple(value for value in transform_choices if value in {*choices_tuple, *transform_choices})
                kwargs = {**kwargs, "choices": allowed}
            help_text = kwargs.get("help")
            if isinstance(help_text, str) and "bspline" not in help_text:
                kwargs["help"] = (
                    f"{help_text}; supports fov-affine and nonrigid growth-aware transforms "
                    "bspline, tps, local-affine-grid, and optical-flow"
                )
        return current_add_argument(self, *name_or_flags, **kwargs)

    setattr(_bayescatrack_add_argument, patch_flag, True)
    setattr(_argparse.ArgumentParser, "add_argument", _bayescatrack_add_argument)


_install_registration_transform_argparse_patch()

CalciumPlaneData = _bridge.CalciumPlaneData
SessionAssociationBundle = _bridge.SessionAssociationBundle
Track2pSession = _bridge.Track2pSession
build_consecutive_session_association_bundles = (
    _bridge.build_consecutive_session_association_bundles
)
build_session_pair_association_bundle = _bridge.build_session_pair_association_bundle
export_subject_to_npz = _bridge.export_subject_to_npz
find_track2p_session_dirs = _bridge.find_track2p_session_dirs
load_raw_npy_plane = _bridge.load_raw_npy_plane
load_suite2p_plane = _bridge.load_suite2p_plane
load_track2p_subject = _bridge.load_track2p_subject
main = _cli.main
summarize_subject = _bridge.summarize_subject

__all__ = tuple(dict.fromkeys((*_bridge.__all__, "main")))