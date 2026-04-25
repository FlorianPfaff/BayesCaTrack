import importlib.util

import bayescatrack
from bayescatrack import association
from bayescatrack import io as bayescatrack_io
from bayescatrack import reference, registration, track2p_registration
from bayescatrack.core import bridge as bayescatrack_bridge
from bayescatrack.datasets import track2p as bayescatrack_track2p
from tests._support import run_module


def test_root_package_exports_expected_public_api():
    expected_names = set(bayescatrack_bridge.__all__)
    assert expected_names.issubset(set(bayescatrack.__all__))


def test_subpackages_expose_expected_package_native_modules():
    for module in (association, bayescatrack_track2p, bayescatrack_io):
        assert module.__all__
    for module in (reference, registration, track2p_registration):
        assert module.__name__.startswith("bayescatrack.")


def test_legacy_bridge_package_is_not_part_of_source_layout():
    assert importlib.util.find_spec("track2p_pyrecest_bridge") is None


def test_bayescatrack_module_entry_point_help():
    proc = run_module("-m", "bayescatrack", "--help")
    assert "summary" in proc.stdout
    assert "export" in proc.stdout
