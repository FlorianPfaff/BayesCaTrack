from __future__ import annotations

from bayescatrack import cli
from bayescatrack.experiments import track2p_shifted_iou_benchmark

# pylint: disable=protected-access


def test_benchmark_dispatches_shifted_iou_subcommand(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_shifted_iou_main(argv: list[str]) -> int:
        seen["argv"] = list(argv)
        return 17

    monkeypatch.setattr(track2p_shifted_iou_benchmark, "main", fake_shifted_iou_main)

    status = cli._handle_benchmark(["track2p-shifted-iou", "--data", "dataset"])

    assert status == 17
    assert seen["argv"] == ["--data", "dataset"]
