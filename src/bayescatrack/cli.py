"""BayesCaTrack command line entry point."""

from __future__ import annotations

import argparse
import sys

from bayescatrack.core.bridge import main as _core_main

_TOP_LEVEL_HELP = """usage: bayescatrack {summary,export,benchmark} ...

BayesCaTrack command line tools.

commands:
  summary      Print a JSON summary for one Track2p-style subject directory.
  export       Export a PyRecEst-ready NPZ bundle for one subject.
  benchmark    Run reproducible benchmark harnesses.

Run 'bayescatrack <command> --help' for command-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch BayesCaTrack CLI commands."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_LEVEL_HELP)
        return 0

    if args[0] != "benchmark":
        return int(_core_main(args))

    return _handle_benchmark(args[1:])


def _handle_benchmark(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="bayescatrack benchmark",
            description="Run BayesCaTrack benchmark harnesses.",
        )
        subparsers = parser.add_subparsers(dest="benchmark", required=False)
        subparsers.add_parser(
            "track2p", help="Track2p baseline and global-assignment ablations"
        )
        subparsers.add_parser(
            "compare", help="Aggregate benchmark CSVs into a comparison table"
        )
        subparsers.add_parser("suite", help="Run a JSON benchmark manifest")
        parser.parse_args(args)
        return 0

    if args[0] == "track2p":
        from bayescatrack.experiments.track2p_benchmark import (
            main as _track2p_benchmark_main,
        )

        return int(_track2p_benchmark_main(args[1:]))
    if args[0] == "compare":
        from bayescatrack.experiments.benchmark_comparison import (
            main as _benchmark_comparison_main,
        )

        return int(_benchmark_comparison_main(args[1:]))
    if args[0] == "suite":
        from bayescatrack.experiments.benchmark_manifest import (
            main as _benchmark_manifest_main,
        )

        return int(_benchmark_manifest_main(args[1:]))

    parser = argparse.ArgumentParser(prog="bayescatrack benchmark")
    parser.error(f"unknown benchmark {args[0]!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
