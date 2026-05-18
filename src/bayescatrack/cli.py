"""BayesCaTrack command line entry point."""

from __future__ import annotations

import argparse
import sys

from bayescatrack.core.bridge import main as _core_main

_TOP_LEVEL_HELP = """usage: bayescatrack {summary,export,benchmark,growth} ...

BayesCaTrack command line tools.

commands:
  summary      Print a JSON summary for one Track2p-style subject directory.
  export       Export a PyRecEst-ready NPZ bundle for one subject.
  benchmark    Run reproducible benchmark harnesses.
  growth       Analyze global growth/displacement patterns from track tables.

Run 'bayescatrack <command> --help' for command-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch BayesCaTrack CLI commands."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_LEVEL_HELP)
        return 0

    if args[0] == "growth":
        from bayescatrack.analysis.growth import main as _growth_main

        return int(_growth_main(args[1:]))

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
            "track2p-sweep",
            help="Sweep Track2p global-assignment cost scales and thresholds",
        )
        subparsers.add_parser(
            "track2p-teacher-debug",
            help="Export Bayes/Track2p/manual-GT disagreement diagnostics",
        )
        subparsers.add_parser(
            "track2p-teacher-diagnostics",
            help="Alias for track2p-teacher-debug",
        )
        subparsers.add_parser(
            "edge-ranking",
            help="Rank manual-GT Track2p edges within pairwise cost/feature matrices",
        )
        subparsers.add_parser(
            "registration-qa",
            help="Report registration quality on manual-GT Track2p links",
        )
        subparsers.add_parser(
            "oracle-affine-qa",
            help="Compare baseline registration to manual-GT oracle affine geometry",
        )
        subparsers.add_parser(
            "growth-registration-qa",
            help="Report spatially resolved growth/deformation registration QA",
        )
        subparsers.add_parser(
            "validate-track2p-inputs",
            help="Validate manual-GT ROI coverage before Track2p benchmarks",
        )
        subparsers.add_parser(
            "audit-manual-gt-rois",
            help="Audit manual-GT ROI index spaces before Track2p benchmarks",
        )
        subparsers.add_parser(
            "audit-manual-gt-roi-index-space",
            help="Alias for audit-manual-gt-rois",
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
    if args[0] == "track2p-sweep":
        from bayescatrack.experiments.track2p_cost_sweep import (
            main as _track2p_cost_sweep_main,
        )

        return int(_track2p_cost_sweep_main(args[1:]))
    if args[0] in {"track2p-teacher-debug", "track2p-teacher-diagnostics"}:
        from bayescatrack.experiments.track2p_teacher_debug import (
            main as _track2p_teacher_debug_main,
        )

        return int(_track2p_teacher_debug_main(args[1:]))
    if args[0] == "edge-ranking":
        from bayescatrack.experiments.track2p_edge_ranking import (
            main as _track2p_edge_ranking_main,
        )

        return int(_track2p_edge_ranking_main(args[1:]))
    if args[0] == "registration-qa":
        from bayescatrack.experiments.registration_qa_report import (
            main as _registration_qa_main,
        )

        return int(_registration_qa_main(args[1:]))
    if args[0] == "oracle-affine-qa":
        from bayescatrack.experiments.oracle_affine_registration_qa import (
            main as _oracle_affine_qa_main,
        )

        return int(_oracle_affine_qa_main(args[1:]))
    if args[0] == "growth-registration-qa":
        from bayescatrack.experiments.growth_registration_qa import (
            main as _growth_registration_qa_main,
        )

        return int(_growth_registration_qa_main(args[1:]))
    if args[0] == "validate-track2p-inputs":
        from bayescatrack.experiments.track2p_input_validator import (
            main as _track2p_input_validator_main,
        )

        return int(_track2p_input_validator_main(args[1:]))
    if args[0] in {"audit-manual-gt-rois", "audit-manual-gt-roi-index-space"}:
        from bayescatrack.experiments.track2p_roi_index_audit import (
            main as _track2p_roi_index_audit_main,
        )

        return int(_track2p_roi_index_audit_main(args[1:]))
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
