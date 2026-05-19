"""Run reproducible Track2p benchmark suites from a JSON manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

from bayescatrack.experiments.benchmark_comparison import (
    ComparisonInput,
    aggregate_rows,
    load_labeled_rows,
    write_comparison,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
    write_results,
)

ManifestObject = Mapping[str, Any]
BenchmarkRunner = Literal[
    "track2p", "track2p-solver-prior-loso", "track2p-monotone-loso"
]
TRACK2P_CONFIG_FIELDS = {field.name for field in fields(Track2pBenchmarkConfig)}
RUN_METADATA_FIELDS = {
    "name",
    "output",
    "format",
    "runner",
    "solver_prior_search",
    "monotone_ranker_options",
}
COMPARISON_FIELDS = {"name", "inputs", "output", "format", "highlight_best"}


@dataclass(frozen=True)
class BenchmarkRunSpec:
    """One configured benchmark run from a manifest."""

    name: str
    config: Track2pBenchmarkConfig
    output: Path
    output_format: OutputFormat = "csv"
    runner: BenchmarkRunner = "track2p"
    solver_prior_search: Any | None = None
    monotone_ranker_options: Any | None = None


@dataclass(frozen=True)
class BenchmarkComparisonSpec:
    """One aggregate comparison table from manifest run outputs."""

    name: str
    inputs: Mapping[str, str]
    output: Path
    output_format: str = "markdown"
    highlight_best: bool = False


@dataclass(frozen=True)
class BenchmarkManifest:
    """Parsed benchmark manifest with resolved filesystem paths."""

    path: Path
    runs: tuple[BenchmarkRunSpec, ...]
    comparisons: tuple[BenchmarkComparisonSpec, ...] = ()


@dataclass(frozen=True)
class BenchmarkOutputSummary:
    """Output metadata for one completed manifest artifact."""

    name: str
    output: Path
    rows: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "output": str(self.output),
            "rows": int(self.rows),
        }


@dataclass(frozen=True)
class BenchmarkManifestResult:
    """Completed benchmark manifest outputs."""

    runs: tuple[BenchmarkOutputSummary, ...]
    comparisons: tuple[BenchmarkOutputSummary, ...]

    def to_dict(self) -> dict[str, list[dict[str, int | str]]]:
        return {
            "runs": [run.to_dict() for run in self.runs],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def load_benchmark_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    progress: bool | None = None,
) -> BenchmarkManifest:
    """Load a JSON benchmark manifest and resolve paths relative to it."""

    path = Path(manifest_path)
    raw_manifest = _load_json_object(path)
    base_dir = path.resolve().parent
    output_base_dir = base_dir if output_dir is None else Path(output_dir)

    defaults = raw_manifest.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("Manifest 'defaults' must be a JSON object")
    _reject_unknown_keys(defaults, TRACK2P_CONFIG_FIELDS, location="defaults")

    raw_runs = raw_manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Manifest must contain a non-empty 'runs' list")

    runs = tuple(
        _parse_run_spec(
            raw_run,
            defaults=defaults,
            base_dir=base_dir,
            output_base_dir=output_base_dir,
            progress=progress,
        )
        for raw_run in raw_runs
    )
    _validate_unique_names(run.name for run in runs)

    raw_comparisons = raw_manifest.get("comparisons", [])
    if not isinstance(raw_comparisons, list):
        raise ValueError("Manifest 'comparisons' must be a JSON list when provided")
    comparisons = tuple(
        _parse_comparison_spec(
            raw_comparison,
            output_base_dir=output_base_dir,
        )
        for raw_comparison in raw_comparisons
    )
    _validate_unique_names(comparison.name for comparison in comparisons)

    return BenchmarkManifest(path=path, runs=runs, comparisons=comparisons)


def run_benchmark_manifest(manifest: BenchmarkManifest) -> BenchmarkManifestResult:
    """Run all benchmark entries and comparison tables from a manifest."""

    run_summaries: list[BenchmarkOutputSummary] = []
    run_outputs: dict[str, Path] = {}
    for run_spec in manifest.runs:
        results = _run_benchmark_spec(run_spec)
        rows = [result.to_dict() for result in results]
        write_results(rows, run_spec.output, run_spec.output_format)
        run_summaries.append(
            BenchmarkOutputSummary(
                name=run_spec.name,
                output=run_spec.output,
                rows=len(rows),
            )
        )
        run_outputs[run_spec.name] = run_spec.output

    comparison_summaries: list[BenchmarkOutputSummary] = []
    for comparison_spec in manifest.comparisons:
        comparison_inputs = _comparison_inputs(
            comparison_spec, run_outputs=run_outputs, manifest_path=manifest.path
        )
        rows = aggregate_rows(load_labeled_rows(comparison_inputs))
        write_comparison(
            rows,
            comparison_spec.output,
            comparison_spec.output_format,
            highlight_best=comparison_spec.highlight_best,
        )
        comparison_summaries.append(
            BenchmarkOutputSummary(
                name=comparison_spec.name,
                output=comparison_spec.output,
                rows=len(rows),
            )
        )

    return BenchmarkManifestResult(
        runs=tuple(run_summaries), comparisons=tuple(comparison_summaries)
    )


def _run_benchmark_spec(run_spec: BenchmarkRunSpec) -> list[SubjectBenchmarkResult]:
    """Execute one manifest run with the selected benchmark runner."""

    if run_spec.runner == "track2p":
        return run_track2p_benchmark(run_spec.config)
    if run_spec.runner == "track2p-solver-prior-loso":
        from bayescatrack.experiments.solver_prior_tuning import (
            run_track2p_loso_solver_priors,
        )

        return run_track2p_loso_solver_priors(
            run_spec.config, search=run_spec.solver_prior_search
        ).to_benchmark_results()
    if run_spec.runner == "track2p-monotone-loso":
        from bayescatrack.experiments.track2p_monotone_loso_calibration import (
            run_track2p_monotone_loso_calibration,
        )

        return run_track2p_monotone_loso_calibration(
            run_spec.config, monotone_options=run_spec.monotone_ranker_options
        ).to_benchmark_results()
    raise ValueError(f"Unsupported benchmark runner: {run_spec.runner!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the manifest benchmark CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark suite",
        description="Run a reproducible benchmark suite from a JSON manifest.",
    )
    parser.add_argument("manifest", type=Path, help="JSON benchmark manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Resolve manifest output paths relative to this directory instead of the manifest directory",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override per-run benchmark progress reporting",
    )
    parser.add_argument(
        "--summary-format",
        choices=("json", "table"),
        default="json",
        help="Stdout summary format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark suite CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = load_benchmark_manifest(
        args.manifest, output_dir=args.output_dir, progress=args.progress
    )
    result = run_benchmark_manifest(manifest)
    if args.summary_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_summary_table(result))
    return 0


def _parse_run_spec(
    raw_run: Any,
    *,
    defaults: ManifestObject,
    base_dir: Path,
    output_base_dir: Path,
    progress: bool | None,
) -> BenchmarkRunSpec:
    if not isinstance(raw_run, Mapping):
        raise ValueError("Each manifest run must be a JSON object")
    _reject_unknown_keys(
        raw_run, TRACK2P_CONFIG_FIELDS | RUN_METADATA_FIELDS, location="runs[]"
    )

    run_data = {**defaults, **raw_run}
    runner = _benchmark_runner(run_data.get("runner", "track2p"))
    name = str(run_data.get("name", _default_run_name(run_data)))
    output_format = _output_format(run_data.get("format", "csv"))
    output = _resolve_output_path(
        run_data.get("output"),
        default_name=f"{_slugify(name)}.{_benchmark_output_suffix(output_format)}",
        output_base_dir=output_base_dir,
    )
    config_kwargs = _track2p_config_kwargs(run_data, base_dir=base_dir)
    config = Track2pBenchmarkConfig(**config_kwargs)
    solver_prior_search = _parse_solver_prior_search(
        run_data.get("solver_prior_search"), runner=runner
    )
    monotone_ranker_options = _parse_monotone_ranker_options(
        run_data.get("monotone_ranker_options"), runner=runner
    )
    if progress is not None:
        config = replace(config, progress=progress)
    return BenchmarkRunSpec(
        name=name,
        config=config,
        output=output,
        output_format=output_format,
        runner=runner,
        solver_prior_search=solver_prior_search,
        monotone_ranker_options=monotone_ranker_options,
    )


def _parse_comparison_spec(
    raw_comparison: Any,
    *,
    output_base_dir: Path,
) -> BenchmarkComparisonSpec:
    if not isinstance(raw_comparison, Mapping):
        raise ValueError("Each manifest comparison must be a JSON object")
    _reject_unknown_keys(raw_comparison, COMPARISON_FIELDS, location="comparisons[]")

    inputs = raw_comparison.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("Manifest comparison 'inputs' must be a non-empty JSON object")
    comparison_inputs = {str(label): str(source) for label, source in inputs.items()}

    name = str(raw_comparison.get("name", "comparison"))
    output_format = str(raw_comparison.get("format", "markdown"))
    if output_format not in {"markdown", "csv"}:
        raise ValueError("Manifest comparison format must be 'markdown' or 'csv'")
    output = _resolve_output_path(
        raw_comparison.get("output"),
        default_name=f"{_slugify(name)}.{_comparison_output_suffix(output_format)}",
        output_base_dir=output_base_dir,
    )
    raw_highlight_best = raw_comparison.get("highlight_best", False)
    if not isinstance(raw_highlight_best, bool):
        raise ValueError("Manifest comparison 'highlight_best' must be a boolean")
    return BenchmarkComparisonSpec(
        name=name,
        inputs=comparison_inputs,
        output=output,
        output_format=output_format,
        highlight_best=raw_highlight_best,
    )


def _track2p_config_kwargs(
    run_data: ManifestObject, *, base_dir: Path
) -> dict[str, Any]:
    config_kwargs = {
        key: value for key, value in run_data.items() if key in TRACK2P_CONFIG_FIELDS
    }
    missing_required = [key for key in ("data", "method") if key not in config_kwargs]
    if missing_required:
        raise ValueError(
            f"Manifest run is missing required Track2p config keys: {', '.join(missing_required)}"
        )
    for key in ("data", "reference"):
        if key in config_kwargs and config_kwargs[key] is not None:
            config_kwargs[key] = _resolve_input_path(
                config_kwargs[key], base_dir=base_dir
            )
    return config_kwargs


def _comparison_inputs(
    comparison_spec: BenchmarkComparisonSpec,
    *,
    run_outputs: Mapping[str, Path],
    manifest_path: Path,
) -> list[ComparisonInput]:
    inputs: list[ComparisonInput] = []
    base_dir = manifest_path.resolve().parent
    for label, source in comparison_spec.inputs.items():
        if source in run_outputs:
            source_path = run_outputs[source]
        else:
            source_path = _resolve_input_path(source, base_dir=base_dir)
        inputs.append(ComparisonInput(label=label, path=source_path))
    return inputs


def _load_json_object(path: Path) -> ManifestObject:
    if path.suffix.casefold() != ".json":
        raise ValueError(f"Benchmark manifests must be JSON files, got {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Benchmark manifest must be a JSON object")
    return data


def _resolve_input_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_output_path(
    value: Any, *, default_name: str, output_base_dir: Path
) -> Path:
    if value is None:
        path = Path("benchmark-results") / default_name
    else:
        path = Path(str(value))
    if path.is_absolute():
        return path
    return output_base_dir / path


def _reject_unknown_keys(
    raw_object: ManifestObject, allowed: set[str], *, location: str
) -> None:
    unknown_keys = sorted(str(key) for key in raw_object if key not in allowed)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in manifest {location}: {', '.join(unknown_keys)}"
        )


def _validate_unique_names(names: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"Manifest names must be unique: {', '.join(sorted(duplicates))}"
        )


def _default_run_name(run_data: ManifestObject) -> str:
    method = str(run_data.get("method", "run"))
    cost = run_data.get("cost")
    if cost is None:
        return method
    return f"{method}-{cost}"


def _output_format(value: Any) -> OutputFormat:
    output_format = str(value)
    if output_format not in {"table", "json", "csv"}:
        raise ValueError("Manifest run format must be 'table', 'json', or 'csv'")
    return cast(OutputFormat, output_format)


def _benchmark_runner(value: Any) -> BenchmarkRunner:
    runner = str(value)
    allowed = {"track2p", "track2p-solver-prior-loso", "track2p-monotone-loso"}
    if runner not in allowed:
        raise ValueError(
            "Manifest run runner must be 'track2p', 'track2p-solver-prior-loso', "
            "or 'track2p-monotone-loso'"
        )
    return cast(BenchmarkRunner, runner)


def _parse_solver_prior_search(value: Any, *, runner: BenchmarkRunner) -> Any | None:
    if value is None:
        return None
    if runner != "track2p-solver-prior-loso":
        raise ValueError(
            "Manifest run 'solver_prior_search' is only valid with "
            "runner='track2p-solver-prior-loso'"
        )
    if not isinstance(value, Mapping):
        raise ValueError("Manifest run 'solver_prior_search' must be a JSON object")
    from bayescatrack.experiments.solver_prior_tuning import SolverPriorSearchConfig

    allowed = {field.name for field in fields(SolverPriorSearchConfig)}
    _reject_unknown_keys(value, allowed, location="solver_prior_search")
    return SolverPriorSearchConfig(**dict(value))


def _parse_monotone_ranker_options(
    value: Any, *, runner: BenchmarkRunner
) -> Any | None:
    if value is None:
        return None
    if runner != "track2p-monotone-loso":
        raise ValueError(
            "Manifest run 'monotone_ranker_options' is only valid with "
            "runner='track2p-monotone-loso'"
        )
    if not isinstance(value, Mapping):
        raise ValueError("Manifest run 'monotone_ranker_options' must be a JSON object")
    from bayescatrack.association.monotone_ranker import MonotoneRankerOptions

    allowed = {field.name for field in fields(MonotoneRankerOptions)}
    _reject_unknown_keys(value, allowed, location="monotone_ranker_options")
    return MonotoneRankerOptions(**dict(value))


def _benchmark_output_suffix(output_format: OutputFormat) -> str:
    return {"csv": "csv", "json": "json", "table": "md"}[output_format]


def _comparison_output_suffix(output_format: str) -> str:
    return "csv" if output_format == "csv" else "md"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "benchmark"


def _format_summary_table(result: BenchmarkManifestResult) -> str:
    rows: list[tuple[str, str, int, Path]] = []
    rows.extend(
        ("run", summary.name, summary.rows, summary.output) for summary in result.runs
    )
    rows.extend(
        ("comparison", summary.name, summary.rows, summary.output)
        for summary in result.comparisons
    )
    if not rows:
        return "No benchmark outputs were written."

    header = "| kind | name | rows | output |"
    separator = "| --- | --- | ---: | --- |"
    body = [header, separator]
    body.extend(
        f"| {kind} | {name} | {row_count} | {output} |"
        for kind, name, row_count, output in rows
    )
    return "\n".join(body)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
