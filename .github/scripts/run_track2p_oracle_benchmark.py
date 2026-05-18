"""Run the M1 oracle-GT-link Track2p benchmark."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import cast

from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind,
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
    write_results,
)


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def main() -> int:
    results_dir = Path("benchmark-results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "oracle_gt_links.csv"

    config = Track2pBenchmarkConfig(
        data=Path(os.environ["TRACK2P_DATA_PATH"]),
        reference=Path(os.environ["TRACK2P_REFERENCE_PATH"]),
        reference_kind=cast(
            ReferenceKind,
            os.environ.get("TRACK2P_REFERENCE_KIND", "manual-gt"),
        ),
        method="oracle-gt-links",
        plane_name=os.environ.get("TRACK2P_PLANE", "plane0"),
        input_format=os.environ.get("TRACK2P_INPUT_FORMAT", "auto"),
        include_non_cells=_bool_env("TRACK2P_INCLUDE_NON_CELLS", default=True),
        include_behavior=_bool_env("TRACK2P_INCLUDE_BEHAVIOR"),
        seed_session=int(os.environ.get("TRACK2P_SEED_SESSION", "0")),
        restrict_to_reference_seed_rois=_bool_env(
            "TRACK2P_RESTRICT_TO_REFERENCE_SEED_ROIS",
            default=True,
        ),
    )
    rows = [result.to_dict() for result in run_track2p_benchmark(config)]
    write_results(rows, output_path, "csv")

    failures: list[str] = []
    for row in csv.DictReader(output_path.open(newline="", encoding="utf-8")):
        pairwise = float(row["pairwise_f1"])
        complete = float(row["complete_track_f1"])
        if pairwise < 0.999 or complete < 0.999:
            failures.append(
                f"{row.get('subject', 'unknown')}: pairwise={pairwise:.3f}, complete={complete:.3f}"
            )
    if failures:
        raise SystemExit("Oracle GT link benchmark failed M1: " + "; ".join(failures))

    print(output_path.read_text(encoding="utf-8"))
    print(f"Oracle GT link benchmark passed M1 for {len(rows)} subject(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
