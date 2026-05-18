"""Backend-audit helpers that expose nonrigid registration diagnostics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import session_edge_pairs
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.registration_qa_report import (  # pylint: disable=protected-access
    OutputFormat,
    RegistrationQAConfig,
    _benchmark_config,
    _registration_metadata,
)
from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=protected-access
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair

NONRIGID_REGISTRATION_TEXT_FIELDS: tuple[str, ...] = (
    "nonrigid_registration_backend",
    "nonrigid_registration_grid_shape",
)
NONRIGID_REGISTRATION_NUMERIC_FIELDS: tuple[str, ...] = (
    "nonrigid_registration_landmarks",
    "nonrigid_registration_fit_rmse",
    "nonrigid_registration_inverse_warp_valid_fraction",
    "nonrigid_registration_tps_regularization",
    "nonrigid_registration_optical_flow_iterations",
    "nonrigid_registration_optical_flow_alpha",
)
NONRIGID_REGISTRATION_BOOL_FIELDS: tuple[str, ...] = (
    "nonrigid_registration_fallback_translation",
)
NONRIGID_BACKEND_AUDIT_TABLE_COLUMNS: tuple[str, ...] = (
    "cost",
    "registration_backend",
    "transform_type",
    "registered_plane_source",
    "edge_count",
    "gt_link_rows",
    "subject_count",
    "subjects",
    "median_fov_translation_shift_y",
    "median_fov_translation_shift_x",
    "median_fov_translation_peak_correlation",
    "nonrigid_registration_backend",
    "nonrigid_registration_grid_shape",
    "median_nonrigid_registration_landmarks",
    "median_nonrigid_registration_fit_rmse",
    "median_nonrigid_registration_inverse_warp_valid_fraction",
    "nonrigid_registration_fallback_translation_rate",
    "median_nonrigid_registration_tps_regularization",
    "median_nonrigid_registration_optical_flow_iterations",
    "median_nonrigid_registration_optical_flow_alpha",
    "registration_backend_reason",
)


def run_registration_backend_audit_report(
    config: RegistrationQAConfig,
) -> list[dict[str, Any]]:
    """Register audited session edges and expose backend diagnostics from ``ops``."""

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    benchmark_config = _benchmark_config(config)
    edge_rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir,
            data_root=config.data,
            config=benchmark_config,
        )
        _validate_reference_for_benchmark(
            reference,
            subject_dir=subject_dir,
            config=benchmark_config,
        )
        sessions = _load_subject_sessions(subject_dir, benchmark_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference,
            curated_only=config.curated_only,
        )
        for source_index, target_index in session_edge_pairs(
            len(sessions), max_gap=config.max_gap
        ):
            gt_link_rows = _gt_link_count(reference_matrix, source_index, target_index)
            if gt_link_rows == 0:
                continue
            registered_plane = register_plane_pair(
                sessions[source_index].plane_data,
                sessions[target_index].plane_data,
                transform_type=config.transform_type,
            )
            edge_rows.append(
                {
                    "cost": config.cost,
                    "subject": subject_dir.name,
                    "source_session_index": source_index,
                    "target_session_index": target_index,
                    "gt_link_rows": gt_link_rows,
                    "transform_type": config.transform_type,
                    **_registration_metadata(config.transform_type, registered_plane),
                    **_nonrigid_registration_metadata(registered_plane),
                }
            )
    return summarize_registration_backend_audit_edges(edge_rows)


def summarize_registration_backend_audit_edges(
    edge_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-edge registration-backend diagnostics."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        key = (
            str(row.get("cost", "")),
            str(row.get("registration_backend", "")),
            str(row.get("transform_type", "")),
            str(row.get("registered_plane_source", "")),
            str(row.get("registration_backend_reason", "")),
            str(row.get("nonrigid_registration_backend", "")),
            str(row.get("nonrigid_registration_grid_shape", "")),
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for (
        cost,
        registration_backend,
        transform_type,
        registered_plane_source,
        registration_backend_reason,
        nonrigid_backend,
        nonrigid_grid_shape,
    ), rows in sorted(groups.items()):
        subjects = sorted({str(row["subject"]) for row in rows})
        summary.append(
            {
                "cost": cost,
                "registration_backend": registration_backend,
                "transform_type": transform_type,
                "registered_plane_source": registered_plane_source,
                "registration_backend_reason": registration_backend_reason,
                "edge_count": len(rows),
                "gt_link_rows": int(sum(int(row["gt_link_rows"]) for row in rows)),
                "subject_count": len(subjects),
                "subjects": ",".join(subjects),
                "median_fov_translation_shift_y": _stat(
                    rows, "fov_translation_shift_y"
                ),
                "median_fov_translation_shift_x": _stat(
                    rows, "fov_translation_shift_x"
                ),
                "median_fov_translation_peak_correlation": _stat(
                    rows,
                    "fov_translation_peak_correlation",
                ),
                "nonrigid_registration_backend": nonrigid_backend,
                "nonrigid_registration_grid_shape": nonrigid_grid_shape,
                "median_nonrigid_registration_landmarks": _stat(
                    rows,
                    "nonrigid_registration_landmarks",
                ),
                "median_nonrigid_registration_fit_rmse": _stat(
                    rows,
                    "nonrigid_registration_fit_rmse",
                ),
                "median_nonrigid_registration_inverse_warp_valid_fraction": _stat(
                    rows,
                    "nonrigid_registration_inverse_warp_valid_fraction",
                ),
                "nonrigid_registration_fallback_translation_rate": _mean_bool_optional(
                    rows,
                    "nonrigid_registration_fallback_translation",
                ),
                "median_nonrigid_registration_tps_regularization": _stat(
                    rows,
                    "nonrigid_registration_tps_regularization",
                ),
                "median_nonrigid_registration_optical_flow_iterations": _stat(
                    rows,
                    "nonrigid_registration_optical_flow_iterations",
                ),
                "median_nonrigid_registration_optical_flow_alpha": _stat(
                    rows,
                    "nonrigid_registration_optical_flow_alpha",
                ),
            }
        )
    return summary


def write_nonrigid_registration_backend_audit_results(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write extended backend-audit rows as JSON, CSV, or Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(
        format_nonrigid_registration_backend_audit_table(rows) + "\n",
        encoding="utf-8",
    )


def format_nonrigid_registration_backend_audit_table(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Format extended backend-audit rows as a Markdown table."""

    columns = [
        column
        for column in NONRIGID_BACKEND_AUDIT_TABLE_COLUMNS
        if any(column in row for row in rows)
    ] or list(NONRIGID_BACKEND_AUDIT_TABLE_COLUMNS)
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def _nonrigid_registration_metadata(
    registered_plane: CalciumPlaneData,
) -> dict[str, Any]:
    ops = {} if registered_plane.ops is None else dict(registered_plane.ops)
    metadata: dict[str, Any] = {
        key: _text_ops_value(ops, key) for key in NONRIGID_REGISTRATION_TEXT_FIELDS
    }
    metadata.update(
        {
            key: _float_ops_value(ops, key)
            for key in NONRIGID_REGISTRATION_NUMERIC_FIELDS
        }
    )
    metadata.update(
        {key: _bool_ops_value(ops, key) for key in NONRIGID_REGISTRATION_BOOL_FIELDS}
    )
    return metadata


def _gt_link_count(
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
) -> int:
    count = 0
    for track in reference_matrix:
        if track[source_index] is not None and track[target_index] is not None:
            count += 1
    return count


def _text_ops_value(ops: Mapping[str, Any], key: str) -> str:
    if key not in ops or ops[key] is None:
        return ""
    value = ops[key]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _float_ops_value(ops: Mapping[str, Any], key: str) -> float:
    if key not in ops:
        return np.nan
    try:
        return float(ops[key])
    except (TypeError, ValueError):
        return np.nan


def _bool_ops_value(ops: Mapping[str, Any], key: str) -> bool | float:
    if key not in ops or ops[key] is None:
        return np.nan
    value = ops[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        return bool(value_float) if np.isfinite(value_float) else np.nan
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return np.nan


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _stat(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    if not values.size:
        return np.nan
    return float(np.median(values))


def _mean_bool_optional(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values: list[bool] = []
    for row in rows:
        value = row.get(key, np.nan)
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            value_float = float(value)
            if np.isfinite(value_float):
                values.append(bool(value_float))
    if not values:
        return np.nan
    return float(np.mean(values))


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)
