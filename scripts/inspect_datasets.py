from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure `src` package imports work even when script is run from outside repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.data.dataset_specs import DATASET_SPECS, KDD_FEATURE_COLUMNS, DatasetSpec

LOGGER = logging.getLogger("dataset_inspector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw datasets and emit profiles/schema reports.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Root directory of raw datasets.")
    parser.add_argument(
        "--out_dir", type=Path, default=Path("reports/datasets"), help="Output directory for profile reports."
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=200_000,
        help="Maximum rows used for sampled statistics when datasets are large.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling.")
    return parser.parse_args()


def detect_delimiter(path: Path) -> str:
    preview = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
    if not preview:
        return ","
    first_line = preview[0]
    candidates = [",", "\t", " ", ";", "|"]
    counts = {delim: first_line.count(delim) for delim in candidates}
    best_delim = max(counts, key=counts.get)
    return best_delim if counts[best_delim] > 0 else ","


def ensure_files(spec: DatasetSpec, data_dir: Path) -> list[Path]:
    file_paths = spec.resolved_files(data_dir)
    missing = [str(path) for path in file_paths if not path.exists()]
    if missing:
        missing_text = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing expected files for '{spec.name}':\n  - {missing_text}")
    return file_paths


def resolve_data_dir(requested_data_dir: Path) -> Path:
    """Resolve dataset root robustly for common project layouts.

    Supports these common patterns:
    - <repo>/data/raw
    - <repo>/../data/raw (sibling to repo directory)
    """

    candidates = [requested_data_dir]
    if not requested_data_dir.is_absolute():
        candidates.append((REPO_ROOT / requested_data_dir).resolve())
        candidates.append((REPO_ROOT.parent / requested_data_dir).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate data_dir. Tried:\n"
        f"  - {tried}\n"
        "Hint: if your layout is tabularToGraph/{data,wertyu}, run with "
        "--data_dir ../data/raw from inside 'wertyu'."
    )


def load_kdd(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        delimiter = detect_delimiter(path)
        LOGGER.info("Loading KDD split: %s (delimiter=%r)", path, delimiter)
        frame = pd.read_csv(path, header=None, sep=delimiter, low_memory=False)
        if frame.shape[1] != len(KDD_FEATURE_COLUMNS):
            raise ValueError(
                f"NSL-KDD column count mismatch for {path}: "
                f"expected {len(KDD_FEATURE_COLUMNS)} columns, found {frame.shape[1]}."
            )
        frame.columns = KDD_FEATURE_COLUMNS
        frame["_source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_csv_with_source(paths: list[Path], dataset_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        LOGGER.info("Loading %s split: %s", dataset_name, path)
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sample_frame(df: pd.DataFrame, sample_rows: int, seed: int) -> tuple[pd.DataFrame, bool]:
    if len(df) <= sample_rows:
        return df.copy(), False
    return df.sample(n=sample_rows, random_state=seed), True


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def profile_columns(df_full: pd.DataFrame, df_sample: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    missing_counts = df_full.isna().sum()

    for column in df_full.columns:
        series_full = df_full[column]
        series_sample = df_sample[column]
        column_profile: dict[str, Any] = {
            "dtype": str(series_full.dtype),
            "missing_count": int(missing_counts[column]),
            "missing_pct": round(float(missing_counts[column]) / max(len(df_full), 1) * 100.0, 6),
        }

        numeric_series_sample = pd.to_numeric(series_sample, errors="coerce")
        numeric_valid = numeric_series_sample.notna().sum()

        if pd.api.types.is_numeric_dtype(series_full) or numeric_valid >= 0.95 * max(len(series_sample), 1):
            valid_values = numeric_series_sample.dropna()
            zero_pct = None
            if len(valid_values) > 0:
                zero_pct = float((valid_values == 0).mean() * 100.0)
            column_profile["numeric_summary"] = {
                "min": safe_float(valid_values.min() if len(valid_values) else np.nan),
                "max": safe_float(valid_values.max() if len(valid_values) else np.nan),
                "mean": safe_float(valid_values.mean() if len(valid_values) else np.nan),
                "std": safe_float(valid_values.std(ddof=0) if len(valid_values) else np.nan),
                "zero_pct": zero_pct,
            }
        else:
            value_counts = series_sample.astype("string").fillna("<NA>").value_counts().head(10)
            column_profile["categorical_summary"] = {
                "n_unique_sample": int(series_sample.nunique(dropna=False)),
                "top_10_sample": [{"value": str(idx), "count": int(cnt)} for idx, cnt in value_counts.items()],
            }

        profile[column] = column_profile
    return profile


def detect_labels(df_full: pd.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    label_cols = [col for col in spec.label_cols if col in df_full.columns]
    analysis: dict[str, Any] = {
        "detected_label_columns": label_cols,
        "binary_label_column": spec.binary_col if spec.binary_col in df_full.columns else None,
        "multiclass_label_column": spec.multiclass_col if spec.multiclass_col in df_full.columns else None,
        "distributions": {},
    }

    for col in label_cols:
        counts = df_full[col].astype("string").fillna("<NA>").value_counts(dropna=False)
        total = max(len(df_full), 1)
        analysis["distributions"][col] = [
            {"class": str(name), "count": int(count), "pct": round(float(count) / total * 100.0, 6)}
            for name, count in counts.items()
        ]

    return analysis


def detect_timestamps(df_sample: pd.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for col in spec.possible_timestamp_cols:
        if col not in df_sample.columns:
            continue
        parsed = pd.to_datetime(df_sample[col], errors="coerce", utc=True)
        success_rate = float(parsed.notna().mean() * 100.0)
        results[col] = {
            "parse_success_pct": round(success_rate, 6),
            "non_null_sample": int(df_sample[col].notna().sum()),
        }

    detected = None
    if results:
        detected = max(results, key=lambda key: results[key]["parse_success_pct"])
    return {"detected_timestamp_column": detected, "candidates": results}


def detect_entity_columns(df_full: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    column_map = {col.lower(): col for col in df_full.columns}
    detected: list[str] = []
    for candidate in spec.possible_entity_cols:
        if candidate.lower() in column_map:
            detected.append(column_map[candidate.lower()])
    return sorted(set(detected), key=str.lower)


def detect_split_columns(df_full: pd.DataFrame) -> dict[str, Any]:
    split_candidates = ["split", "is_train", "train", "set", "dataset"]
    lower_map = {col.lower(): col for col in df_full.columns}
    detected = [lower_map[c] for c in split_candidates if c in lower_map]
    return {
        "detected_split_columns": detected,
        "has_explicit_split_column": bool(detected),
        "uses_source_file_fallback": "_source_file" in df_full.columns,
    }


def suggest_column_types(df_full: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for col in df_full.columns:
        if col == "_source_file":
            continue
        if pd.api.types.is_numeric_dtype(df_full[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return categorical, numeric


def to_yaml_value(value: Any, indent: int = 0) -> str:
    spacing = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{spacing}{k}:")
                lines.append(to_yaml_value(v, indent + 2))
            else:
                lines.append(f"{spacing}{k}: {json.dumps(v)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{spacing}-")
                lines.append(to_yaml_value(item, indent + 2))
            else:
                lines.append(f"{spacing}- {json.dumps(item)}")
        return "\n".join(lines)
    return f"{spacing}{json.dumps(value)}"


def build_dataset_report(
    spec: DatasetSpec,
    paths: list[Path],
    df_full: pd.DataFrame,
    df_sample: pd.DataFrame,
    sampled: bool,
) -> dict[str, Any]:
    labels = detect_labels(df_full, spec)
    timestamps = detect_timestamps(df_sample, spec)
    entity_cols = detect_entity_columns(df_full, spec)
    split_info = detect_split_columns(df_full)
    categorical_cols, numeric_cols = suggest_column_types(df_full)

    return {
        "dataset": spec.name,
        "files": [str(path) for path in paths],
        "shape": {
            "rows": int(len(df_full)),
            "cols": int(df_full.shape[1]),
            "stats_based_on_sample": sampled,
            "sample_rows": int(len(df_sample)),
        },
        "columns": list(df_full.columns),
        "column_profiles": profile_columns(df_full, df_sample),
        "label_analysis": labels,
        "timestamp_analysis": timestamps,
        "entity_column_candidates": entity_cols,
        "split_detection": split_info,
        "suggested_categorical_columns": categorical_cols,
        "suggested_numeric_columns": numeric_cols,
    }


def write_json_report(out_dir: Path, dataset_name: str, report: dict[str, Any]) -> Path:
    output_path = out_dir / f"{dataset_name}_profile.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def write_schema_yaml(out_dir: Path, dataset_name: str, report: dict[str, Any]) -> Path:
    schema = {
        "dataset": dataset_name,
        "label_columns": report["label_analysis"]["detected_label_columns"],
        "binary_label_column": report["label_analysis"]["binary_label_column"],
        "multiclass_label_column": report["label_analysis"]["multiclass_label_column"],
        "detected_timestamp_column": report["timestamp_analysis"]["detected_timestamp_column"],
        "detected_entity_columns": report["entity_column_candidates"],
        "categorical_columns_suggestion": report["suggested_categorical_columns"],
        "numeric_columns_suggestion": report["suggested_numeric_columns"],
    }
    output_path = out_dir / f"{dataset_name}_schema.yaml"
    output_path.write_text(to_yaml_value(schema) + "\n", encoding="utf-8")
    return output_path


def write_markdown_summary(out_dir: Path, reports: list[dict[str, Any]]) -> Path:
    lines = ["# Dataset Inspection Summary", ""]
    for report in reports:
        shape = report["shape"]
        labels = report["label_analysis"]["detected_label_columns"]
        timestamp = report["timestamp_analysis"]["detected_timestamp_column"]
        entity_cols = report["entity_column_candidates"]
        split_info = report["split_detection"]

        lines.extend(
            [
                f"## {report['dataset']}",
                "",
                f"- Files: {', '.join(report['files'])}",
                f"- Rows: {shape['rows']} | Columns: {shape['cols']}",
                f"- Stats sampled: {shape['stats_based_on_sample']} (sample_rows={shape['sample_rows']})",
                f"- Detected labels: {', '.join(labels) if labels else 'None'}",
                f"- Candidate entity columns: {', '.join(entity_cols) if entity_cols else 'None'}",
                f"- Candidate timestamp column: {timestamp if timestamp else 'None'}",
                (
                    f"- Explicit split columns: {', '.join(split_info['detected_split_columns'])}"
                    if split_info["detected_split_columns"]
                    else "- Explicit split columns: None (using _source_file fallback when available)"
                ),
                "",
            ]
        )

    output_path = out_dir / "README.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def load_dataset(spec: DatasetSpec, paths: list[Path]) -> pd.DataFrame:
    if spec.name == "kdd":
        return load_kdd(paths)
    return load_csv_with_source(paths, spec.name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    resolved_data_dir = resolve_data_dir(args.data_dir)
    LOGGER.info("Using data_dir: %s", resolved_data_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict[str, Any]] = []
    for spec in DATASET_SPECS.values():
        LOGGER.info("Inspecting dataset: %s", spec.name)
        paths = ensure_files(spec, resolved_data_dir)
        df_full = load_dataset(spec, paths)
        df_sample, sampled = sample_frame(df_full, args.sample_rows, args.seed)
        report = build_dataset_report(spec, paths, df_full, df_sample, sampled)

        json_path = write_json_report(args.out_dir, spec.name, report)
        yaml_path = write_schema_yaml(args.out_dir, spec.name, report)
        LOGGER.info("Wrote %s", json_path)
        LOGGER.info("Wrote %s", yaml_path)
        all_reports.append(report)

    md_path = write_markdown_summary(args.out_dir, all_reports)
    LOGGER.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
