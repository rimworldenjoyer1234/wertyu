from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.data.dataset_specs import KDD_FEATURE_COLUMNS

LOGGER = logging.getLogger("prepare_datasets")

DATASET_NAMES = ("kdd", "unsw-nb15", "ton-iot")
BOOL_HINT_COLUMNS = {
    "land",
    "logged_in",
    "root_shell",
    "is_host_login",
    "is_guest_login",
    "label",
    "y_bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw datasets into split parquet + metadata artifacts.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--schema_dir", type=Path, default=Path("reports/datasets"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--ton_split_mode",
        choices=("random_stratified", "entity_disjoint_src_ip"),
        default="random_stratified",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    return parser.parse_args()


def resolve_dir(requested: Path) -> Path:
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append((REPO_ROOT / requested).resolve())
        candidates.append((REPO_ROOT.parent / requested).resolve())
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve directory. Tried:\n  - {tried}")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"')


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse a constrained YAML subset used by schema files produced in this repo."""
    root: dict[str, Any] = {}
    current_key: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("  - "):
                if current_key is None:
                    continue
                root.setdefault(current_key, [])
                root[current_key].append(parse_scalar(line[4:]))
                continue
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                current_key = key
                if value == "":
                    root[key] = []
                else:
                    root[key] = parse_scalar(value)
    return root


def dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    space = " " * indent
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{space}{key}:")
            for item in value:
                lines.append(f"{space}  - {json.dumps(item)}")
        elif isinstance(value, dict):
            lines.append(f"{space}{key}:")
            lines.append(dump_yaml(value, indent + 2))
        else:
            lines.append(f"{space}{key}: {json.dumps(value)}")
    return "\n".join(lines)


def read_kdd(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    train_path = data_dir / "kdd/KDDTrain+.txt"
    test_path = data_dir / "kdd/KDDTest+.txt"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing KDD file: {path}")

    train = pd.read_csv(train_path, header=None, sep=",", low_memory=False)
    test = pd.read_csv(test_path, header=None, sep=",", low_memory=False)
    if train.shape[1] != len(KDD_FEATURE_COLUMNS) or test.shape[1] != len(KDD_FEATURE_COLUMNS):
        raise ValueError(
            "KDD column count mismatch. Expected 43 columns for both KDDTrain+.txt and KDDTest+.txt"
        )

    train.columns = KDD_FEATURE_COLUMNS
    test.columns = KDD_FEATURE_COLUMNS
    train["_split_source"] = "train_file"
    test["_split_source"] = "test_file"

    df = pd.concat([train, test], ignore_index=True)
    df["label"] = df["label"].astype("string").str.strip().str.rstrip(".").str.lower()
    df["y_multi"] = df["label"].astype("string")
    df["y_bin"] = (df["y_multi"] != "normal").astype("int8")
    return df, [str(train_path), str(test_path)]


def read_unsw(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    train_path = data_dir / "unsw-nb15/UNSW_NB15_training-set.csv"
    test_path = data_dir / "unsw-nb15/UNSW_NB15_testing-set.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing UNSW file: {path}")

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    train["_split_source"] = "train_file"
    test["_split_source"] = "test_file"

    df = pd.concat([train, test], ignore_index=True)
    if "attack_cat" in df.columns:
        df["y_multi"] = df["attack_cat"].astype("string").str.strip()
    if "label" in df.columns:
        df["y_bin"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype("int8")
    else:
        raise ValueError("UNSW-NB15 requires 'label' column for y_bin generation.")
    return df, [str(train_path), str(test_path)]


def read_ton(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = data_dir / "ton-iot/train_test_network.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing TON-IoT file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["_split_source"] = "single_file"
    if "label" in df.columns:
        df["y_bin"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype("int8")
    else:
        raise ValueError("TON-IoT requires 'label' column for y_bin generation.")
    if "type" in df.columns:
        df["y_multi"] = df["type"].astype("string").str.strip()
    return df, [str(path)]


def stratified_indices(y: pd.Series, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for cls in y.dropna().unique():
        cls_idx = y.index[y == cls].to_numpy()
        if len(cls_idx) == 0:
            continue
        n_take = int(round(len(cls_idx) * ratio))
        if ratio > 0 and n_take == 0 and len(cls_idx) > 0:
            n_take = 1
        n_take = min(n_take, len(cls_idx))
        if n_take > 0:
            picked.extend(rng.choice(cls_idx, size=n_take, replace=False).tolist())
    return np.array(sorted(set(picked)))


def split_train_val_from_pool(training_pool: pd.DataFrame, y_col: str, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_idx = stratified_indices(training_pool[y_col], val_ratio, seed)
    val = training_pool.loc[val_idx].copy()
    train = training_pool.drop(index=val_idx).copy()
    return train, val


def split_ton(
    df: pd.DataFrame,
    mode: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if mode == "entity_disjoint_src_ip" and "src_ip" not in df.columns:
        warnings.append("TON entity-disjoint requested but src_ip is missing; falling back to random_stratified")
        mode = "random_stratified"

    if mode == "entity_disjoint_src_ip":
        entities = df["src_ip"].astype("string").dropna().unique().tolist()
        rng = random.Random(seed)
        rng.shuffle(entities)
        n_test_entities = max(1, int(round(len(entities) * test_ratio)))
        test_entities = set(entities[:n_test_entities])
        test_mask = df["src_ip"].astype("string").isin(test_entities)
        test = df[test_mask].copy()
        remaining = df[~test_mask].copy()
        train, val = split_train_val_from_pool(remaining, "y_bin", val_ratio, seed)
        return train, val, test, mode

    test_idx = stratified_indices(df["y_bin"], test_ratio, seed)
    test = df.loc[test_idx].copy()
    train_pool = df.drop(index=test_idx).copy()
    train, val = split_train_val_from_pool(train_pool, "y_bin", val_ratio, seed)
    return train, val, test, mode


def class_distribution(df: pd.DataFrame, col: str) -> dict[str, dict[str, float | int]]:
    if col not in df.columns:
        return {}
    total = max(len(df), 1)
    counts = df[col].astype("string").fillna("<NA>").value_counts(dropna=False)
    return {k: {"count": int(v), "pct": round(float(v) * 100.0 / total, 6)} for k, v in counts.items()}


def split_distribution(splits: dict[str, pd.DataFrame], col: str) -> dict[str, dict[str, dict[str, float | int]]]:
    return {split_name: class_distribution(split_df, col) for split_name, split_df in splits.items()}


def convert_types(df: pd.DataFrame, categorical_suggestion: Iterable[str], numeric_suggestion: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_suggestion:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_suggestion:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("__MISSING__").astype("category")

    for col in df.columns:
        if col in BOOL_HINT_COLUMNS:
            values = pd.to_numeric(df[col], errors="coerce")
            unique_vals = set(values.dropna().astype(int).unique().tolist())
            if unique_vals.issubset({0, 1}):
                df[col] = values.fillna(0).astype("int8")
    return df


def drop_all_nan_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    return df.drop(columns=drop_cols), drop_cols


def build_column_sets(df: pd.DataFrame, schema: dict[str, Any], warnings: list[str]) -> tuple[list[str], list[str], list[str], list[str], str | None, dict[str, str | None]]:
    binary_label_col = schema.get("binary_label_column")
    multiclass_col = schema.get("multiclass_label_column")
    original_label_cols = schema.get("label_columns", []) or []
    entity_cols = [c for c in (schema.get("detected_entity_columns", []) or []) if c in df.columns]
    timestamp_col = schema.get("detected_timestamp_column")
    if timestamp_col not in df.columns:
        timestamp_col = None

    label_cols = [c for c in ["y_bin", "y_multi", binary_label_col, multiclass_col, *original_label_cols] if c in df.columns]
    label_cols = sorted(set(label_cols), key=str)

    meta_cols = [c for c in ["_split_source", "difficulty_level", timestamp_col, *entity_cols] if c and c in df.columns]
    meta_cols = sorted(set(meta_cols), key=str)

    feature_cols = [c for c in df.columns if c not in set(label_cols) | set(meta_cols)]
    if not feature_cols:
        warnings.append("No feature columns detected after excluding labels/meta columns")

    categorical_cols = [
        c
        for c in (schema.get("categorical_columns_suggestion", []) or [])
        if c in feature_cols
    ]
    numeric_cols = [
        c
        for c in (schema.get("numeric_columns_suggestion", []) or [])
        if c in feature_cols
    ]

    # Fallback inference when schema suggestions are missing/partial.
    remaining_features = [c for c in feature_cols if c not in set(categorical_cols) | set(numeric_cols)]
    for col in remaining_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    label_dict = {
        "y_bin": "y_bin" if "y_bin" in df.columns else None,
        "y_multi": "y_multi" if "y_multi" in df.columns else None,
        "binary_label_column": binary_label_col if binary_label_col in df.columns else None,
        "multiclass_label_column": multiclass_col if multiclass_col in df.columns else None,
    }
    return feature_cols, categorical_cols, numeric_cols, entity_cols, timestamp_col, label_dict


def sanity_checks(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    label_cols: list[str],
    split_mode: str,
) -> dict[str, Any]:
    overlap = sorted(set(feature_cols).intersection(label_cols))
    degenerate = {name: split_df["y_bin"].nunique(dropna=False) <= 1 for name, split_df in splits.items()}
    nan_only_in_features = [c for c in feature_cols if all(split_df[c].isna().all() for split_df in splits.values())]

    report: dict[str, Any] = {
        "no_feature_label_overlap": len(overlap) == 0,
        "feature_label_overlap_columns": overlap,
        "split_y_bin_non_degenerate": {k: (not v) for k, v in degenerate.items()},
        "no_nan_only_feature_columns": len(nan_only_in_features) == 0,
        "nan_only_feature_columns": nan_only_in_features,
    }

    if split_mode == "entity_disjoint_src_ip" and all("src_ip" in s.columns for s in splits.values()):
        tr = set(splits["train"]["src_ip"].astype("string").unique().tolist())
        va = set(splits["val"]["src_ip"].astype("string").unique().tolist())
        te = set(splits["test"]["src_ip"].astype("string").unique().tolist())
        report["src_ip_disjoint_train_test"] = len(tr.intersection(te)) == 0
        report["src_ip_disjoint_val_test"] = len(va.intersection(te)) == 0
    return report


def write_dataset_outputs(
    out_root: Path,
    dataset_name: str,
    schema: dict[str, Any],
    raw_files: list[str],
    split_mode: str,
    seed: int,
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
    entity_cols: list[str],
    timestamp_col: str | None,
    label_dict: dict[str, str | None],
    warnings: list[str],
) -> dict[str, Any]:
    dataset_out = out_root / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_df.to_parquet(dataset_out / f"{split_name}.parquet", index=False)

    label_cols_all = [c for c in [label_dict.get("y_bin"), label_dict.get("y_multi")] if c]
    metadata = {
        "dataset_name": dataset_name,
        "raw_files": raw_files,
        "split_mode": split_mode,
        "seed": seed,
        "split_counts": {k: int(len(v)) for k, v in splits.items()},
        "y_bin_distribution": split_distribution(splits, "y_bin"),
        "y_multi_distribution": split_distribution(splits, "y_multi") if "y_multi" in splits["train"].columns else {},
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "entity_columns": entity_cols,
        "timestamp_column": timestamp_col,
        "label_columns": {
            **label_dict,
            "all_label_columns_present": sorted(
                set([c for c in schema.get("label_columns", []) if c in splits["train"].columns] + label_cols_all)
            ),
        },
        "warnings": warnings,
        "sanity_checks": sanity_checks(splits, feature_cols, label_cols_all + (schema.get("label_columns", []) or []), split_mode),
    }

    (dataset_out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    schema_used = dict(schema)
    schema_used["derived_split_mode"] = split_mode
    schema_used["derived_feature_columns"] = feature_cols
    schema_used["derived_entity_columns"] = entity_cols
    schema_used["derived_timestamp_column"] = timestamp_col
    schema_used["warnings"] = warnings
    (dataset_out / "schema_used.yaml").write_text(dump_yaml(schema_used) + "\n", encoding="utf-8")
    return metadata


def prepare_dataset(
    dataset_name: str,
    data_dir: Path,
    schema_dir: Path,
    out_dir: Path,
    seed: int,
    val_ratio: float,
    ton_split_mode: str,
    test_ratio: float,
) -> dict[str, Any]:
    schema_path = schema_dir / f"{dataset_name}_schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    schema = parse_simple_yaml(schema_path)

    warnings: list[str] = []
    if dataset_name == "kdd":
        df, raw_files = read_kdd(data_dir)
        test = df[df["_split_source"] == "test_file"].copy()
        pool = df[df["_split_source"] == "train_file"].copy()
        train, val = split_train_val_from_pool(pool, "y_bin", val_ratio, seed)
        split_mode = "file_based_with_val_from_train"
    elif dataset_name == "unsw-nb15":
        df, raw_files = read_unsw(data_dir)
        test = df[df["_split_source"] == "test_file"].copy()
        pool = df[df["_split_source"] == "train_file"].copy()
        train, val = split_train_val_from_pool(pool, "y_bin", val_ratio, seed)
        split_mode = "file_based_with_val_from_train"
    elif dataset_name == "ton-iot":
        df, raw_files = read_ton(data_dir)
        train, val, test, split_mode = split_ton(df, ton_split_mode, test_ratio, val_ratio, seed, warnings)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    splits = {"train": train, "val": val, "test": test}

    combined = pd.concat(list(splits.values()), ignore_index=True)
    combined, dropped_all_nan = drop_all_nan_columns(combined)
    if dropped_all_nan:
        warnings.append(f"Dropped all-NaN columns: {', '.join(dropped_all_nan)}")

    feature_cols, categorical_cols, numeric_cols, entity_cols, timestamp_col, label_dict = build_column_sets(
        combined, schema, warnings
    )

    for split_name in list(splits):
        split_df = splits[split_name][[c for c in combined.columns if c in splits[split_name].columns]].copy()
        split_df = convert_types(split_df, categorical_cols, numeric_cols)
        splits[split_name] = split_df

    return write_dataset_outputs(
        out_root=out_dir,
        dataset_name=dataset_name,
        schema=schema,
        raw_files=raw_files,
        split_mode=split_mode,
        seed=seed,
        splits=splits,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        entity_cols=entity_cols,
        timestamp_col=timestamp_col,
        label_dict=label_dict,
        warnings=warnings,
    )


def write_global_readme(out_dir: Path, metadatas: list[dict[str, Any]]) -> None:
    lines = ["# Processed Datasets", ""]
    for md in metadatas:
        lines.extend(
            [
                f"## {md['dataset_name']}",
                f"- Split mode: {md['split_mode']}",
                f"- Split counts: {md['split_counts']}",
                f"- Entity columns: {', '.join(md['entity_columns']) if md['entity_columns'] else 'None'}",
                f"- Timestamp column: {md['timestamp_column']}",
                f"- y_bin train distribution: {md['y_bin_distribution'].get('train', {})}",
                f"- Warnings: {md['warnings'] if md['warnings'] else 'None'}",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    data_dir = resolve_dir(args.data_dir)
    schema_dir = resolve_dir(args.schema_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Using data_dir=%s", data_dir)
    LOGGER.info("Using schema_dir=%s", schema_dir)
    LOGGER.info("Using out_dir=%s", out_dir)

    metadatas: list[dict[str, Any]] = []
    for name in DATASET_NAMES:
        LOGGER.info("Preparing dataset: %s", name)
        md = prepare_dataset(
            dataset_name=name,
            data_dir=data_dir,
            schema_dir=schema_dir,
            out_dir=out_dir,
            seed=args.seed,
            val_ratio=args.val_ratio,
            ton_split_mode=args.ton_split_mode,
            test_ratio=args.test_ratio,
        )
        metadatas.append(md)

    write_global_readme(out_dir, metadatas)
    LOGGER.info("Finished. Outputs at %s", out_dir)


if __name__ == "__main__":
    main()
