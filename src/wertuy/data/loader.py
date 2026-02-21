from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_processed_dataset(dataset_name: str, split: str, processed_dir: Path = Path("data/processed")) -> pd.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    dataset_path = processed_dir / dataset_name / f"{split}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed split not found: {dataset_path}")
    return pd.read_parquet(dataset_path)


def load_metadata(dataset_name: str, processed_dir: Path = Path("data/processed")) -> dict:
    metadata_path = processed_dir / dataset_name / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))
