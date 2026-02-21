from __future__ import annotations

import argparse
from pathlib import Path

from src.wertuy.data.loader import load_metadata, load_processed_dataset


DATASETS = ("kdd", "unsw-nb15", "ton-iot")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Light sanity checks for processed datasets")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for dataset in DATASETS:
        md = load_metadata(dataset, args.processed_dir)
        feature_cols = md.get("feature_columns", [])
        label_cols = set(md.get("label_columns", {}).get("all_label_columns_present", []))

        if not feature_cols:
            raise AssertionError(f"{dataset}: feature_columns is empty")

        overlap = set(feature_cols).intersection(label_cols)
        if overlap:
            raise AssertionError(f"{dataset}: leakage detected, overlap={sorted(overlap)}")

        for split in SPLITS:
            df = load_processed_dataset(dataset, split, args.processed_dir)
            if "y_bin" not in df.columns:
                raise AssertionError(f"{dataset}/{split}: missing y_bin")
            y_unique = set(df["y_bin"].dropna().astype(int).unique().tolist())
            if not y_unique.issubset({0, 1}):
                raise AssertionError(f"{dataset}/{split}: y_bin has non-binary values {sorted(y_unique)}")

        print(f"[OK] {dataset}: features={len(feature_cols)} labels={sorted(label_cols)}")


if __name__ == "__main__":
    main()
