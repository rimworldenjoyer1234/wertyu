from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check fast graph outputs")
    p.add_argument("--graphs_dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--datasets", nargs="+", default=["kdd", "unsw-nb15", "ton-iot"])
    return p.parse_args()


def pick_one_graph(dataset_dir: Path) -> Path:
    candidates = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("base_sim_knn_")])
    if not candidates:
        raise FileNotFoundError(f"No base_sim_knn graph folders found in {dataset_dir}")
    return candidates[0]


def main() -> None:
    args = parse_args()

    for ds in args.datasets:
        ds_dir = args.graphs_dir / ds
        if not ds_dir.exists():
            print(f"[WARN] Missing dataset dir: {ds_dir}")
            continue

        gdir = pick_one_graph(ds_dir)
        edge_index = np.load(gdir / "edge_index.npy")
        y_bin = np.load(gdir / "y_bin.npy")
        split = np.load(gdir / "split_masks.npz")
        cfg = json.loads((gdir / "config.json").read_text(encoding="utf-8"))

        n = y_bin.shape[0]
        if edge_index.shape[0] != 2:
            raise AssertionError(f"{gdir}: edge_index first dim must be 2")
        if edge_index.shape[1] > 0:
            if edge_index.min() < 0 or edge_index.max() >= n:
                raise AssertionError(f"{gdir}: edge indices out of range [0,{n})")

        uniq = set(np.unique(y_bin).tolist())
        if not uniq.issubset({0, 1}):
            raise AssertionError(f"{gdir}: y_bin non-binary values: {sorted(uniq)}")

        tr = set(split["train_idx"].tolist())
        va = set(split["val_idx"].tolist())
        te = set(split["test_idx"].tolist())
        if tr & va or tr & te or va & te:
            raise AssertionError(f"{gdir}: split index overlap detected")

        k = int(cfg.get("k", 0))
        e = edge_index.shape[1]
        expected = n * k
        # allow deviations from duplicates or tiny-n edge effects
        if expected > 0:
            ratio = e / expected
            if ratio < 0.85 or ratio > 1.05:
                raise AssertionError(f"{gdir}: E not close to N*k (E={e}, N*k={expected}, ratio={ratio:.3f})")

        print(f"[OK] {ds} :: {gdir.name} :: N={n} E={e} k={k}")


if __name__ == "__main__":
    main()
