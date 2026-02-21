from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = ("kdd", "unsw-nb15", "ton-iot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check generated graph bundles")
    parser.add_argument("--graphs_dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--max_graphs_per_dataset", type=int, default=2)
    return parser.parse_args()


def list_graph_dirs(dataset_dir: Path, limit: int) -> list[Path]:
    candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
    return sorted(candidates)[:limit]


def has_self_loops(edge_index: np.ndarray) -> bool:
    return bool(np.any(edge_index[0] == edge_index[1]))


def reciprocity(edge_index: np.ndarray) -> float:
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if not pairs:
        return 0.0
    recip = sum(1 for (i, j) in pairs if (j, i) in pairs)
    return recip / len(pairs)


def main() -> None:
    args = parse_args()

    for dataset in DATASETS:
        dataset_dir = args.graphs_dir / dataset
        if not dataset_dir.exists():
            print(f"[WARN] Missing dataset graph dir: {dataset_dir}")
            continue

        graph_dirs = list_graph_dirs(dataset_dir, args.max_graphs_per_dataset)
        for gdir in graph_dirs:
            edge_index = np.load(gdir / "edge_index.npy")
            node_features = np.load(gdir / "node_features.npy")
            y_bin = np.load(gdir / "y_bin.npy")
            split = np.load(gdir / "split_masks.npz")
            stats = json.loads((gdir / "graph_stats.json").read_text(encoding="utf-8"))
            config = json.loads((gdir / "config.json").read_text(encoding="utf-8"))

            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise AssertionError(f"{gdir}: edge_index shape invalid: {edge_index.shape}")

            n = node_features.shape[0]
            if y_bin.shape[0] != n:
                raise AssertionError(f"{gdir}: y_bin length mismatch")
            if edge_index.shape[1] > 0:
                if edge_index.min() < 0 or edge_index.max() >= n:
                    raise AssertionError(f"{gdir}: edge indices out of bounds for N={n}")

            train_idx = split["train_idx"]
            val_idx = split["val_idx"]
            test_idx = split["test_idx"]

            if len(set(train_idx).intersection(set(val_idx))) > 0:
                raise AssertionError(f"{gdir}: train/val overlap")
            if len(set(train_idx).intersection(set(test_idx))) > 0:
                raise AssertionError(f"{gdir}: train/test overlap")
            if len(set(val_idx).intersection(set(test_idx))) > 0:
                raise AssertionError(f"{gdir}: val/test overlap")

            expected_total = n if config["graph_scope"] == "transductive" else len(train_idx) + len(val_idx)
            covered = len(set(train_idx).union(set(val_idx)).union(set(test_idx)))
            if covered != expected_total:
                raise AssertionError(f"{gdir}: split index coverage mismatch ({covered} vs {expected_total})")

            if config["self_loops"] == "none" and has_self_loops(edge_index):
                raise AssertionError(f"{gdir}: found self loops but policy is none")
            if config["self_loops"] == "all":
                loops = set(edge_index[0][edge_index[0] == edge_index[1]].tolist())
                if len(loops) < n:
                    raise AssertionError(f"{gdir}: missing all-node self loops")

            rec = reciprocity(edge_index)
            if config["directedness"] == "sym" and rec < 0.99 and edge_index.shape[1] > 0:
                raise AssertionError(f"{gdir}: low reciprocity for sym graph: {rec:.3f}")
            if config["directedness"] == "mutual" and abs(rec - 1.0) > 1e-6 and edge_index.shape[1] > 0:
                raise AssertionError(f"{gdir}: mutual graph reciprocity not 1.0")

            if abs(stats.get("reciprocity", rec) - rec) > 0.02:
                raise AssertionError(f"{gdir}: reciprocity in stats inconsistent")

            print(f"[OK] {dataset} :: {gdir.name} :: N={n} E={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
