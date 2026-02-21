from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_graph_bundle(
    out_dir: Path,
    edge_index: np.ndarray,
    node_features,
    y_bin: np.ndarray,
    split_indices: dict[str, np.ndarray],
    graph_stats: dict[str, Any],
    config: dict[str, Any],
    edge_weight: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "edge_index.npy", edge_index.astype(np.int32))
    if edge_weight is not None:
        np.save(out_dir / "edge_weight.npy", edge_weight.astype(np.float32))

    np.save(out_dir / "node_features.npy", np.asarray(node_features, dtype=np.float32))
    np.save(out_dir / "y_bin.npy", y_bin.astype(np.int8))

    np.savez(
        out_dir / "split_masks.npz",
        train_idx=split_indices.get("train", np.array([], dtype=np.int32)),
        val_idx=split_indices.get("val", np.array([], dtype=np.int32)),
        test_idx=split_indices.get("test", np.array([], dtype=np.int32)),
    )

    (out_dir / "graph_stats.json").write_text(json.dumps(graph_stats, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def config_matches(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    return existing == config


def save_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
