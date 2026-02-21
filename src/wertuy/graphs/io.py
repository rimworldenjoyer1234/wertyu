from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_graph_bundle(
    out_dir: Path,
    edge_index: np.ndarray,
    node_features: np.ndarray,
    y_bin: np.ndarray,
    split_indices: dict[str, np.ndarray],
    graph_stats: dict[str, Any],
    config: dict[str, Any],
    y_multi: np.ndarray | None = None,
    y_multi_mapping: dict[str, int] | None = None,
    node_id_map: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "edge_index.npy", edge_index.astype(np.int64))
    np.save(out_dir / "node_features.npy", node_features.astype(np.float32))
    np.save(out_dir / "y_bin.npy", y_bin.astype(np.int8))

    if y_multi is not None:
        np.save(out_dir / "y_multi.npy", y_multi.astype(np.int32))
        if y_multi_mapping is not None:
            (out_dir / "y_multi_mapping.json").write_text(json.dumps(y_multi_mapping, indent=2), encoding="utf-8")

    np.savez(
        out_dir / "split_masks.npz",
        train_idx=split_indices.get("train", np.array([], dtype=np.int64)),
        val_idx=split_indices.get("val", np.array([], dtype=np.int64)),
        test_idx=split_indices.get("test", np.array([], dtype=np.int64)),
    )
    if node_id_map is not None:
        np.save(out_dir / "node_id_map.npy", node_id_map.astype(np.int64))

    (out_dir / "graph_stats.json").write_text(json.dumps(graph_stats, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def config_matches(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    return existing == config


def save_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)
