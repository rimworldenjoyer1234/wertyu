from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_sample_indices(
    y_bin: pd.Series,
    split_source: pd.Series,
    max_nodes: int,
    seed: int,
) -> np.ndarray:
    n = len(y_bin)
    if max_nodes is None or max_nodes >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"idx": np.arange(n), "y_bin": y_bin.values, "split": split_source.values})

    sampled: list[int] = []
    grouped = df.groupby(["split", "y_bin"], dropna=False)
    for _, group in grouped:
        frac = len(group) / n
        take = int(round(max_nodes * frac))
        if take == 0 and len(group) > 0:
            take = 1
        take = min(take, len(group))
        sampled.extend(rng.choice(group["idx"].to_numpy(), size=take, replace=False).tolist())

    sampled = sorted(set(sampled))
    if len(sampled) > max_nodes:
        sampled = sampled[:max_nodes]
    elif len(sampled) < max_nodes:
        remaining = sorted(set(range(n)) - set(sampled))
        if remaining:
            add = rng.choice(np.array(remaining), size=min(max_nodes - len(sampled), len(remaining)), replace=False)
            sampled.extend(add.tolist())
            sampled = sorted(set(sampled))

    return np.array(sampled, dtype=np.int64)


def reindex_split_indices(split_labels: pd.Series) -> dict[str, np.ndarray]:
    split_map: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, split in enumerate(split_labels.astype(str).tolist()):
        if split in split_map:
            split_map[split].append(idx)
    return {k: np.array(v, dtype=np.int64) for k, v in split_map.items()}
