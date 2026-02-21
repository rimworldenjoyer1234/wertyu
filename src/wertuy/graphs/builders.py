from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


DATASET_ENTITY_COLUMNS: dict[str, list[str]] = {
    "kdd": ["protocol_type", "service", "flag"],
    "unsw-nb15": ["proto", "service", "state"],
    "ton-iot": ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "service"],
}


def build_similarity_knn_graph(
    x: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> np.ndarray:
    if len(x) == 0:
        return np.empty((2, 0), dtype=np.int64)
    n_neighbors = min(k + 1, len(x))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(x)
    nbr_idx = model.kneighbors(return_distance=False)

    src = np.repeat(np.arange(len(x), dtype=np.int64), n_neighbors - 1)
    dst = nbr_idx[:, 1:].reshape(-1).astype(np.int64)
    return np.vstack([src, dst])


def _wire_group_limited(indices: list[int], per_node_cap: int, rng: random.Random) -> list[tuple[int, int]]:
    if len(indices) <= 1:
        return []
    if len(indices) == 2:
        a, b = indices
        return [(a, b), (b, a)]

    edges: list[tuple[int, int]] = []
    shuffled = indices[:]
    rng.shuffle(shuffled)

    for idx, node in enumerate(shuffled):
        neighbors = []
        for hop in range(1, min(per_node_cap, len(shuffled) - 1) + 1):
            neighbors.append(shuffled[(idx + hop) % len(shuffled)])
        for nb in neighbors:
            if nb != node:
                edges.append((node, nb))
    return edges


def build_relational_shared_entity_graph(
    df,
    entity_columns: Sequence[str],
    per_node_cap: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    rng = random.Random(seed)
    missing = [c for c in entity_columns if c not in df.columns]
    used_columns = [c for c in entity_columns if c in df.columns]

    group_map: dict[str, dict[str, list[int]]] = {col: defaultdict(list) for col in used_columns}
    for idx, row in df.reset_index(drop=True).iterrows():
        for col in used_columns:
            value = row[col]
            if value is None:
                continue
            key = str(value)
            if key == "nan" or key == "<NA>":
                continue
            group_map[col][key].append(idx)

    edges: set[tuple[int, int]] = set()
    for col in used_columns:
        for _, indices in group_map[col].items():
            if len(indices) <= 1:
                continue
            for edge in _wire_group_limited(indices, per_node_cap=per_node_cap, rng=rng):
                edges.add(edge)

    arr = np.array(sorted(edges), dtype=np.int64)
    edge_index = arr.T if arr.size else np.empty((2, 0), dtype=np.int64)

    warnings = []
    for col in missing:
        warnings.append(f"Entity column '{col}' missing; skipped")
    if not used_columns:
        warnings.append("No entity columns available for relational graph")

    return edge_index, warnings
