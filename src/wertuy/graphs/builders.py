from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np


DATASET_ENTITY_COLUMNS: dict[str, list[str]] = {
    "kdd": ["protocol_type", "service", "flag"],
    "unsw-nb15": ["proto", "service", "state"],
    "ton-iot": ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "service"],
}


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _knn_indices_cosine(x: np.ndarray, k: int, chunk_size: int = 2048) -> np.ndarray:
    x = _normalize_rows(x.astype(np.float32))
    n = x.shape[0]
    k_eff = min(k, max(n - 1, 0))
    if k_eff <= 0:
        return np.empty((n, 0), dtype=np.int64)

    all_idx = np.empty((n, k_eff), dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = x[start:end] @ x.T
        row_ids = np.arange(start, end)
        sims[np.arange(end - start), row_ids] = -np.inf
        top = np.argpartition(-sims, kth=min(k_eff - 1, sims.shape[1] - 1), axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        all_idx[start:end] = np.take_along_axis(top, order, axis=1)
    return all_idx


def _knn_indices_euclidean(x: np.ndarray, k: int, chunk_size: int = 1024) -> np.ndarray:
    x = x.astype(np.float32)
    n = x.shape[0]
    k_eff = min(k, max(n - 1, 0))
    if k_eff <= 0:
        return np.empty((n, 0), dtype=np.int64)

    x_sq = np.sum(x * x, axis=1)
    all_idx = np.empty((n, k_eff), dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = x[start:end]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_sq + x_sq.reshape(1, -1) - 2.0 * (q @ x.T)
        d2 = np.maximum(d2, 0.0)
        row_ids = np.arange(start, end)
        d2[np.arange(end - start), row_ids] = np.inf
        top = np.argpartition(d2, kth=min(k_eff - 1, d2.shape[1] - 1), axis=1)[:, :k_eff]
        top_d = np.take_along_axis(d2, top, axis=1)
        order = np.argsort(top_d, axis=1)
        all_idx[start:end] = np.take_along_axis(top, order, axis=1)
    return all_idx


def build_similarity_knn_graph(
    x: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> np.ndarray:
    if len(x) == 0:
        return np.empty((2, 0), dtype=np.int64)

    if metric == "cosine":
        nbr_idx = _knn_indices_cosine(x, k)
    elif metric == "euclidean":
        nbr_idx = _knn_indices_euclidean(x, k)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if nbr_idx.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)

    src = np.repeat(np.arange(len(x), dtype=np.int64), nbr_idx.shape[1])
    dst = nbr_idx.reshape(-1).astype(np.int64)
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
            if key in {"nan", "<NA>", "None"}:
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
