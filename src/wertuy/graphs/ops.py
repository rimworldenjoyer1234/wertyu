from __future__ import annotations

import numpy as np


def dedupe_edges(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return edge_index.astype(np.int64)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    arr = np.array(sorted(pairs), dtype=np.int64)
    return arr.T if arr.size else np.empty((2, 0), dtype=np.int64)


def apply_directedness(edge_index: np.ndarray, mode: str) -> np.ndarray:
    if mode not in {"directed", "sym", "mutual"}:
        raise ValueError(f"Unsupported directedness mode: {mode}")
    edge_index = dedupe_edges(edge_index)
    if mode == "directed":
        return edge_index

    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if mode == "sym":
        sym_pairs = set(pairs)
        sym_pairs.update((j, i) for i, j in pairs)
        arr = np.array(sorted(sym_pairs), dtype=np.int64)
        return arr.T if arr.size else np.empty((2, 0), dtype=np.int64)

    mutual_pairs = {(i, j) for (i, j) in pairs if (j, i) in pairs}
    arr = np.array(sorted(mutual_pairs), dtype=np.int64)
    return arr.T if arr.size else np.empty((2, 0), dtype=np.int64)


def apply_self_loops(edge_index: np.ndarray, num_nodes: int, policy: str) -> np.ndarray:
    if policy not in {"none", "all", "isolated_only"}:
        raise ValueError(f"Unsupported self-loop policy: {policy}")

    edge_index = dedupe_edges(edge_index)
    if policy == "none":
        return edge_index

    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if policy == "all":
        for n in range(num_nodes):
            pairs.add((n, n))
    else:
        deg = np.zeros(num_nodes, dtype=np.int64)
        for i, j in pairs:
            deg[i] += 1
            if i != j:
                deg[j] += 1
        isolated = np.where(deg == 0)[0].tolist()
        for n in isolated:
            pairs.add((n, n))

    arr = np.array(sorted(pairs), dtype=np.int64)
    return arr.T if arr.size else np.empty((2, 0), dtype=np.int64)
