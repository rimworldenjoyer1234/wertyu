from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _undirected_adjacency(num_nodes: int, edge_index: np.ndarray) -> list[set[int]]:
    adj = [set() for _ in range(num_nodes)]
    for i, j in zip(edge_index[0], edge_index[1]):
        ii = int(i)
        jj = int(j)
        adj[ii].add(jj)
        adj[jj].add(ii)
    return adj


def _components_stats(num_nodes: int, edge_index: np.ndarray) -> tuple[int, float]:
    if num_nodes == 0:
        return 0, 0.0
    adj = _undirected_adjacency(num_nodes, edge_index)
    seen = [False] * num_nodes
    sizes: list[int] = []

    for start in range(num_nodes):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        size = 0
        while q:
            node = q.popleft()
            size += 1
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        sizes.append(size)

    lcc = max(sizes) if sizes else 0
    return len(sizes), float(lcc) / float(num_nodes)


def _degree_stats(num_nodes: int, edge_index: np.ndarray) -> dict[str, float]:
    if num_nodes == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0}
    deg = np.zeros(num_nodes, dtype=np.int64)
    for i, j in zip(edge_index[0], edge_index[1]):
        ii = int(i)
        jj = int(j)
        deg[ii] += 1
        if ii != jj:
            deg[jj] += 1
    return {
        "min": float(np.min(deg)),
        "median": float(np.median(deg)),
        "max": float(np.max(deg)),
        "p95": float(np.percentile(deg, 95)),
    }


def compute_graph_metrics(
    num_nodes: int,
    edge_index: np.ndarray,
    directedness: str,
    preprocessing_time_sec: float,
    edge_construction_time_sec: float,
    node_features_bytes: int,
) -> dict[str, Any]:
    e = int(edge_index.shape[1])
    n = int(num_nodes)

    out_deg = np.bincount(edge_index[0], minlength=n) if n > 0 else np.array([], dtype=np.int64)
    in_deg = np.bincount(edge_index[1], minlength=n) if n > 0 else np.array([], dtype=np.int64)
    deg_u = out_deg + in_deg

    if n > 1:
        if directedness == "directed":
            density = float(e) / float(n * (n - 1))
        else:
            density = float(2 * e) / float(n * (n - 1))
    else:
        density = 0.0

    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    recip = 0
    for i, j in pairs:
        if (j, i) in pairs:
            recip += 1
    reciprocity = float(recip) / float(max(len(pairs), 1))

    num_components, lcc_frac = _components_stats(n, edge_index)
    isolated = int(np.sum(deg_u == 0)) if n > 0 else 0
    edge_index_bytes = int(edge_index.nbytes)

    stats = {
        "N": n,
        "E": e,
        "mean_out_degree": float(np.mean(out_deg)) if n > 0 else 0.0,
        "mean_in_degree": float(np.mean(in_deg)) if n > 0 else 0.0,
        "mean_degree_undirected_view": float(np.mean(deg_u)) if n > 0 else 0.0,
        "density": density,
        "num_components": int(num_components),
        "LCC_fraction": float(lcc_frac),
        "isolated_nodes_count": isolated,
        "isolated_nodes_pct": float(isolated) / float(max(n, 1)) * 100.0,
        "reciprocity": reciprocity,
        "degree_stats": _degree_stats(n, edge_index),
        "preprocessing_time_sec": preprocessing_time_sec,
        "edge_construction_time_sec": edge_construction_time_sec,
        "edge_index_bytes": edge_index_bytes,
        "node_features_bytes": int(node_features_bytes),
        "total_bytes": int(edge_index_bytes + node_features_bytes),
    }
    return stats
