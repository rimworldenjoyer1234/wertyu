from __future__ import annotations

from typing import Any

import numpy as np

from src.wertuy.graphs.ops import Adjacency


def _union_find_components(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[int, np.ndarray]:
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in zip(src.tolist(), dst.tolist()):
        union(int(a), int(b))

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return int(labels.max() + 1 if n > 0 else 0), labels.astype(np.int32)


def _feature_nbytes(X) -> int:
    arr = np.asarray(X)
    return int(arr.nbytes)


def compute_graph_metrics(
    A: Adjacency,
    X,
    feature_time_sec: float,
    knn_time_sec: float,
    ops_time_sec: float,
    metrics_time_sec: float,
) -> dict[str, Any]:
    n = int(A.num_nodes)
    e = int(A.nnz)

    out_deg = np.bincount(A.src, minlength=n)
    in_deg = np.bincount(A.dst, minlength=n)

    density = float(e) / float(max(n * (n - 1), 1))

    # Undirected view for components / degree stats.
    s2 = np.concatenate([A.src, A.dst]).astype(np.int32)
    d2 = np.concatenate([A.dst, A.src]).astype(np.int32)
    keys_u = s2.astype(np.int64) * n + d2.astype(np.int64)
    uniq_u = np.unique(keys_u)
    su = (uniq_u // n).astype(np.int32)
    du = (uniq_u % n).astype(np.int32)

    n_comp, labels = _union_find_components(n, su, du)
    if n > 0:
        comp_sizes = np.bincount(labels)
        lcc_fraction = float(comp_sizes.max()) / float(n)
    else:
        lcc_fraction = 0.0

    deg_u = np.bincount(np.concatenate([su, du]), minlength=n)
    isolated = int(np.sum(deg_u == 0)) if n > 0 else 0

    keys = A.src.astype(np.int64) * n + A.dst.astype(np.int64)
    rev_keys = A.dst.astype(np.int64) * n + A.src.astype(np.int64)
    mutual = np.intersect1d(keys, rev_keys, assume_unique=False)
    reciprocity = float(mutual.shape[0]) / float(max(e, 1))

    if n > 0:
        deg_stats = {
            "min": float(np.min(deg_u)),
            "median": float(np.median(deg_u)),
            "max": float(np.max(deg_u)),
            "p95": float(np.percentile(deg_u, 95)),
        }
    else:
        deg_stats = {"min": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0}

    edge_index_bytes = int(2 * e * 4)
    x_bytes = _feature_nbytes(X)

    return {
        "N": n,
        "E": e,
        "mean_out_degree": float(out_deg.mean()) if n > 0 else 0.0,
        "mean_in_degree": float(in_deg.mean()) if n > 0 else 0.0,
        "density": density,
        "num_components": int(n_comp),
        "LCC_fraction": lcc_fraction,
        "isolated_nodes_count": isolated,
        "isolated_nodes_pct": float(isolated) * 100.0 / float(max(n, 1)),
        "reciprocity": reciprocity,
        "degree_stats": deg_stats,
        "feature_time_sec": feature_time_sec,
        "knn_time_sec": knn_time_sec,
        "ops_time_sec": ops_time_sec,
        "metrics_time_sec": metrics_time_sec,
        "edge_index_bytes": edge_index_bytes,
        "X_bytes": x_bytes,
        "total_bytes": int(edge_index_bytes + x_bytes),
    }
