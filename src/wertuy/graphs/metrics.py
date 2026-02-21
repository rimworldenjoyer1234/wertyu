from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def feature_nbytes(X) -> int:
    if sparse.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    arr = np.asarray(X)
    return int(arr.nbytes)


def compute_graph_metrics(
    A: sparse.csr_matrix,
    X,
    feature_time_sec: float,
    knn_time_sec: float,
    ops_time_sec: float,
    metrics_time_sec: float,
) -> dict[str, Any]:
    n = int(A.shape[0])
    e = int(A.nnz)

    out_deg = np.asarray(A.getnnz(axis=1)).reshape(-1)
    in_deg = np.asarray(A.getnnz(axis=0)).reshape(-1)

    density = float(e) / float(max(n * (n - 1), 1))

    U = A.maximum(A.T).tocsr()
    n_comp, labels = connected_components(U, directed=False, return_labels=True)
    if n > 0:
        comp_sizes = np.bincount(labels)
        lcc_fraction = float(comp_sizes.max()) / float(n)
    else:
        lcc_fraction = 0.0
    deg_u = np.asarray(U.getnnz(axis=1)).reshape(-1)
    isolated = int(np.sum(deg_u == 0)) if n > 0 else 0

    R = A.minimum(A.T).tocsr()
    reciprocity = float(R.nnz) / float(max(e, 1))

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
    x_bytes = feature_nbytes(X)

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
