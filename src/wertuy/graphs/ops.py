from __future__ import annotations

import numpy as np
from scipy import sparse


def build_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    weights: np.ndarray | None = None,
) -> sparse.csr_matrix:
    if src.size == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    data = weights.astype(np.float32) if weights is not None else np.ones(src.shape[0], dtype=np.float32)
    A = sparse.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    A.sum_duplicates()
    A.data = np.ones_like(A.data, dtype=np.float32)
    A.eliminate_zeros()
    return A


def apply_directedness(A: sparse.csr_matrix, mode: str) -> sparse.csr_matrix:
    if mode == "directed":
        out = A.tocsr()
    elif mode == "sym":
        out = A.maximum(A.T).tocsr()
    elif mode == "mutual":
        out = A.minimum(A.T).tocsr()
    else:
        raise ValueError(f"Unsupported directedness mode: {mode}")
    out.sum_duplicates()
    out.data = np.ones_like(out.data, dtype=np.float32)
    out.eliminate_zeros()
    return out


def apply_self_loops(A: sparse.csr_matrix, policy: str) -> sparse.csr_matrix:
    n = A.shape[0]
    if policy == "none":
        return A
    if policy == "all":
        I = sparse.identity(n, dtype=np.float32, format="csr")
        out = (A + I).tocsr()
        out.data = np.ones_like(out.data, dtype=np.float32)
        out.eliminate_zeros()
        return out
    if policy == "isolated_only":
        U = A.maximum(A.T).tocsr()
        deg = U.getnnz(axis=1)
        isolated = np.where(deg == 0)[0].astype(np.int32)
        if isolated.size == 0:
            return A
        data = np.ones(isolated.size, dtype=np.float32)
        I_iso = sparse.coo_matrix((data, (isolated, isolated)), shape=A.shape).tocsr()
        out = (A + I_iso).tocsr()
        out.data = np.ones_like(out.data, dtype=np.float32)
        out.eliminate_zeros()
        return out
    raise ValueError(f"Unsupported self-loop policy: {policy}")


def to_edge_index(A: sparse.csr_matrix) -> np.ndarray:
    coo = A.tocoo()
    return np.vstack([coo.row.astype(np.int32), coo.col.astype(np.int32)])
