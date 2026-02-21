from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Adjacency:
    num_nodes: int
    src: np.ndarray
    dst: np.ndarray
    weights: np.ndarray | None = None

    @property
    def nnz(self) -> int:
        return int(self.src.shape[0])


def _unique_edges(src: np.ndarray, dst: np.ndarray, num_nodes: int, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if src.size == 0:
        return src.astype(np.int32), dst.astype(np.int32), weights
    keys = src.astype(np.int64) * int(num_nodes) + dst.astype(np.int64)
    uniq_keys, idx = np.unique(keys, return_index=True)
    src_u = (uniq_keys // int(num_nodes)).astype(np.int32)
    dst_u = (uniq_keys % int(num_nodes)).astype(np.int32)
    if weights is None:
        return src_u, dst_u, None
    return src_u, dst_u, weights[idx].astype(np.float32)


def build_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    weights: np.ndarray | None = None,
) -> Adjacency:
    src_u, dst_u, w_u = _unique_edges(src.astype(np.int32), dst.astype(np.int32), num_nodes, weights)
    return Adjacency(num_nodes=num_nodes, src=src_u, dst=dst_u, weights=w_u)


def apply_directedness(A: Adjacency, mode: str) -> Adjacency:
    if mode == "directed":
        return A

    n = A.num_nodes
    src = A.src
    dst = A.dst
    if mode == "sym":
        s2 = np.concatenate([src, dst]).astype(np.int32)
        d2 = np.concatenate([dst, src]).astype(np.int32)
        su, du, _ = _unique_edges(s2, d2, n)
        return Adjacency(num_nodes=n, src=su, dst=du, weights=None)

    if mode == "mutual":
        keys = src.astype(np.int64) * n + dst.astype(np.int64)
        rev_keys = dst.astype(np.int64) * n + src.astype(np.int64)
        mutual_keys = np.intersect1d(keys, rev_keys, assume_unique=False)
        su = (mutual_keys // n).astype(np.int32)
        du = (mutual_keys % n).astype(np.int32)
        return Adjacency(num_nodes=n, src=su, dst=du, weights=None)

    raise ValueError(f"Unsupported directedness mode: {mode}")


def apply_self_loops(A: Adjacency, policy: str) -> Adjacency:
    n = A.num_nodes
    src = A.src
    dst = A.dst
    if policy == "none":
        return A
    if policy == "all":
        loops = np.arange(n, dtype=np.int32)
        s2 = np.concatenate([src, loops])
        d2 = np.concatenate([dst, loops])
        su, du, _ = _unique_edges(s2, d2, n)
        return Adjacency(num_nodes=n, src=su, dst=du, weights=None)
    if policy == "isolated_only":
        deg = np.bincount(np.concatenate([src, dst]), minlength=n)
        iso = np.where(deg == 0)[0].astype(np.int32)
        if iso.size == 0:
            return A
        s2 = np.concatenate([src, iso])
        d2 = np.concatenate([dst, iso])
        su, du, _ = _unique_edges(s2, d2, n)
        return Adjacency(num_nodes=n, src=su, dst=du, weights=None)
    raise ValueError(f"Unsupported self-loop policy: {policy}")


def to_edge_index(A: Adjacency) -> np.ndarray:
    return np.vstack([A.src.astype(np.int32), A.dst.astype(np.int32)])
