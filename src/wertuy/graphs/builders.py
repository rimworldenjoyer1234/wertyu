from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


DATASET_ENTITY_COLUMNS: dict[str, list[str]] = {
    "kdd": ["protocol_type", "service", "flag"],
    "unsw-nb15": ["proto", "service", "state"],
    "ton-iot": ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "service"],
}


def _normalize_l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def _numpy_knn(X: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(np.float32)
    n = X.shape[0]
    k_eff = min(k, max(0, n - 1))
    if k_eff == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    if metric == "cosine":
        Z = _normalize_l2(X)
        sims = Z @ Z.T
        np.fill_diagonal(sims, -np.inf)
        nbr = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        sc = np.take_along_axis(sims, nbr, axis=1)
        ord_idx = np.argsort(-sc, axis=1)
        nbr = np.take_along_axis(nbr, ord_idx, axis=1)
        w = np.take_along_axis(sc, ord_idx, axis=1)
    else:
        x2 = np.sum(X * X, axis=1)
        d2 = x2.reshape(-1, 1) + x2.reshape(1, -1) - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        np.fill_diagonal(d2, np.inf)
        nbr = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        dv = np.take_along_axis(d2, nbr, axis=1)
        ord_idx = np.argsort(dv, axis=1)
        nbr = np.take_along_axis(nbr, ord_idx, axis=1)
        dist = np.sqrt(np.take_along_axis(dv, ord_idx, axis=1))
        w = np.exp(-dist)

    src = np.repeat(np.arange(n, dtype=np.int32), k_eff)
    dst = nbr.reshape(-1).astype(np.int32)
    weights = w.reshape(-1).astype(np.float32)
    return src, dst, weights


def build_similarity_knn_graph(
    X,
    k: int,
    metric: str = "cosine",
    use_sklearn: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n = X.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), None

    # Optional sklearn backend (off by default to avoid noisy ABI crashes on broken SciPy/NumPy installs).
    if use_sklearn:
        try:
            from sklearn.neighbors import NearestNeighbors

            n_neighbors = min(k + 1, n)
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto", n_jobs=-1)
            nn.fit(X)
            distances, indices = nn.kneighbors(X, return_distance=True)
            src = np.repeat(np.arange(n, dtype=np.int32), n_neighbors - 1)
            dst = indices[:, 1:].reshape(-1).astype(np.int32)
            if metric == "cosine":
                weights = (1.0 - distances[:, 1:].reshape(-1)).astype(np.float32)
            elif metric == "euclidean":
                weights = np.exp(-distances[:, 1:].reshape(-1)).astype(np.float32)
            else:
                weights = None
            return src, dst, weights
        except Exception:
            pass

    src, dst, weights = _numpy_knn(np.asarray(X), k=k, metric=metric)
    return src, dst, weights


def build_relational_shared_entity_graph(
    df,
    entity_columns: Sequence[str],
    per_node_cap: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    missing = [c for c in entity_columns if c not in df.columns]
    used_columns = [c for c in entity_columns if c in df.columns]

    group_map: dict[str, dict[str, list[int]]] = {col: defaultdict(list) for col in used_columns}
    for idx, row in df.reset_index(drop=True).iterrows():
        for col in used_columns:
            val = row[col]
            if val is None:
                continue
            key = str(val)
            if key in {"nan", "<NA>", "None"}:
                continue
            group_map[col][key].append(idx)

    src_list: list[int] = []
    dst_list: list[int] = []
    for col in used_columns:
        for _, nodes in group_map[col].items():
            if len(nodes) <= 1:
                continue
            L = len(nodes)
            cap = min(per_node_cap, L - 1)
            for i, u in enumerate(nodes):
                for t in range(1, cap + 1):
                    v = nodes[(i + t) % L]
                    if u != v:
                        src_list.append(u)
                        dst_list.append(v)

    warnings = [f"Entity column '{col}' missing; skipped" for col in missing]
    if not used_columns:
        warnings.append("No entity columns available for relational graph")

    return np.asarray(src_list, dtype=np.int32), np.asarray(dst_list, dtype=np.int32), warnings
