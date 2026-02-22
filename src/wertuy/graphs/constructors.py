from __future__ import annotations

import numpy as np

from src.wertuy.graphs.budget import prune_to_budget


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


def knn_directed(x: np.ndarray, k: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    k_eff = min(k, max(0, n - 1))
    if k_eff == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, algorithm="auto", n_jobs=-1)
        nn.fit(x)
        dist, idx = nn.kneighbors(x, return_distance=True)
        src = np.repeat(np.arange(n, dtype=np.int32), k_eff)
        dst = idx[:, 1:].reshape(-1).astype(np.int32)
        score = (1.0 - dist[:, 1:].reshape(-1)).astype(np.float32) if metric == "cosine" else np.exp(-dist[:, 1:].reshape(-1)).astype(np.float32)
        return src, dst, score
    except Exception:
        pass

    z = _l2norm(x.astype(np.float32)) if metric == "cosine" else x.astype(np.float32)
    if metric == "cosine":
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)
        nbr = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
        sc = np.take_along_axis(sim, nbr, axis=1)
        order = np.argsort(-sc, axis=1)
        nbr = np.take_along_axis(nbr, order, axis=1)
        sc = np.take_along_axis(sc, order, axis=1)
    else:
        x2 = np.sum(z * z, axis=1)
        d2 = x2[:, None] + x2[None, :] - 2 * (z @ z.T)
        d2 = np.maximum(d2, 0)
        np.fill_diagonal(d2, np.inf)
        nbr = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        dv = np.take_along_axis(d2, nbr, axis=1)
        order = np.argsort(dv, axis=1)
        nbr = np.take_along_axis(nbr, order, axis=1)
        sc = np.exp(-np.sqrt(np.take_along_axis(d2, nbr, axis=1)))
    src = np.repeat(np.arange(n, dtype=np.int32), k_eff)
    dst = nbr.reshape(-1).astype(np.int32)
    return src, dst, sc.reshape(-1).astype(np.float32)


def knn_sym(src: np.ndarray, dst: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(src.max(initial=0), dst.max(initial=0)) + 1) if src.size else 0
    keys = src.astype(np.int64) * max(n,1) + dst.astype(np.int64)
    rev_keys = dst.astype(np.int64) * max(n,1) + src.astype(np.int64)
    both_keys = np.concatenate([keys, rev_keys])
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    both_score = np.concatenate([score, score])
    uniq, idx = np.unique(both_keys, return_index=True)
    return both_src[idx].astype(np.int32), both_dst[idx].astype(np.int32), both_score[idx].astype(np.float32)


def knn_mutual(src: np.ndarray, dst: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(src.max(initial=0), dst.max(initial=0)) + 1) if src.size else 0
    keys = src.astype(np.int64) * max(n,1) + dst.astype(np.int64)
    rev = dst.astype(np.int64) * max(n,1) + src.astype(np.int64)
    mutual = np.intersect1d(keys, rev, assume_unique=False)
    if mutual.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    msrc = (mutual // max(n,1)).astype(np.int32)
    mdst = (mutual % max(n,1)).astype(np.int32)
    key_to_score = {int(k): float(s) for k, s in zip(keys.tolist(), score.tolist())}
    ms = np.array([key_to_score.get(int(k), 1.0) for k in mutual.tolist()], dtype=np.float32)
    return msrc, mdst, ms


def topm_global_from_knn_pool(x: np.ndarray, candidate_k: int, target_edges: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src, dst, score = knn_directed(x, k=candidate_k, metric=metric)
    return prune_to_budget(src, dst, score, target_edges)
