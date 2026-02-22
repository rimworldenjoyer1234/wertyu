from __future__ import annotations

import numpy as np


def target_edge_count(num_nodes: int, avg_degree: int, directed: bool = True) -> int:
    if directed:
        return int(num_nodes * avg_degree)
    return int(num_nodes * avg_degree / 2)


def prune_to_budget(src: np.ndarray, dst: np.ndarray, score: np.ndarray, target_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if src.size <= target_edges:
        return src, dst, score
    order = np.argsort(-score)
    keep = order[:target_edges]
    return src[keep], dst[keep], score[keep]
