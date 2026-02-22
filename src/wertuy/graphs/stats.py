from __future__ import annotations

import numpy as np


def graph_regime_stats(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> dict:
    e = int(src.size)
    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    dbar = float(out_deg.mean()) if num_nodes > 0 else 0.0
    density = float(e) / float(max(num_nodes * (num_nodes - 1), 1))
    return {
        "N": int(num_nodes),
        "E": e,
        "achieved_avg_degree": dbar,
        "mean_out_degree": float(out_deg.mean()) if num_nodes > 0 else 0.0,
        "mean_in_degree": float(in_deg.mean()) if num_nodes > 0 else 0.0,
        "density": density,
        "edge_index_bytes": int(2 * e * 8),
    }
