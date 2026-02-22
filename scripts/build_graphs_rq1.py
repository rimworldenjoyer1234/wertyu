from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.data.loader import load_metadata, load_processed_dataset
from src.wertuy.graphs.budget import target_edge_count
from src.wertuy.graphs.constructors import knn_directed, knn_mutual, knn_sym, topm_global_from_knn_pool
from src.wertuy.graphs.stats import graph_regime_stats
from src.wertuy.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--graphs_dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--budgets", default="4,8,16,32")
    p.add_argument("--method", choices=["knn_directed", "knn_sym", "knn_mutual", "topm"], required=True)
    p.add_argument("--pca_dim", type=int, default=32)
    p.add_argument("--max_nodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_matrix(df: pd.DataFrame, feature_cols: list[str], pca_dim: int | None) -> np.ndarray:
    x = pd.get_dummies(df[feature_cols], dummy_na=True)
    X = x.to_numpy(dtype=np.float32)
    if pca_dim is not None and X.shape[1] > pca_dim and X.shape[0] > pca_dim:
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        X = (Xc @ vt[:pca_dim].T).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return (X / n).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    tr = load_processed_dataset(args.dataset, "train", args.processed_dir); tr["_split"] = "train"
    va = load_processed_dataset(args.dataset, "val", args.processed_dir); va["_split"] = "val"
    te = load_processed_dataset(args.dataset, "test", args.processed_dir); te["_split"] = "test"
    df = pd.concat([tr, va, te], ignore_index=True)

    if args.max_nodes and len(df) > args.max_nodes:
        df = df.groupby(["_split", "y_bin"], group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(len(g) * args.max_nodes / len(df)))), random_state=args.seed)
        ).reset_index(drop=True)

    md = load_metadata(args.dataset, args.processed_dir)
    feature_cols = md["feature_columns"]
    X = make_matrix(df, feature_cols, args.pca_dim)

    budgets = [int(x) for x in args.budgets.split(",") if x]
    out_root = args.graphs_dir / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    for b in budgets:
        tgt_e = target_edge_count(len(df), b, directed=True)
        k = max(1, b)
        s0 = __import__("time").perf_counter()
        s, d, w = knn_directed(X, k=k, metric="cosine")
        if args.method == "knn_sym":
            s, d, w = knn_sym(s, d, w)
        elif args.method == "knn_mutual":
            s, d, w = knn_mutual(s, d, w)
        elif args.method == "topm":
            s, d, w = topm_global_from_knn_pool(X, candidate_k=max(2 * k, 16), target_edges=tgt_e, metric="cosine")
        edge_time = __import__("time").perf_counter() - s0

        if args.method != "topm" and s.size > tgt_e:
            order = np.argsort(-w)[:tgt_e]
            s, d, w = s[order], d[order], w[order]

        gid = f"rq1_{args.method}_d{b}_pca{args.pca_dim}"
        gdir = out_root / gid
        gdir.mkdir(parents=True, exist_ok=True)
        edge_index = np.vstack([s, d]).astype(np.int64)
        torch.save(torch.tensor(edge_index, dtype=torch.long), gdir / "edge_index.pt")
        np.save(gdir / "node_features.npy", X)
        np.save(gdir / "y_bin.npy", df["y_bin"].to_numpy(dtype=np.int8))
        np.save(gdir / "edge_weight.npy", w.astype(np.float32))
        split_masks = {
            "train_idx": np.where(df["_split"].values == "train")[0],
            "val_idx": np.where(df["_split"].values == "val")[0],
            "test_idx": np.where(df["_split"].values == "test")[0],
        }
        np.savez(gdir / "split_masks.npz", **split_masks)

        stats = graph_regime_stats(len(df), s, d)
        stats.update(
            {
                "method": args.method,
                "budget_target_avg_degree": b,
                "target_E": int(tgt_e),
                "achieved_E": int(s.size),
                "edge_creation_time_seconds": float(edge_time),
                "estimated_graph_memory_bytes": int(2 * s.size * 8 + w.nbytes),
                "graph_id": gid,
            }
        )
        (gdir / "graph_metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
