from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.data.featurize import build_feature_matrix
from src.wertuy.graphs.builders import build_similarity_knn_graph
from src.wertuy.graphs.io import config_matches, save_graph_bundle, save_summary_csv
from src.wertuy.graphs.metrics import compute_graph_metrics
from src.wertuy.graphs.ops import apply_directedness, apply_self_loops, build_adjacency, to_edge_index

LOGGER = logging.getLogger("build_graphs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast/scalable graph builder")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--datasets", nargs="+", default=["kdd", "unsw-nb15", "ton-iot"])
    parser.add_argument("--graph_scope", choices=["transductive", "train_only"], default="transductive")
    parser.add_argument("--knn_ks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--directedness", nargs="+", choices=["directed", "sym", "mutual"], default=["directed"])
    parser.add_argument("--self_loops", nargs="+", choices=["none", "all", "isolated_only"], default=["none"])
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stage", choices=["featurize", "graphs", "all"], default="all")
    parser.add_argument("--fast_grid", action="store_true")
    parser.add_argument("--store_base_only", action="store_true", default=True)
    parser.add_argument("--use_sklearn_backend", action="store_true", help="Opt-in sklearn/scipy backend for featurization/kNN")
    return parser.parse_args()


def resolve_dir(path: Path) -> Path:
    cands = [path]
    if not path.is_absolute():
        cands.extend([(REPO_ROOT / path).resolve(), (REPO_ROOT.parent / path).resolve()])
    for c in cands:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError("Could not resolve directory: " + ", ".join(str(c) for c in cands))


def apply_fast_grid(args: argparse.Namespace) -> argparse.Namespace:
    if not args.fast_grid:
        return args
    args.graph_scope = "train_only"
    args.knn_ks = [8, 16, 32]
    args.directedness = ["directed"]
    args.self_loops = ["none"]
    args.metric = "cosine"
    if args.pca_dim is None:
        args.pca_dim = 64
    return args


def sample_nodes(
    X,
    y_bin: np.ndarray,
    splits: dict[str, np.ndarray],
    max_nodes: int | None,
    seed: int,
):
    n = y_bin.shape[0]
    if max_nodes is None or n <= max_nodes:
        idx = np.arange(n, dtype=np.int32)
        return X, y_bin, splits, idx

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"idx": np.arange(n), "y": y_bin})
    sample: list[int] = []
    for cls, g in df.groupby("y"):
        _ = cls
        frac = len(g) / n
        take = max(1, int(round(max_nodes * frac)))
        take = min(take, len(g))
        sample.extend(rng.choice(g["idx"].to_numpy(), size=take, replace=False).tolist())
    sample = sorted(set(sample))[:max_nodes]
    sample_arr = np.asarray(sample, dtype=np.int32)

    if hasattr(X, "tocsr"):
        Xs = X[sample_arr]
    else:
        Xs = X[sample_arr]
    ys = y_bin[sample_arr]

    remap = {old: new for new, old in enumerate(sample_arr.tolist())}
    new_splits: dict[str, np.ndarray] = {}
    for name, arr in splits.items():
        keep = [remap[v] for v in arr.tolist() if int(v) in remap]
        new_splits[name] = np.asarray(keep, dtype=np.int32)
    return Xs, ys, new_splits, sample_arr


def graph_id(k: int, metric: str, pca_dim: int | None, scope: str) -> str:
    pca_tag = f"pca{pca_dim}" if pca_dim is not None else "pcaNone"
    return f"base_sim_knn_k{k}_{metric}_{pca_tag}_{scope}"


def main() -> None:
    args = apply_fast_grid(parse_args())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    processed_dir = resolve_dir(args.processed_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "_feature_cache"
    all_rows: list[dict] = []

    for dataset in args.datasets:
        dataset_rows: list[dict] = []
        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        X, y_bin, splits = build_feature_matrix(
            dataset_name=dataset,
            scope=args.graph_scope,
            processed_dir=processed_dir,
            cache_dir=cache_dir,
            seed=args.seed,
            pca_dim=args.pca_dim,
            metric=args.metric,
            use_sklearn_backend=args.use_sklearn_backend,
        )
        feature_time = time.perf_counter() - t0

        X, y_bin, splits, sampled_idx = sample_nodes(X, y_bin, splits, args.max_nodes, args.seed)

        if args.stage == "featurize":
            LOGGER.info("Featurize stage complete for %s", dataset)
            continue

        for k in args.knn_ks:
            gid = graph_id(k, args.metric, args.pca_dim, args.graph_scope)
            graph_dir = dataset_out / gid
            config = {
                "dataset": dataset,
                "constructor": "sim_knn",
                "k": k,
                "metric": args.metric,
                "scope": args.graph_scope,
                "pca_dim": args.pca_dim,
                "seed": args.seed,
                "max_nodes": args.max_nodes,
                "store_base_only": bool(args.store_base_only),
                "directedness": "directed",
                "self_loops": "none",
                "sampled_node_indices": "node_index_map_sample.npy",
            }
            cfg_path = graph_dir / "config.json"
            if graph_dir.exists() and not args.overwrite and config_matches(cfg_path, config):
                LOGGER.info("Skipping existing %s", gid)
                continue

            t1 = time.perf_counter()
            src, dst, w = build_similarity_knn_graph(X, k=k, metric=args.metric, use_sklearn=args.use_sklearn_backend)
            knn_time = time.perf_counter() - t1

            t2 = time.perf_counter()
            A = build_adjacency(src, dst, num_nodes=int(y_bin.shape[0]), weights=w)
            A = apply_directedness(A, "directed")
            A = apply_self_loops(A, "none")
            ops_time = time.perf_counter() - t2

            t3 = time.perf_counter()
            stats = compute_graph_metrics(
                A=A,
                X=X,
                feature_time_sec=feature_time,
                knn_time_sec=knn_time,
                ops_time_sec=ops_time,
                metrics_time_sec=0.0,
            )
            stats["metrics_time_sec"] = time.perf_counter() - t3

            edge_index = to_edge_index(A)
            edge_weight = A.weights.astype(np.float32) if (A.weights is not None and A.nnz > 0) else None

            save_graph_bundle(
                out_dir=graph_dir,
                edge_index=edge_index,
                edge_weight=edge_weight,
                node_features=X,
                y_bin=y_bin,
                split_indices=splits,
                graph_stats=stats,
                config=config,
            )
            np.save(graph_dir / "node_index_map_sample.npy", sampled_idx.astype(np.int32))

            row = {
                "dataset": dataset,
                "graph_id": gid,
                "k": k,
                "metric": args.metric,
                "scope": args.graph_scope,
                "N": stats["N"],
                "E": stats["E"],
                "density": stats["density"],
                "mean_out_degree": stats["mean_out_degree"],
                "mean_in_degree": stats["mean_in_degree"],
                "isolated_nodes_pct": stats["isolated_nodes_pct"],
                "reciprocity": stats["reciprocity"],
                "feature_time_sec": stats["feature_time_sec"],
                "knn_time_sec": stats["knn_time_sec"],
                "ops_time_sec": stats["ops_time_sec"],
                "metrics_time_sec": stats["metrics_time_sec"],
                "total_bytes": stats["total_bytes"],
            }
            dataset_rows.append(row)
            all_rows.append(row)

        save_summary_csv(dataset_rows, dataset_out / "summary.csv")

    save_summary_csv(all_rows, out_dir / "ALL_SUMMARY.csv")
    (out_dir / "ALL_SUMMARY.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    LOGGER.info("Done. Graph outputs at %s", out_dir)


if __name__ == "__main__":
    main()
