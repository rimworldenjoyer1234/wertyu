from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.data.featurize import fit_encoder, transform
from src.wertuy.data.loader import load_metadata, load_processed_dataset
from src.wertuy.graphs.builders import (
    DATASET_ENTITY_COLUMNS,
    build_relational_shared_entity_graph,
    build_similarity_knn_graph,
)
from src.wertuy.graphs.io import config_matches, save_graph_bundle, save_summary_csv
from src.wertuy.graphs.metrics import compute_graph_metrics
from src.wertuy.graphs.ops import apply_directedness, apply_self_loops, dedupe_edges
from src.wertuy.graphs.sampling import reindex_split_indices, stratified_sample_indices

LOGGER = logging.getLogger("build_graphs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph bundles from processed datasets.")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--datasets", nargs="+", default=["kdd", "unsw-nb15", "ton-iot"])
    parser.add_argument("--graph_scope", choices=["transductive", "train_only"], default="transductive")
    parser.add_argument("--knn_ks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--rel_ms", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--directedness", nargs="+", choices=["directed", "sym", "mutual"], default=["directed"])
    parser.add_argument("--self_loops", nargs="+", choices=["none", "all", "isolated_only"], default=["none"])
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dir(path: Path) -> Path:
    cands = [path]
    if not path.is_absolute():
        cands.append((REPO_ROOT / path).resolve())
        cands.append((REPO_ROOT.parent / path).resolve())
    for c in cands:
        if c.exists() and c.is_dir():
            return c
    tried = "\n  - ".join(str(x) for x in cands)
    raise FileNotFoundError(f"Could not resolve directory. Tried:\n  - {tried}")


def build_scope_dataframe(dataset_name: str, processed_dir: Path, scope: str) -> pd.DataFrame:
    train = load_processed_dataset(dataset_name, "train", processed_dir)
    val = load_processed_dataset(dataset_name, "val", processed_dir)
    test = load_processed_dataset(dataset_name, "test", processed_dir)

    train = train.copy(); train["_split"] = "train"
    val = val.copy(); val["_split"] = "val"
    test = test.copy(); test["_split"] = "test"

    if scope == "train_only":
        return pd.concat([train, val], ignore_index=True)
    return pd.concat([train, val, test], ignore_index=True)


def maybe_sample_scope(df: pd.DataFrame, max_nodes: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_nodes is None or len(df) <= max_nodes:
        return df.reset_index(drop=True), np.arange(len(df), dtype=np.int64)
    idx = stratified_sample_indices(df["y_bin"], df["_split"], max_nodes=max_nodes, seed=seed)
    sampled = df.iloc[idx].reset_index(drop=True)
    return sampled, idx.astype(np.int64)


def encode_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, dict[str, int] | None]:
    y_bin = pd.to_numeric(df["y_bin"], errors="coerce").fillna(0).astype("int8").to_numpy()
    if "y_multi" not in df.columns:
        return y_bin, None, None
    classes = sorted(df["y_multi"].astype("string").fillna("<NA>").unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_multi = df["y_multi"].astype("string").fillna("<NA>").map(mapping).astype("int32").to_numpy()
    return y_bin, y_multi, mapping


def _base_config(args: argparse.Namespace, dataset: str, constructor: str, budget: int, directedness: str, self_loops: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "constructor": constructor,
        "budget": budget,
        "directedness": directedness,
        "self_loops": self_loops,
        "graph_scope": args.graph_scope,
        "metric": args.metric,
        "pca_dim": args.pca_dim,
        "seed": args.seed,
        "max_nodes": args.max_nodes,
    }


def _graph_id_sim(k: int, metric: str, directedness: str, self_loops: str, scope: str) -> str:
    return f"sim_knn_k{k}_{metric}_{directedness}_self_{self_loops}_{scope}"


def _graph_id_rel(m: int, cols: list[str], directedness: str, self_loops: str, scope: str) -> str:
    return f"rel_shared_entity_m{m}_cols={'+'.join(cols)}_{directedness}_self_{self_loops}_{scope}"


def append_summary_row(rows: list[dict[str, Any]], dataset: str, graph_id: str, stats: dict[str, Any], config: dict[str, Any]) -> None:
    rows.append(
        {
            "dataset": dataset,
            "graph_id": graph_id,
            "constructor": config["constructor"],
            "budget": config["budget"],
            "directedness": config["directedness"],
            "self_loops": config["self_loops"],
            "scope": config["graph_scope"],
            "N": stats["N"],
            "E": stats["E"],
            "density": stats["density"],
            "mean_out_degree": stats["mean_out_degree"],
            "mean_in_degree": stats["mean_in_degree"],
            "isolated_nodes_pct": stats["isolated_nodes_pct"],
            "reciprocity": stats["reciprocity"],
            "preprocessing_time_sec": stats["preprocessing_time_sec"],
            "edge_construction_time_sec": stats["edge_construction_time_sec"],
            "total_bytes": stats["total_bytes"],
        }
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    processed_dir = resolve_dir(args.processed_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    global_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        LOGGER.info("Building graphs for dataset=%s", dataset)
        metadata = load_metadata(dataset, processed_dir)
        feature_cols = metadata.get("feature_columns", [])

        scope_df = build_scope_dataframe(dataset, processed_dir, args.graph_scope)
        scope_df, sampled_orig_idx = maybe_sample_scope(scope_df, args.max_nodes, args.seed)
        split_indices = reindex_split_indices(scope_df["_split"])

        preprocess_t0 = time.perf_counter()
        train_df_for_fit = scope_df[scope_df["_split"].isin(["train", "val"])]
        encoder = fit_encoder(train_df_for_fit, feature_cols, pca_dim=args.pca_dim)
        x = transform(encoder, scope_df)
        preprocess_time = time.perf_counter() - preprocess_t0

        y_bin, y_multi, y_multi_mapping = encode_targets(scope_df)

        dataset_rows: list[dict[str, Any]] = []
        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        # Similarity graphs
        for k in args.knn_ks:
            edge_t0 = time.perf_counter()
            base_edges = build_similarity_knn_graph(x, k=k, metric=args.metric)
            edge_build_time = time.perf_counter() - edge_t0

            for directedness in args.directedness:
                for self_loops in args.self_loops:
                    graph_id = _graph_id_sim(k, args.metric, directedness, self_loops, args.graph_scope)
                    graph_dir = dataset_out / graph_id
                    config = _base_config(args, dataset, "sim_knn", k, directedness, self_loops)
                    config["entity_columns"] = []
                    config["sampled_original_indices_file"] = "node_id_map.npy"

                    cfg_path = graph_dir / "config.json"
                    if graph_dir.exists() and not args.overwrite and config_matches(cfg_path, config):
                        LOGGER.info("Skipping existing graph %s", graph_id)
                        continue

                    edge_index = apply_directedness(base_edges, directedness)
                    edge_index = apply_self_loops(edge_index, num_nodes=len(scope_df), policy=self_loops)
                    edge_index = dedupe_edges(edge_index)

                    stats = compute_graph_metrics(
                        num_nodes=len(scope_df),
                        edge_index=edge_index,
                        directedness=directedness,
                        preprocessing_time_sec=preprocess_time,
                        edge_construction_time_sec=edge_build_time,
                        node_features_bytes=int(x.nbytes),
                    )
                    stats["warnings"] = []

                    save_graph_bundle(
                        out_dir=graph_dir,
                        edge_index=edge_index,
                        node_features=x,
                        y_bin=y_bin,
                        y_multi=y_multi,
                        y_multi_mapping=y_multi_mapping,
                        split_indices=split_indices,
                        graph_stats=stats,
                        config=config,
                        node_id_map=sampled_orig_idx,
                    )
                    append_summary_row(dataset_rows, dataset, graph_id, stats, config)

        # Relational graphs
        entity_cols = DATASET_ENTITY_COLUMNS.get(dataset, [])
        for m in args.rel_ms:
            edge_t0 = time.perf_counter()
            base_edges, rel_warnings = build_relational_shared_entity_graph(
                scope_df,
                entity_columns=entity_cols,
                per_node_cap=m,
                seed=args.seed,
            )
            edge_build_time = time.perf_counter() - edge_t0

            used_cols = [c for c in entity_cols if c in scope_df.columns]
            for directedness in args.directedness:
                for self_loops in args.self_loops:
                    graph_id = _graph_id_rel(m, used_cols, directedness, self_loops, args.graph_scope)
                    graph_dir = dataset_out / graph_id
                    config = _base_config(args, dataset, "rel_shared_entity", m, directedness, self_loops)
                    config["entity_columns"] = entity_cols
                    config["used_entity_columns"] = used_cols
                    config["sampled_original_indices_file"] = "node_id_map.npy"

                    cfg_path = graph_dir / "config.json"
                    if graph_dir.exists() and not args.overwrite and config_matches(cfg_path, config):
                        LOGGER.info("Skipping existing graph %s", graph_id)
                        continue

                    edge_index = apply_directedness(base_edges, directedness)
                    edge_index = apply_self_loops(edge_index, num_nodes=len(scope_df), policy=self_loops)
                    edge_index = dedupe_edges(edge_index)

                    stats = compute_graph_metrics(
                        num_nodes=len(scope_df),
                        edge_index=edge_index,
                        directedness=directedness,
                        preprocessing_time_sec=preprocess_time,
                        edge_construction_time_sec=edge_build_time,
                        node_features_bytes=int(x.nbytes),
                    )
                    stats["warnings"] = rel_warnings

                    save_graph_bundle(
                        out_dir=graph_dir,
                        edge_index=edge_index,
                        node_features=x,
                        y_bin=y_bin,
                        y_multi=y_multi,
                        y_multi_mapping=y_multi_mapping,
                        split_indices=split_indices,
                        graph_stats=stats,
                        config=config,
                        node_id_map=sampled_orig_idx,
                    )
                    append_summary_row(dataset_rows, dataset, graph_id, stats, config)

        save_summary_csv(dataset_rows, dataset_out / "summary.csv")
        global_rows.extend(dataset_rows)

    save_summary_csv(global_rows, out_dir / "ALL_SUMMARY.csv")
    (out_dir / "ALL_SUMMARY.json").write_text(json.dumps(global_rows, indent=2), encoding="utf-8")
    LOGGER.info("Done. Summaries at %s", out_dir)


if __name__ == "__main__":
    main()
