from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.training.train_supervised import TrainConfig, train_supervised
from src.wertuy.utils.seed import set_global_seed

LOGGER = logging.getLogger("run_rq1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--graphs_dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--results_dir", type=Path, default=Path("results/rq1"))
    p.add_argument("--model", choices=["gcn", "gat"], required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--graph_pattern", default="rq1_", help="Graph directory prefix to include")
    p.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if no graphs/results are found")
    p.add_argument("--no-strict", dest="strict", action="store_false", help="Warn and continue when files are missing")
    return p.parse_args()


def _load_edge_index(gdir: Path) -> np.ndarray:
    pt_path = gdir / "edge_index.pt"
    npy_path = gdir / "edge_index.npy"
    if pt_path.exists():
        try:
            edge_obj = torch.load(pt_path, weights_only=True)
        except TypeError:
            edge_obj = torch.load(pt_path)
        if isinstance(edge_obj, torch.Tensor):
            return edge_obj.detach().cpu().numpy()
        return np.asarray(edge_obj)
    if npy_path.exists():
        return np.load(npy_path)
    raise FileNotFoundError(f"Missing edge index in {gdir} (expected edge_index.pt or edge_index.npy)")


def _load_graph_metadata(gdir: Path) -> dict:
    meta_path = gdir / "graph_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_path = gdir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return {
            "method": cfg.get("constructor", "unknown"),
            "budget_target_avg_degree": cfg.get("k", None),
            "graph_id": gdir.name,
            "source": "config_fallback",
        }
    return {"method": "unknown", "budget_target_avg_degree": None, "graph_id": gdir.name, "source": "minimal"}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    graph_root = args.graphs_dir / args.dataset
    if not graph_root.exists():
        msg = f"Graph directory does not exist: {graph_root}. Build graphs first."
        if args.strict:
            raise FileNotFoundError(msg)
        LOGGER.warning(msg)
        return

    graph_dirs = sorted([d for d in graph_root.iterdir() if d.is_dir() and d.name.startswith(args.graph_pattern)])
    seeds = [int(s) for s in args.seeds.split(",") if s]

    if not graph_dirs:
        msg = (
            f"No graph folders found in {graph_root} with prefix '{args.graph_pattern}'. "
            "If you built fast-grid graphs, use --graph_pattern base_sim_knn_."
        )
        if args.strict:
            raise RuntimeError(msg)
        LOGGER.warning(msg)
        return

    LOGGER.info("Found %d graph folders for dataset=%s", len(graph_dirs), args.dataset)
    LOGGER.info("Running model=%s across seeds=%s", args.model, seeds)

    total_runs = 0
    skipped_graphs = 0
    for gdir in graph_dirs:
        LOGGER.info("Processing graph: %s", gdir.name)
        try:
            edge_index = _load_edge_index(gdir)
            x = np.load(gdir / "node_features.npy").astype(np.float32)
            y = np.load(gdir / "y_bin.npy").astype(np.int8)
            masks = np.load(gdir / "split_masks.npz")
            train_idx, val_idx, test_idx = masks["train_idx"], masks["val_idx"], masks["test_idx"]
            graph_meta = _load_graph_metadata(gdir)
        except Exception as exc:
            skipped_graphs += 1
            msg = f"Skipping graph {gdir.name}: {exc}"
            if args.strict:
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
            continue

        for seed in seeds:
            set_global_seed(seed)
            cfg = TrainConfig(model=args.model, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim, device=args.device)
            out = train_supervised(x, edge_index, y, train_idx, val_idx, test_idx, cfg)
            payload = {
                "dataset": args.dataset,
                "model": args.model,
                "seed": seed,
                "graph_id": gdir.name,
                "graph_metadata": graph_meta,
                "train_config": vars(cfg),
                "val": out["val"],
                "test": out["test"],
                "training_time_per_epoch_seconds": out["training_time_per_epoch_seconds"],
                "total_training_time_seconds": out["total_training_time_seconds"],
                "epochs_ran": out["epochs_ran"],
            }
            out_dir = args.results_dir / args.dataset / args.model / gdir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{seed}.json"
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            total_runs += 1
            LOGGER.info("Saved %s", out_file)

    if total_runs == 0 and args.strict:
        raise RuntimeError("No runs were executed. Check graph folders, required files, and seed list.")
    LOGGER.info("Completed %d runs (skipped_graphs=%d)", total_runs, skipped_graphs)


if __name__ == "__main__":
    main()
