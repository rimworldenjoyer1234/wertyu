from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wertuy.training.train_supervised import TrainConfig, train_supervised
from src.wertuy.utils.seed import set_global_seed


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph_root = args.graphs_dir / args.dataset
    graph_dirs = sorted([d for d in graph_root.iterdir() if d.is_dir() and d.name.startswith("rq1_")])
    seeds = [int(s) for s in args.seeds.split(",") if s]

    for gdir in graph_dirs:
        edge_index = torch.load(gdir / "edge_index.pt").numpy()
        x = np.load(gdir / "node_features.npy").astype(np.float32)
        y = np.load(gdir / "y_bin.npy").astype(np.int8)
        masks = np.load(gdir / "split_masks.npz")
        train_idx, val_idx, test_idx = masks["train_idx"], masks["val_idx"], masks["test_idx"]
        graph_meta = json.loads((gdir / "graph_metadata.json").read_text(encoding="utf-8"))

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
            (out_dir / f"{seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
