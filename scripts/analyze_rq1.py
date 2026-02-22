from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("analyze_rq1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate and plot RQ1 experiment results")
    p.add_argument("--results_dir", type=Path, default=Path("results/rq1"))
    p.add_argument(
        "--graph_pattern",
        default="",
        help="Optional graph_id prefix filter (e.g. 'rq1_' or 'base_sim_knn_'). Empty means all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = args.results_dir
    rows = []

    for p in root.glob("*/*/*/*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable JSON %s: %s", p, exc)
            continue

        graph_id = d.get("graph_id", "")
        if args.graph_pattern and not graph_id.startswith(args.graph_pattern):
            continue

        rows.append(
            {
                "dataset": d.get("dataset"),
                "model": d.get("model"),
                "graph_id": graph_id,
                "seed": d.get("seed"),
                "method": d.get("graph_metadata", {}).get("method", "unknown"),
                "budget": d.get("graph_metadata", {}).get("budget_target_avg_degree"),
                "mcc_val": d.get("val", {}).get("mcc"),
                "mcc_test": d.get("test", {}).get("mcc"),
                "edge_creation_time_seconds": d.get("graph_metadata", {}).get("edge_creation_time_seconds"),
                "estimated_graph_memory_bytes": d.get("graph_metadata", {}).get("estimated_graph_memory_bytes"),
                "total_training_time_seconds": d.get("total_training_time_seconds"),
            }
        )

    if not rows:
        LOGGER.warning(
            "No result JSON files found under %s (pattern */*/*/*.json, graph_pattern=%r)",
            root,
            args.graph_pattern,
        )
        LOGGER.warning("Run scripts/run_rq1.py first and verify graph IDs/results paths.")
        return

    df = pd.DataFrame(rows)
    out_csv = root / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote summary CSV: %s (rows=%d)", out_csv, len(df))

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for (ds, model), g in df.groupby(["dataset", "model"]):
        agg = g.groupby(["budget", "method"], as_index=False)["mcc_test"].mean()
        plt.figure()
        for m, gm in agg.groupby("method"):
            plt.plot(gm["budget"], gm["mcc_test"], marker="o", label=m)
        plt.xlabel("Target avg degree")
        plt.ylabel("Test MCC")
        plt.title(f"MCC vs Budget | {ds} | {model}")
        plt.legend()
        plt.tight_layout()
        out1 = figdir / f"mcc_vs_budget_{ds}_{model}.png"
        plt.savefig(out1)
        plt.close()

        cost = g.groupby(["budget", "method"], as_index=False)[["edge_creation_time_seconds", "total_training_time_seconds"]].mean()
        plt.figure()
        for m, gm in cost.groupby("method"):
            plt.plot(gm["budget"], gm["edge_creation_time_seconds"], marker="o", label=f"{m}:construct")
            plt.plot(gm["budget"], gm["total_training_time_seconds"], marker="x", linestyle="--", label=f"{m}:train")
        plt.xlabel("Target avg degree")
        plt.ylabel("Seconds")
        plt.title(f"Cost vs Budget | {ds}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out2 = figdir / f"cost_vs_budget_{ds}.png"
        plt.savefig(out2)
        plt.close()

        pto = g.groupby("method", as_index=False).agg({"mcc_test": "mean", "edge_creation_time_seconds": "mean", "total_training_time_seconds": "mean"})
        pto["total_cost"] = pto["edge_creation_time_seconds"] + pto["total_training_time_seconds"]
        plt.figure()
        plt.scatter(pto["total_cost"], pto["mcc_test"])
        for _, r in pto.iterrows():
            plt.annotate(r["method"], (r["total_cost"], r["mcc_test"]))
        plt.xlabel("Total cost (sec)")
        plt.ylabel("Test MCC")
        plt.title(f"Pareto | {ds} | {model}")
        plt.tight_layout()
        out3 = figdir / f"pareto_{ds}_{model}.png"
        plt.savefig(out3)
        plt.close()

        LOGGER.info("Wrote figures: %s, %s, %s", out1, out2, out3)


if __name__ == "__main__":
    main()
