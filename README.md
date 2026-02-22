wertuyu

## RQ1 end-to-end

1. Sanity-check processed splits:

```bash
python scripts/sanity_check_processed.py --processed_dir data/processed
```

2. Build RQ1 graphs (example UNSW-NB15):

```bash
python scripts/build_graphs_rq1.py --dataset unsw-nb15 --processed_dir data/processed --graphs_dir data/graphs --method knn_directed --budgets 4,8,16,32 --pca_dim 32 --max_nodes 5000 --seed 42
python scripts/build_graphs_rq1.py --dataset unsw-nb15 --processed_dir data/processed --graphs_dir data/graphs --method knn_sym --budgets 4,8,16,32 --pca_dim 32 --max_nodes 5000 --seed 42
python scripts/build_graphs_rq1.py --dataset unsw-nb15 --processed_dir data/processed --graphs_dir data/graphs --method knn_mutual --budgets 4,8,16,32 --pca_dim 32 --max_nodes 5000 --seed 42
python scripts/build_graphs_rq1.py --dataset unsw-nb15 --processed_dir data/processed --graphs_dir data/graphs --method topm --budgets 4,8,16,32 --pca_dim 32 --max_nodes 5000 --seed 42
```

3. Run RQ1 models:

```bash
python -m scripts.run_rq1 --dataset unsw-nb15 --graphs_dir data/graphs --processed_dir data/processed --results_dir results/rq1 --model gcn --seeds 0,1,2 --epochs 100 --lr 0.001 --hidden_dim 64 --device cpu
python -m scripts.run_rq1 --dataset unsw-nb15 --graphs_dir data/graphs --processed_dir data/processed --results_dir results/rq1 --model gat --seeds 0,1,2 --epochs 100 --lr 0.001 --hidden_dim 64 --device cpu
```

4. Aggregate and plot:

```bash
python scripts/analyze_rq1.py
```

Outputs:
- Cached graphs: `data/graphs/<dataset>/rq1_*`
- Per-seed run JSON: `results/rq1/<dataset>/<model>/<graph_id>/<seed>.json`
- Summary: `results/rq1/summary.csv`
- Figures: `results/rq1/figures/*.png`
