from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.wertuy.metrics.classification import auc_scores, matthews_corrcoef_binary
from src.wertuy.models.gat import SimpleGAT
from src.wertuy.models.gcn import SimpleGCN


@dataclass
class TrainConfig:
    model: str
    epochs: int = 100
    lr: float = 1e-3
    hidden_dim: int = 64
    weight_decay: float = 1e-4
    patience: int = 20
    device: str = "cpu"
    class_weight_neg: float = 1.0
    class_weight_pos: float = 1.0


def best_threshold_by_mcc(y_true: np.ndarray, prob: np.ndarray, n_steps: int = 200) -> tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_thr = 0.5
    best_mcc = -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        mcc = matthews_corrcoef_binary(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(t)
    return best_thr, best_mcc


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _metrics_with_threshold(logits: torch.Tensor, y: torch.Tensor, threshold: float) -> dict:
    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = (prob >= threshold).astype(int)
    yt = y.detach().cpu().numpy().astype(int)
    mcc = matthews_corrcoef_binary(yt, pred)
    auc = auc_scores(yt, prob)
    conf = _confusion(yt, pred)
    return {
        "mcc": mcc,
        "threshold": float(threshold),
        "pred_pos_rate": float(pred.mean()) if pred.size else 0.0,
        "y_pos_rate": float(yt.mean()) if yt.size else 0.0,
        **conf,
        **auc,
    }


def train_supervised(
    x: np.ndarray,
    edge_index: np.ndarray,
    y_bin: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
) -> dict:
    device = torch.device(cfg.device)
    X = torch.tensor(x, dtype=torch.float32, device=device)
    E = torch.tensor(edge_index, dtype=torch.long, device=device)
    y = torch.tensor(y_bin.astype(np.int64), dtype=torch.long, device=device)

    if cfg.model == "gcn":
        model = SimpleGCN(X.size(1), cfg.hidden_dim, 2).to(device)
    elif cfg.model == "gat":
        model = SimpleGAT(X.size(1), cfg.hidden_dim, 2).to(device)
    else:
        raise ValueError(cfg.model)

    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    va = torch.tensor(val_idx, dtype=torch.long, device=device)
    te = torch.tensor(test_idx, dtype=torch.long, device=device)

    y_train_np = y_bin[train_idx].astype(int) if len(train_idx) else np.array([0], dtype=int)
    pos = int((y_train_np == 1).sum())
    neg = int((y_train_np == 0).sum())
    pos_safe = max(pos, 1)
    pos_weight = float(neg / pos_safe) if neg > 0 else 1.0
    pos_weight = float(np.clip(pos_weight, 1.0, 100.0))
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    cfg.class_weight_neg = 1.0
    cfg.class_weight_pos = pos_weight

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best = -1.0
    best_state = None
    best_threshold = 0.5
    bad = 0
    ep_times = []
    t0 = time.perf_counter()

    for _ in range(cfg.epochs):
        ep0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        logits = model(X, E)
        loss = crit(logits[tr], y[tr])
        loss.backward()
        opt.step()
        ep_times.append(time.perf_counter() - ep0)

        model.eval()
        with torch.no_grad():
            val_logits = model(X, E)[va]
            if va.numel() > 0:
                val_prob = torch.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
                val_true = y[va].detach().cpu().numpy().astype(int)
                thr, val_mcc_best = best_threshold_by_mcc(val_true, val_prob)
            else:
                thr, val_mcc_best = 0.5, 0.0

        if val_mcc_best > best:
            best = val_mcc_best
            best_threshold = thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        full = model(X, E)
        val_m = _metrics_with_threshold(full[va], y[va], best_threshold) if va.numel() > 0 else {
            "mcc": 0.0,
            "threshold": best_threshold,
            "pred_pos_rate": 0.0,
            "y_pos_rate": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "auc_pr": None,
            "auc_roc": None,
        }
        test_m = _metrics_with_threshold(full[te], y[te], best_threshold) if te.numel() > 0 else {
            "mcc": 0.0,
            "threshold": best_threshold,
            "pred_pos_rate": 0.0,
            "y_pos_rate": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "auc_pr": None,
            "auc_roc": None,
        }

    total_t = time.perf_counter() - t0
    return {
        "val": val_m,
        "test": test_m,
        "best_threshold": float(best_threshold),
        "best_val_mcc": float(best),
        "class_weight_neg": 1.0,
        "class_weight_pos": pos_weight,
        "training_time_per_epoch_seconds": float(np.mean(ep_times)) if ep_times else 0.0,
        "total_training_time_seconds": total_t,
        "epochs_ran": len(ep_times),
    }
