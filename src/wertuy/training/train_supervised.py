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


def _metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    yt = y.detach().cpu().numpy().astype(int)
    mcc = matthews_corrcoef_binary(yt, pred)
    auc = auc_scores(yt, prob)
    return {"mcc": mcc, **auc}


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

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    best = -1.0
    best_state = None
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
            mcc = _metrics(val_logits, y[va])["mcc"]
        if mcc > best:
            best = mcc
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
        val_m = _metrics(full[va], y[va]) if va.numel() > 0 else {"mcc": 0.0, "auc_pr": None, "auc_roc": None}
        test_m = _metrics(full[te], y[te]) if te.numel() > 0 else {"mcc": 0.0, "auc_pr": None, "auc_roc": None}

    total_t = time.perf_counter() - t0
    return {
        "val": val_m,
        "test": test_m,
        "training_time_per_epoch_seconds": float(np.mean(ep_times)) if ep_times else 0.0,
        "total_training_time_seconds": total_t,
        "epochs_ran": len(ep_times),
    }
