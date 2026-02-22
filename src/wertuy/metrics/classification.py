from __future__ import annotations

import numpy as np


def matthews_corrcoef_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def auc_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auc_pr": float(average_precision_score(y_true, y_score)),
            "auc_roc": float(roc_auc_score(y_true, y_score)),
        }
    except Exception:
        return {"auc_pr": None, "auc_roc": None}
