from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureEncoder:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categorical_levels: dict[str, list[str]]
    pca_components: np.ndarray | None = None
    pca_mean: np.ndarray | None = None


def _fit_pca_randomized(x: np.ndarray, n_components: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic randomized range finder + SVD on low-dimensional projection.
    rng = np.random.default_rng(seed)
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    f = xc.shape[1]
    oversample = min(8, max(2, f - n_components))
    proj_dim = min(f, n_components + oversample)
    omega = rng.normal(size=(f, proj_dim)).astype(np.float32)
    y = xc @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ xc
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    components = vt[:n_components].astype(np.float32)
    return x_mean.reshape(-1).astype(np.float32), components


def fit_encoder(df_train: pd.DataFrame, feature_cols: list[str], pca_dim: int | None = None) -> FeatureEncoder:
    x = df_train[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    for col in numeric_cols:
        values = pd.to_numeric(x[col], errors="coerce")
        m = float(values.mean()) if not np.isnan(values.mean()) else 0.0
        s = float(values.std(ddof=0)) if not np.isnan(values.std(ddof=0)) else 1.0
        if s == 0.0:
            s = 1.0
        numeric_mean[col] = m
        numeric_std[col] = s

    categorical_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        vals = x[col].astype("string").fillna("__MISSING__")
        levels = sorted(vals.unique().tolist())
        categorical_levels[col] = levels

    encoder = FeatureEncoder(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        categorical_levels=categorical_levels,
        pca_components=None,
        pca_mean=None,
    )

    if pca_dim is not None:
        z = transform(encoder, df_train)
        n_components = min(pca_dim, z.shape[0], z.shape[1])
        if n_components >= 2:
            pca_mean, pca_components = _fit_pca_randomized(z, n_components=n_components, seed=42)
            encoder.pca_mean = pca_mean
            encoder.pca_components = pca_components

    return encoder


def transform(encoder_object: FeatureEncoder, df: pd.DataFrame) -> np.ndarray:
    x = df[encoder_object.feature_cols].copy()

    parts: list[np.ndarray] = []

    if encoder_object.numeric_cols:
        numeric_mat = np.zeros((len(x), len(encoder_object.numeric_cols)), dtype=np.float32)
        for j, col in enumerate(encoder_object.numeric_cols):
            vals = pd.to_numeric(x[col], errors="coerce").fillna(encoder_object.numeric_mean[col]).to_numpy(dtype=np.float32)
            numeric_mat[:, j] = (vals - encoder_object.numeric_mean[col]) / encoder_object.numeric_std[col]
        parts.append(numeric_mat)

    if encoder_object.categorical_cols:
        cat_blocks: list[np.ndarray] = []
        for col in encoder_object.categorical_cols:
            vals = x[col].astype("string").fillna("__MISSING__")
            levels = encoder_object.categorical_levels[col]
            level_to_idx = {lvl: idx for idx, lvl in enumerate(levels)}
            block = np.zeros((len(x), len(levels)), dtype=np.float32)
            for i, v in enumerate(vals.tolist()):
                j = level_to_idx.get(v)
                if j is not None:
                    block[i, j] = 1.0
            cat_blocks.append(block)
        if cat_blocks:
            parts.append(np.concatenate(cat_blocks, axis=1))

    if not parts:
        z = np.zeros((len(x), 0), dtype=np.float32)
    else:
        z = np.concatenate(parts, axis=1).astype(np.float32)

    if encoder_object.pca_components is not None and encoder_object.pca_mean is not None and z.shape[1] > 0:
        zc = z - encoder_object.pca_mean.reshape(1, -1)
        z = (zc @ encoder_object.pca_components.T).astype(np.float32)

    return z.astype(np.float32)
