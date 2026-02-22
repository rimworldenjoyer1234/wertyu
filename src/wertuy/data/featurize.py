from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.wertuy.data.loader import load_metadata, load_processed_dataset


def _resolve_processed_scope(dataset_name: str, scope: str, processed_dir: Path) -> pd.DataFrame:
    train = load_processed_dataset(dataset_name, "train", processed_dir).copy()
    val = load_processed_dataset(dataset_name, "val", processed_dir).copy()
    test = load_processed_dataset(dataset_name, "test", processed_dir).copy()
    train["_split"] = "train"
    val["_split"] = "val"
    test["_split"] = "test"
    if scope == "train_only":
        return pd.concat([train, val], ignore_index=True)
    if scope == "transductive":
        return pd.concat([train, val, test], ignore_index=True)
    raise ValueError(f"Unsupported scope: {scope}")


def _split_indices(split_labels: pd.Series) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, s in enumerate(split_labels.astype(str).tolist()):
        if s in out:
            out[s].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in out.items()}


def _normalize_l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (X / norms).astype(np.float32)


def _build_numpy_features(df_train: pd.DataFrame, df_all: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    blocks: list[np.ndarray] = []
    enc_meta: dict[str, Any] = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}

    if numeric_cols:
        num_train = df_train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        mean = num_train.mean().fillna(0.0)
        std = num_train.std(ddof=0).replace(0, 1).fillna(1.0)
        num_all = df_all[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(mean)
        num_scaled = ((num_all - mean) / std).to_numpy(dtype=np.float32)
        blocks.append(num_scaled)
        enc_meta["numeric_mean"] = mean.to_dict()
        enc_meta["numeric_std"] = std.to_dict()

    cat_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        lvls = sorted(df_train[col].astype("string").fillna("__MISSING__").unique().tolist())
        cat_levels[col] = lvls
    enc_meta["categorical_levels"] = cat_levels

    if categorical_cols:
        cat_parts: list[np.ndarray] = []
        for col in categorical_cols:
            vals = df_all[col].astype("string").fillna("__MISSING__")
            levels = cat_levels[col]
            idx = {v: i for i, v in enumerate(levels)}
            mat = np.zeros((len(df_all), len(levels)), dtype=np.float32)
            for r, v in enumerate(vals.tolist()):
                j = idx.get(v)
                if j is not None:
                    mat[r, j] = 1.0
            cat_parts.append(mat)
        if cat_parts:
            blocks.append(np.concatenate(cat_parts, axis=1))

    X = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(df_all), 0), dtype=np.float32)
    return X, enc_meta


def _try_sklearn_pipeline(
    df_train: pd.DataFrame,
    df_all: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    pca_dim: int | None,
    metric: str,
    use_sklearn_backend: bool,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    if not use_sklearn_backend:
        return np.empty((0, 0), dtype=np.float32), {}, False
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        return np.empty((0, 0), dtype=np.float32), {}, False

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    encoder = ColumnTransformer(transformers=transformers, remainder="drop")
    encoder.fit(df_train)
    X = encoder.transform(df_all)
    X = np.asarray(X, dtype=np.float32)

    svd_meta: dict[str, Any] = {}
    if pca_dim is not None and X.shape[1] > 1 and X.shape[0] > 2:
        n_comp = min(int(pca_dim), X.shape[1] - 1, X.shape[0] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=seed)
            X = svd.fit_transform(X).astype(np.float32)
            svd_meta["svd_components"] = int(n_comp)

    if metric == "cosine":
        X = _normalize_l2(X)

    meta = {
        "backend": "sklearn",
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        **svd_meta,
    }
    return X, meta, True


def build_feature_matrix(
    dataset_name: str,
    scope: str,
    processed_dir: Path,
    cache_dir: Path,
    seed: int,
    pca_dim: int | None,
    metric: str,
    use_sklearn_backend: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cache_root = cache_dir / dataset_name / scope
    cache_root.mkdir(parents=True, exist_ok=True)
    config_path = cache_root / "config.json"

    cfg = {
        "dataset_name": dataset_name,
        "scope": scope,
        "seed": seed,
        "pca_dim": pca_dim,
        "metric": metric,
    }

    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in cfg.items()):
            X = np.load(cache_root / "X.npy")
            y_bin = np.load(cache_root / "y_bin.npy")
            split_masks = np.load(cache_root / "split_masks.npz")
            split_indices = {k: split_masks[f"{k}_idx"].astype(np.int32) for k in ["train", "val", "test"]}
            return X, y_bin, split_indices

    df = _resolve_processed_scope(dataset_name, scope, processed_dir)
    metadata = load_metadata(dataset_name, processed_dir)
    feature_cols = metadata.get("feature_columns", [])
    if not feature_cols:
        raise ValueError(f"No feature_columns found in metadata for {dataset_name}")

    df_train = df.loc[df["_split"] == "train", feature_cols].copy()
    df_all = df[feature_cols].copy()

    X, enc_meta, used_sklearn = _try_sklearn_pipeline(
        df_train=df_train,
        df_all=df_all,
        feature_cols=feature_cols,
        seed=seed,
        pca_dim=pca_dim,
        metric=metric,
        use_sklearn_backend=use_sklearn_backend,
    )
    if not used_sklearn:
        X, enc_meta = _build_numpy_features(df_train, df_all, feature_cols)
        if pca_dim is not None and X.shape[1] > 1 and X.shape[0] > 2:
            n_comp = min(int(pca_dim), X.shape[1] - 1, X.shape[0] - 1)
            if n_comp >= 2:
                Xc = X - X.mean(axis=0, keepdims=True)
                _, _, vt = np.linalg.svd(Xc, full_matrices=False)
                X = (Xc @ vt[:n_comp].T).astype(np.float32)
        if metric == "cosine":
            X = _normalize_l2(X)
        enc_meta["backend"] = "numpy_fallback"

    y_bin = pd.to_numeric(df["y_bin"], errors="coerce").fillna(0).astype("int8").to_numpy()
    split_indices = _split_indices(df["_split"])

    np.save(cache_root / "X.npy", X.astype(np.float32))
    np.save(cache_root / "y_bin.npy", y_bin)
    np.savez(
        cache_root / "split_masks.npz",
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
    )

    node_index_map = {"row_index": list(range(len(df))), "split": df["_split"].astype(str).tolist()}
    (cache_root / "node_index_map.json").write_text(json.dumps(node_index_map), encoding="utf-8")

    cfg_out = dict(cfg)
    cfg_out.update(
        {
            "backend": enc_meta.get("backend", "numpy_fallback"),
            "feature_cols": feature_cols,
            "n_nodes": int(len(df)),
            "n_features": int(X.shape[1]),
        }
    )
    config_path.write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    (cache_root / "encoder_meta.json").write_text(json.dumps(enc_meta, default=str, indent=2), encoding="utf-8")
    return X.astype(np.float32), y_bin, split_indices
