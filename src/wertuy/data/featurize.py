from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

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


def _save_X(X: Any, path_base: Path) -> str:
    if sparse.issparse(X):
        sparse.save_npz(path_base.with_suffix(".npz"), X)
        return path_base.with_suffix(".npz").name
    np.save(path_base.with_suffix(".npy"), np.asarray(X, dtype=np.float32))
    return path_base.with_suffix(".npy").name


def _load_X(path_base: Path, filename: str) -> Any:
    full = path_base.parent / filename
    if filename.endswith(".npz"):
        return sparse.load_npz(full)
    return np.load(full)


def build_feature_matrix(
    dataset_name: str,
    scope: str,
    processed_dir: Path,
    cache_dir: Path,
    seed: int,
    pca_dim: int | None,
    metric: str,
) -> tuple[Any, np.ndarray, dict[str, np.ndarray]]:
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
        if all(old.get(k) == v for k, v in cfg.items()) and "x_file" in old:
            X = _load_X(cache_root / "X", old["x_file"])
            y_bin = np.load(cache_root / "y_bin.npy")
            split_masks = np.load(cache_root / "split_masks.npz")
            split_indices = {k: split_masks[f"{k}_idx"].astype(np.int32) for k in ["train", "val", "test"]}
            return X, y_bin, split_indices

    df = _resolve_processed_scope(dataset_name, scope, processed_dir)
    metadata = load_metadata(dataset_name, processed_dir)
    feature_cols = metadata.get("feature_columns", [])
    if not feature_cols:
        raise ValueError(f"No feature_columns found in metadata for {dataset_name}")

    train_mask = df["_split"] == "train"
    df_train = df.loc[train_mask, feature_cols].copy()
    df_all = df[feature_cols].copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
        )
    encoder = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    encoder.fit(df_train)
    X = encoder.transform(df_all)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X, dtype=np.float32))
    else:
        X = X.tocsr().astype(np.float32)

    svd = None
    if pca_dim is not None:
        n_comp = min(int(pca_dim), max(2, X.shape[1] - 1), max(2, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X = svd.fit_transform(X).astype(np.float32)
        if metric == "cosine":
            X = normalize(X, norm="l2").astype(np.float32)
    else:
        if metric == "cosine":
            X = normalize(X, norm="l2")

    y_bin = pd.to_numeric(df["y_bin"], errors="coerce").fillna(0).astype("int8").to_numpy()
    split_indices = _split_indices(df["_split"])

    x_file = _save_X(X, cache_root / "X")
    np.save(cache_root / "y_bin.npy", y_bin)
    np.savez(
        cache_root / "split_masks.npz",
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
    )
    joblib.dump(encoder, cache_root / "encoder.joblib")
    if svd is not None:
        joblib.dump(svd, cache_root / "svd.joblib")

    node_index_map = {
        "row_index": list(range(len(df))),
        "split": df["_split"].astype(str).tolist(),
    }
    (cache_root / "node_index_map.json").write_text(json.dumps(node_index_map), encoding="utf-8")

    cfg_out = dict(cfg)
    cfg_out.update(
        {
            "x_file": x_file,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "n_nodes": int(len(df)),
            "n_features": int(X.shape[1]),
        }
    )
    config_path.write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    return X, y_bin, split_indices
