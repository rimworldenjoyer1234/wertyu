from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureEncoder:
    pipeline: ColumnTransformer
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    pca: PCA | None = None


def fit_encoder(df_train: pd.DataFrame, feature_cols: list[str], pca_dim: int | None = None) -> FeatureEncoder:
    x = df_train[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(with_mean=True, with_std=True), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(x)

    pca = None
    if pca_dim is not None:
        transformed = preprocessor.transform(x)
        transformed = np.asarray(transformed, dtype=np.float32)
        n_components = min(pca_dim, transformed.shape[1], transformed.shape[0])
        if n_components >= 2:
            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
            pca.fit(transformed)

    return FeatureEncoder(
        pipeline=preprocessor,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        pca=pca,
    )


def transform(encoder_object: FeatureEncoder, df: pd.DataFrame) -> np.ndarray:
    x = df[encoder_object.feature_cols].copy()
    transformed = encoder_object.pipeline.transform(x)
    transformed = np.asarray(transformed, dtype=np.float32)
    if encoder_object.pca is not None:
        transformed = encoder_object.pca.transform(transformed).astype(np.float32)
    return transformed.astype(np.float32)
