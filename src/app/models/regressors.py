from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    model: MultiOutputRegressor
    feature_names: List[str]


def build_model(n_estimators: int, max_depth: int | None, n_jobs: int) -> MultiOutputRegressor:
    base = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base)


def build_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"return_{h}d"] = out.groupby("ticker")["close"].pct_change(h)
    return out


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = {"ticker", "begin", "open", "high", "low", "close", "volume"}
    target_cols = [c for c in df.columns if c.startswith("return_")]
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in target_cols]
    features = df[feature_cols].fillna(0.0)
    return features, feature_cols


def train_regressor(
    df: pd.DataFrame,
    horizons: List[int],
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> TrainResult:
    df_targets = build_targets(df, horizons).dropna(subset=["close"])
    df_targets = df_targets.dropna()

    X, feature_names = prepare_features(df_targets)
    y = df_targets[[f"return_{h}d" for h in horizons]].fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(n_estimators, max_depth, n_jobs)
    # Временная валидация при желании (можно расширить)
    tscv = TimeSeriesSplit(n_splits=3)
    for _train_idx, _val_idx in tscv.split(X_scaled):
        # Мини-валидация без метрик — просто чтобы убедиться, что модель тренируется
        model.fit(X_scaled[_train_idx], y.iloc[_train_idx])
    # Обучаем на всем наборе
    model.fit(X_scaled, y)
    model.scaler = scaler  # type: ignore[attr-defined]
    return TrainResult(model=model, feature_names=feature_names)


def predict_returns(train_result: TrainResult, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    X = df[train_result.feature_names].fillna(0.0)
    scaler = getattr(train_result.model, "scaler", None)
    X_scaled = scaler.transform(X) if scaler else X
    preds = train_result.model.predict(X_scaled)
    pred_df = pd.DataFrame(preds, columns=[f"p{h}" for h in horizons])
    return pred_df

