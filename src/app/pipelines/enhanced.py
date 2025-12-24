from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import time
import pandas as pd

from ..config import AppConfig, ensure_output_dir
from ..data_loader import load_candles, load_news_daily_features
from ..features.price import add_price_features
from ..features.news import merge_daily_news_features
from ..models.regressors import TrainResult, predict_returns, train_regressor
from ..paths import DataPaths


HORIZONS: List[int] = list(range(1, 21))


def build_training_frame(train_df: pd.DataFrame, news_daily: pd.DataFrame | None) -> pd.DataFrame:
    df = add_price_features(train_df)
    df = merge_daily_news_features(df, news_daily)
    return df


def build_inference_frame(test_df: pd.DataFrame, news_daily: pd.DataFrame | None) -> pd.DataFrame:
    df = add_price_features(test_df)
    df = merge_daily_news_features(df, news_daily)
    return df


def run_enhanced_with_metrics(cfg: AppConfig) -> Tuple[Path, Dict[str, float]]:
    """
    Запуск enhanced-пайплайна с возвратом пути к submission и метрик.
    Метрики:
    - train_rows/test_rows: объём данных
    - n_features: количество фичей на входе модели
    - runtime_seconds: полное время выполнения пайплайна
    """
    t0 = time.perf_counter()

    paths = DataPaths(cfg.data_dir)
    train_candles, test_candles = load_candles(paths)
    train_news_daily, test_news_daily = load_news_daily_features(paths)

    train_df = build_training_frame(train_candles, train_news_daily)
    test_df = build_inference_frame(test_candles, test_news_daily)

    train_result: TrainResult = train_regressor(
        train_df,
        horizons=HORIZONS,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
    )

    preds = predict_returns(train_result, test_df, HORIZONS)
    submission = (
        pd.concat([test_df[["ticker"]].reset_index(drop=True), preds], axis=1)
        .drop_duplicates(subset=["ticker"], keep="last")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    output_dir = ensure_output_dir(cfg)
    out_path = output_dir / cfg.submission_name
    submission.to_csv(out_path, index=False)

    runtime = time.perf_counter() - t0
    metrics: Dict[str, float] = {
        "train_rows": float(len(train_df)),
        "test_rows": float(len(test_df)),
        "n_features": float(len(train_result.feature_names)),
        "runtime_seconds": float(runtime),
    }

    return out_path, metrics


def run_enhanced_pipeline(cfg: AppConfig) -> Path:
    """
    Обёртка для обратной совместимости: возвращает только путь к submission.
    """
    out_path, _ = run_enhanced_with_metrics(cfg)
    return out_path

