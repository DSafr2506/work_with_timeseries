from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовые технические индикаторы по тикеру.
    Оставляем ограниченный набор, чтобы не усложнять вычисления.
    """
    df = df.copy()
    df["begin"] = pd.to_datetime(df["begin"])
    df.sort_values(["ticker", "begin"], inplace=True)

    grouped = df.groupby("ticker", group_keys=False)

    # Скользящие средние и отношения
    for window in (5, 10, 20):
        df[f"sma_{window}"] = grouped["close"].transform(lambda s: s.rolling(window).mean())
        df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]

    # Волатильность
    for window in (5, 10, 20):
        df[f"volatility_{window}"] = grouped["close"].transform(lambda s: s.pct_change().rolling(window).std())

    # RSI
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = grouped["close"].transform(_rsi)

    # Простые лаги
    for lag in (1, 2, 3, 5, 10):
        df[f"close_lag_{lag}"] = grouped["close"].shift(lag)
        df[f"volume_lag_{lag}"] = grouped["volume"].shift(lag)

    # Временные признаки
    df["day_of_week"] = df["begin"].dt.dayofweek
    df["month"] = df["begin"].dt.month
    df["quarter"] = df["begin"].dt.quarter

    return df

