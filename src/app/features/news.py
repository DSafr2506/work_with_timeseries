from __future__ import annotations

import pandas as pd


def merge_daily_news_features(
    candles: pd.DataFrame, news_daily: pd.DataFrame, date_col: str = "publish_date"
) -> pd.DataFrame:
    """
    Мержит готовые дневные новостные фичи с котировками по дате.
    Ожидает, что news_daily уже агрегирован по дням.
    """
    if news_daily is None or news_daily.empty:
        return candles

    df = candles.copy()
    df["begin"] = pd.to_datetime(df["begin"])
    news = news_daily.copy()
    news[date_col] = pd.to_datetime(news[date_col])

    # Присоединяем по дате (без тикеров, т.к. новости общие)
    df = df.merge(
        news,
        left_on=df["begin"].dt.normalize(),
        right_on=news[date_col].dt.normalize(),
        how="left",
    )
    df.drop(columns=["key_0", date_col], errors="ignore", inplace=True)
    return df

