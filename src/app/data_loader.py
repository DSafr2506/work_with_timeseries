from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from .paths import DataPaths


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path}")
    return pd.read_csv(path)


def load_candles(paths: DataPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка train/test свечей."""
    train = _read_csv_safe(paths.train_candles)
    test = _read_csv_safe(paths.test_candles)
    return train, test


def load_news_daily_features(paths: DataPaths) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Попытка загрузить уже посчитанные дневные новостные фичи.
    Возвращает None, если файлов нет – пайплайн продолжит без них.
    """
    train_feat = _read_csv_safe(paths.processed_train_news) if paths.processed_train_news.exists() else None
    test_feat = _read_csv_safe(paths.processed_test_news) if paths.processed_test_news.exists() else None
    return train_feat, test_feat

