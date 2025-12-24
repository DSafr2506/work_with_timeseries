from __future__ import annotations

from pathlib import Path


class DataPaths:
    """
    Хранилище путей к файлам данных.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.train_candles = base_dir / "candles.csv"
        self.test_candles = base_dir / "candles_2.csv"
        self.train_news = base_dir / "news.csv"
        self.test_news = base_dir / "news_2.csv"
        self.processed_train_news = base_dir.parent / "processed" / "participants" / "train_news_daily_features.csv"
        self.processed_test_news = base_dir.parent / "processed" / "participants" / "test_news_daily_features.csv"

    def as_dict(self) -> dict[str, Path]:
        return {
            "train_candles": self.train_candles,
            "test_candles": self.test_candles,
            "train_news": self.train_news,
            "test_news": self.test_news,
            "processed_train_news": self.processed_train_news,
            "processed_test_news": self.processed_test_news,
        }

