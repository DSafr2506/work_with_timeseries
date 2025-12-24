from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DATA_DIR_DEFAULT = Path("data/raw/participants")
OUTPUT_DIR_DEFAULT = Path("outputs")


@dataclass
class AppConfig:
    """
    Конфигурация приложения.
    """

    data_dir: Path = DATA_DIR_DEFAULT
    output_dir: Path = OUTPUT_DIR_DEFAULT
    submission_name: str = "enhanced_submission.csv"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: Optional[int] = 12
    n_jobs: int = -1

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Чтение конфигурации из переменных окружения с безопасными значениями по умолчанию."""
        data_dir = Path(os.getenv("FORECAST_DATA_DIR", DATA_DIR_DEFAULT))
        output_dir = Path(os.getenv("FORECAST_OUTPUT_DIR", OUTPUT_DIR_DEFAULT))
        submission_name = os.getenv("FORECAST_SUBMISSION", "enhanced_submission.csv")

        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            submission_name=submission_name,
            random_state=int(os.getenv("FORECAST_RANDOM_STATE", "42")),
            n_estimators=int(os.getenv("FORECAST_N_ESTIMATORS", "200")),
            max_depth=int(os.getenv("FORECAST_MAX_DEPTH", "12"))
            if os.getenv("FORECAST_MAX_DEPTH")
            else None,
            n_jobs=int(os.getenv("FORECAST_N_JOBS", "-1")),
        )


def ensure_output_dir(cfg: AppConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir

