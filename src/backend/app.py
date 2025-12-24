from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from src.app.config import AppConfig
from src.app.pipelines.enhanced import run_enhanced_with_metrics


app = FastAPI(title="Forecast Enhanced Backend", version="0.1.0")


@dataclass
class JobState:
    started_at: float
    finished_at: Optional[float] = None
    status: str = "pending"  # pending | running | done | error
    submission_path: Optional[Path] = None
    metrics: Dict[str, float] | None = None
    error_message: Optional[str] = None


JOB: Optional[JobState] = None


class RunRequest(BaseModel):
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    submission_name: Optional[str] = None


class RunResponse(BaseModel):
    status: str
    submission_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class JobStatusResponse(BaseModel):
    status: str
    started_at: Optional[float]
    finished_at: Optional[float]
    runtime_seconds: Optional[float]
    submission_path: Optional[str]
    metrics: Optional[Dict[str, float]]
    error_message: Optional[str]


def _run_job(cfg: AppConfig) -> None:
    global JOB
    if JOB is None:
        return
    JOB.status = "running"
    try:
        out_path, metrics = run_enhanced_with_metrics(cfg)
        JOB.submission_path = out_path
        JOB.metrics = metrics
        JOB.status = "done"
    except Exception as exc:  # pragma: no cover - логика ошибок
        JOB.status = "error"
        JOB.error_message = str(exc)
    finally:
        JOB.finished_at = time.time()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    """
    Асинхронный запуск enhanced-пайплайна.
    Возвращает текущий статус и, когда задача закончится, путь к submission и метрики.
    """
    global JOB
    cfg = AppConfig.from_env()
    if request.data_dir:
        cfg.data_dir = request.data_dir
    if request.output_dir:
        cfg.output_dir = request.output_dir
    if request.submission_name:
        cfg.submission_name = request.submission_name

    JOB = JobState(started_at=time.time(), status="pending")
    background_tasks.add_task(_run_job, cfg)

    return RunResponse(status="started", submission_path=None, metrics=None)


@app.get("/status", response_model=JobStatusResponse)
async def job_status() -> JobStatusResponse:
    """
    Текущий статус последнего запуска и метрики (если уже посчитаны).
    """
    if JOB is None:
        return JobStatusResponse(
            status="idle",
            started_at=None,
            finished_at=None,
            runtime_seconds=None,
            submission_path=None,
            metrics=None,
            error_message=None,
        )

    runtime = None
    if JOB.finished_at and JOB.started_at:
        runtime = JOB.finished_at - JOB.started_at

    return JobStatusResponse(
        status=JOB.status,
        started_at=JOB.started_at,
        finished_at=JOB.finished_at,
        runtime_seconds=runtime,
        submission_path=str(JOB.submission_path) if JOB.submission_path else None,
        metrics=JOB.metrics,
        error_message=JOB.error_message,
    )


def run() -> None:
    """
    Точка входа для запуска через `uvicorn src.backend.app:app`.
    Оставлена для удобства, но запускать лучше так:

    uvicorn src.backend.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn

    uvicorn.run("src.backend.app:app", host="0.0.0.0", port=8000, reload=False)


