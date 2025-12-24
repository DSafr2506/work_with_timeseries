from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig
from .pipelines.enhanced import run_enhanced_pipeline


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast enhanced pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-enhanced", help="Запуск enhanced решения")
    run_parser.add_argument("--data-dir", type=Path, default=None, help="Каталог с сырыми данными (candles.csv, news.csv)")
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Куда писать submission")
    run_parser.add_argument("--submission-name", type=str, default=None, help="Имя файла submission")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = AppConfig.from_env()

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.submission_name:
        cfg.submission_name = args.submission_name

    if args.command == "run-enhanced":
        out = run_enhanced_pipeline(cfg)
        print(f"✅ Submission сохранён в {out}")
        return 0

    print("Неизвестная команда")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

