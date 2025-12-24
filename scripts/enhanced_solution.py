"""
Облегчённая обёртка: сохраняем точку входа, реальная логика в src.app.
"""

from src.app.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-enhanced"]))

