# Finam x HSE Trade AI Hack - Forecast

## Описание проекта

Проект для прогнозирования доходности акций на горизонтах от 1 до 20 дней. Решение использует технические индикаторы ценовых данных и новостные признаки для обучения модели машинного обучения.

## Требования

- Python 3.10 или выше
- 4+ GB RAM (рекомендуется 8+ GB)
- 2+ GB свободного места на диске
- Операционная система: Windows, macOS, Linux

## Установка

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd finam-x-hse-trade-ai-hack-forecast
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# Активация окружения
# На Windows:
venv\Scripts\activate
# На macOS/Linux:
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

Дополнительные опциональные зависимости:

```bash
pip install openai catboost xgboost lightgbm
```

## Структура проекта

```
finam-x-hse-trade-ai-hack-forecast/
├── src/
│   ├── app/
│   │   ├── cli.py                    # CLI интерфейс
│   │   ├── config.py                  # Конфигурация
│   │   ├── data_loader.py             # Загрузка данных
│   │   ├── features/
│   │   │   ├── price.py               # Ценовые признаки
│   │   │   └── news.py                # Новостные признаки
│   │   ├── models/
│   │   │   └── regressors.py         # Модели регрессии
│   │   └── pipelines/
│   │       └── enhanced.py            # Enhanced пайплайн
│   └── backend/
│       └── app.py                     # FastAPI backend
├── scripts/
│   ├── enhanced_solution.py          # Обертка для обратной совместимости
│   └── baseline_solution.py          # Baseline решение
├── data/
│   ├── raw/participants/
│   │   ├── candles.csv                # Обучающие свечи
│   │   ├── news.csv                   # Обучающие новости
│   │   ├── candles_2.csv             # Тестовые свечи
│   │   └── news_2.csv                 # Тестовые новости
│   └── processed/participants/
│       └── (обработанные файлы)
├── outputs/                           # Выходные файлы
├── requirements.txt                   # Зависимости
├── pyproject.toml                     # Конфигурация проекта
└── README.md                          # Этот файл
```

## Использование

### CLI интерфейс

Запуск enhanced решения через командную строку:

```bash
python -m src.app run-enhanced \
  --data-dir data/raw/participants \
  --output-dir outputs \
  --submission-name enhanced_submission.csv
```

Переменные окружения (опционально):

- FORECAST_DATA_DIR - путь к каталогу с данными
- FORECAST_OUTPUT_DIR - каталог для выходных файлов
- FORECAST_SUBMISSION - имя файла submission
- FORECAST_N_ESTIMATORS - количество деревьев (по умолчанию 100)
- FORECAST_MAX_DEPTH - максимальная глубина деревьев (по умолчанию 10)
- FORECAST_N_JOBS - количество параллельных процессов (по умолчанию -1)

Пример с переменными окружения:

```bash
export FORECAST_DATA_DIR=data/raw/participants
export FORECAST_OUTPUT_DIR=outputs
export FORECAST_SUBMISSION=my_submission.csv
python -m src.app run-enhanced
```

### Backend API

Запуск FastAPI сервера:

```bash
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000
```

Или через entrypoint (после установки пакета):

```bash
forecast-api
```

#### API эндпоинты

**GET /health**

Проверка работоспособности сервиса.

Ответ:
```json
{
  "status": "ok"
}
```

**POST /run**

Асинхронный запуск enhanced пайплайна.

Тело запроса (опционально):
```json
{
  "data_dir": "data/raw/participants",
  "output_dir": "outputs",
  "submission_name": "enhanced_submission.csv"
}
```

Ответ:
```json
{
  "status": "started",
  "submission_path": null,
  "metrics": null
}
```

**GET /status**

Получение статуса последнего запуска и метрик.

Ответ:
```json
{
  "status": "done",
  "started_at": 1234567890.123,
  "finished_at": 1234567900.456,
  "runtime_seconds": 10.333,
  "submission_path": "outputs/enhanced_submission.csv",
  "metrics": {
    "train_rows": 50000,
    "test_rows": 10000,
    "n_features": 150,
    "runtime_seconds": 10.333
  },
  "error_message": null
}
```

Возможные значения status:
- idle - нет запущенных задач
- pending - задача поставлена в очередь
- running - задача выполняется
- done - задача завершена успешно
- error - произошла ошибка

#### Пример использования API

```bash
# Запуск задачи
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "data_dir": "data/raw/participants",
    "output_dir": "outputs",
    "submission_name": "enhanced_submission.csv"
  }'

# Проверка статуса
curl http://localhost:8000/status
```

## Метрики

Пайплайн собирает следующие метрики:

- train_rows - количество строк в обучающей выборке
- test_rows - количество строк в тестовой выборке
- n_features - количество признаков, используемых моделью
- runtime_seconds - полное время выполнения пайплайна в секундах

Метрики доступны через:
- CLI: выводятся в консоль после завершения
- API: через эндпоинт GET /status после завершения задачи

## Решение

### Enhanced Solution

Основное решение проекта, реализованное в модуле `src/app/pipelines/enhanced.py`.

**Модель:**
- RandomForestRegressor с настраиваемыми параметрами
- MultiOutputRegressor для одновременного предсказания всех горизонтов (1-20 дней)

**Признаки:**
- Ценовые признаки: технические индикаторы (SMA, EMA, RSI, Bollinger Bands, MACD и др.)
- Новостные признаки: агрегированные дневные признаки из новостей
- Временные признаки: день недели, месяц, квартал

**Валидация:**
- TimeSeriesSplit для временной валидации
- Предотвращение data leakage через правильную агрегацию новостей

**Выходной формат:**

CSV файл с колонками:
- ticker - тикер акции
- p1, p2, ..., p20 - предсказанные вероятности роста на горизонты 1-20 дней

## Формат данных

### Входные данные

**Свечи (candles.csv, candles_2.csv):**
- ticker - тикер акции
- begin - дата и время начала свечи
- open - цена открытия
- high - максимальная цена
- low - минимальная цена
- close - цена закрытия
- volume - объем торгов

**Новости (news.csv, news_2.csv):**
- publish_date - дата публикации
- title - заголовок новости
- publication - источник публикации

### Выходные данные

**Submission файл:**
CSV файл с колонками ticker, p1, p2, ..., p20, где p1-p20 - предсказанные вероятности роста на горизонты 1-20 дней.

## Часто задаваемые вопросы

**Q: Какой файл запускать первым?**

A: Рекомендуется использовать новый CLI: `python -m src.app run-enhanced`. Старый скрипт `scripts/enhanced_solution.py` сохранен для обратной совместимости.

**Q: Нужен ли OpenRouter API ключ?**

A: Нет, не обязателен. Решение работает без внешних API, используя только локальные данные и признаки.

**Q: Сколько времени займет обучение?**

A: Время зависит от объема данных и параметров модели. Обычно 5-15 минут для стандартного набора данных.

**Q: Какие файлы нужны для запуска?**

A: Минимально необходимые файлы в папке `data/raw/participants/`:
- candles.csv - обучающие свечи
- candles_2.csv - тестовые свечи
- news.csv и news_2.csv опциональны, но улучшают качество модели

**Q: Что делать если не хватает памяти?**

A: 
1. Закройте другие программы
2. Уменьшите параметр FORECAST_N_ESTIMATORS
3. Уменьшите параметр FORECAST_N_JOBS

**Q: Можно ли запустить без интернета?**

A: Да, решение полностью работает офлайн, не требует интернет-соединения.

**Q: Как изменить параметры модели?**

A: Используйте переменные окружения FORECAST_N_ESTIMATORS, FORECAST_MAX_DEPTH, FORECAST_N_JOBS или передайте их через CLI (если поддерживается).

## Устранение проблем

**Проблема: "No module named 'X'"**

Решение: Установите недостающий модуль:
```bash
pip install X
```

**Проблема: "No such file or directory"**

Решение: Проверьте, что вы находитесь в правильной директории и что файлы данных существуют:
```bash
pwd
ls -la data/raw/participants/
```

**Проблема: "NumPy version conflict"**

Решение: Создайте новое виртуальное окружение:
```bash
python -m venv new_env
source new_env/bin/activate  # или new_env\Scripts\activate на Windows
pip install -r requirements.txt
```

**Проблема: Backend не запускается**

Решение: Убедитесь, что установлены все зависимости:
```bash
pip install fastapi uvicorn[standard]
```

## Дополнительная документация

- docs/task.md - Описание задачи хакатона
- docs/evaluation.md - Метрики оценки
- docs/data.md - Описание формата данных


