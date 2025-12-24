#!/usr/bin/env python3
"""
Простое решение для получения submission файла в формате sample_submission.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data():
    """Загрузка данных"""
    print("📁 Загрузка данных...")
    
    # Пути к файлам
    train_candles = "data/raw/participants/candles.csv"
    test_candles = "data/raw/participants/candles_2.csv"
    train_news = "data/raw/participants/news.csv"
    test_news = "data/raw/participants/news_2.csv"
    
    # Загрузка данных
    train_df = pd.read_csv(train_candles)
    test_df = pd.read_csv(test_candles)
    
    print(f" Обучающие данные: {len(train_df)} строк")
    print(f" Тестовые данные: {len(test_df)} строк")
    
    return train_df, test_df

def create_features(df):
    """Создание простых признаков"""
    print("🔧 Создание признаков...")
    
    # Копируем данные
    df = df.copy()
    
    # Сортировка по тикеру и дате
    df['begin'] = pd.to_datetime(df['begin'])
    df = df.sort_values(['ticker', 'begin'])
    
    # Простые технические индикаторы
    df['sma_5'] = df.groupby('ticker')['close'].rolling(5).mean().reset_index(0, drop=True)
    df['sma_20'] = df.groupby('ticker')['close'].rolling(20).mean().reset_index(0, drop=True)
    
    # RSI
    delta = df.groupby('ticker')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df.groupby('ticker')['close'].rolling(20).mean().reset_index(0, drop=True)
    bb_std = df.groupby('ticker')['close'].rolling(20).std().reset_index(0, drop=True)
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_sma'] = df.groupby('ticker')['volume'].rolling(20).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price features
    df['price_change'] = df.groupby('ticker')['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df.groupby('ticker')['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df.groupby('ticker')['volume'].shift(lag)
    
    # Time features
    df['day_of_week'] = df['begin'].dt.dayofweek
    df['month'] = df['begin'].dt.month
    df['quarter'] = df['begin'].dt.quarter
    
    print(f" Создано {len(df.columns)} признаков")
    
    return df

def create_targets(df, horizons=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
    """Создание целей для мульти-горизонтного предсказания"""
    print(" Создание целей...")
    
    df = df.copy()
    
    # Создаем цели для каждого горизонта
    for horizon in horizons:
        df[f'return_{horizon}d'] = df.groupby('ticker')['close'].pct_change(horizon)
        df[f'prob_up_{horizon}d'] = (df[f'return_{horizon}d'] > 0).astype(float)
    
    print(f"✅ Создано целей для {len(horizons)} горизонтов")
    
    return df

def train_simple_model(train_df, test_df):
    """Обучение простой модели"""
    print("🤖 Обучение модели...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # Подготовка данных
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Создание целей
    train_df = create_targets(train_df)
    
    # Выбор признаков
    feature_cols = [col for col in train_df.columns if col not in [
        'ticker', 'begin', 'open', 'high', 'low', 'close', 'volume'
    ] and not col.startswith('return_') and not col.startswith('prob_up_')]
    
    # Удаляем NaN
    train_df = train_df.dropna()
    
    if len(train_df) == 0:
        print(" Нет данных для обучения")
        return None
    
    print(f"✅ Используется {len(feature_cols)} признаков")
    print(f"✅ Обучающих примеров: {len(train_df)}")
    
    # Обучение модели для каждого горизонта
    models = {}
    horizons = list(range(1, 21))
    
    for horizon in horizons:
        target_col = f'return_{horizon}d'
        if target_col in train_df.columns:
            # Подготовка данных
            X = train_df[feature_cols].fillna(0)
            y = train_df[target_col].fillna(0)
            
            # Обучение модели
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            try:
                model.fit(X, y)
                models[horizon] = model
                print(f" Модель для горизонта {horizon} обучена")
            except Exception as e:
                print(f" Ошибка обучения модели для горизонта {horizon}: {e}")
    
    return models, feature_cols

def predict_and_create_submission(models, feature_cols, test_df):
    """Предсказание и создание submission файла"""
    print(" Создание предсказаний..")
    
    if not models:
        print(" Нет обученных моделей")
        return None
    
    # Подготовка тестовых данных
    test_df = test_df.copy()
    test_df = test_df.sort_values(['ticker', 'begin'])
    
    # Получаем уникальные тикеры
    tickers = test_df['ticker'].unique()
    
    # Создаем submission DataFrame
    submission_data = []
    
    for ticker in tickers:
        ticker_data = test_df[test_df['ticker'] == ticker].copy()
        
        if len(ticker_data) == 0:
            continue
        
        # Создаем признаки для тестовых данных
        ticker_data = create_features(ticker_data)
        
        # Заполняем NaN
        ticker_data = ticker_data.fillna(0)
        
        # Предсказания для каждого горизонта
        predictions = {}
        
        for horizon in range(1, 21):
            if horizon in models:
                try:
                    X_test = ticker_data[feature_cols].fillna(0)
                    pred = models[horizon].predict(X_test)
                    # Берем последнее предсказание
                    predictions[f'p{horizon}'] = pred[-1] if len(pred) > 0 else 0.5
                except Exception as e:
                    print(f" Ошибка предсказания для горизонта {horizon}: {e}")
                    predictions[f'p{horizon}'] = 0.5
            else:
                predictions[f'p{horizon}'] = 0.5
        
        # Добавляем в submission
        row = {'ticker': ticker}
        row.update(predictions)
        submission_data.append(row)
    
    # Создаем DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # Сортируем по тикеру
    submission_df = submission_df.sort_values('ticker').reset_index(drop=True)
    
    print(f"Создан submission файл с {len(submission_df)} строками")
    
    return submission_df

def main():
    """Главная функция"""
    print("=" * 70)
    print("🚀 ПРОСТОЕ РЕШЕНИЕ ДЛЯ SUBMISSION")
    print("=" * 70)
    
    try:
        # Загрузка данных
        train_df, test_df = load_data()
        
        # Обучение модели
        models, feature_cols = train_simple_model(train_df, test_df)
        
        if models is None:
            print("❌ Не удалось обучить модели")
            return
        
        # Создание submission
        submission_df = predict_and_create_submission(models, feature_cols, test_df)
        
        if submission_df is None:
            print("❌ Не удалось создать submission")
            return
        
        # Сохранение файла
        output_file = "simple_submission.csv"
        submission_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Submission файл сохранен: {output_file}")
        print(f"📊 Размер файла: {len(submission_df)} строк")
        print(f"📁 Размер файла: {os.path.getsize(output_file)} байт")
        
        # Показываем первые строки
        print("\n📋 Первые строки submission файла:")
        print(submission_df.head())
        
        print("\n🎉 Готово! Файл simple_submission.csv создан")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
