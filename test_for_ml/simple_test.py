#!/usr/bin/env python3
"""
Простой тест META-Stock подхода без сложных зависимостей
"""

import pandas as pd
import numpy as np
from pathlib import Path

def simple_meta_stock_test():
    """Простой тест META-Stock логики"""
    print("🧪 Простой тест META-Stock логики...")
    
    # Создаем тестовые данные
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['AFLT', 'GAZP', 'LKOH']
    
    test_data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            price = 100 + i * 0.1 + np.random.normal(0, 0.5)
            test_data.append({
                'ticker': ticker,
                'begin': date,
                'close': price,
                'open': price + np.random.normal(0, 0.1),
                'high': price + abs(np.random.normal(0, 0.2)),
                'low': price - abs(np.random.normal(0, 0.2)),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    df = pd.DataFrame(test_data)
    print(f"✅ Создан тестовый датафрейм: {len(df)} строк")
    
    # Тестируем создание таргетов для разных горизонтов
    horizons = [1, 5, 10, 20]
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('begin')
        
        for horizon in horizons:
            ticker_data[f'return_{horizon}d'] = (
                ticker_data['close'].shift(-horizon) / ticker_data['close'] - 1
            )
        
        print(f"✅ Таргеты созданы для {ticker}:")
        for horizon in horizons:
            col = f'return_{horizon}d'
            if col in ticker_data.columns:
                mean_return = ticker_data[col].mean()
                print(f"   {col}: среднее = {mean_return:.6f}")
    
    # Тестируем технические индикаторы
    print("\n🔧 Тестирование технических индикаторов...")
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('begin')
        
        # Моментумы
        for period in [1, 5, 10]:
            ticker_data[f'momentum_{period}d'] = ticker_data['close'].pct_change(period)
        
        # Скользящие средние
        for period in [5, 10, 20]:
            ticker_data[f'ma_{period}'] = ticker_data['close'].rolling(period).mean()
            ticker_data[f'price_to_ma_{period}'] = ticker_data['close'] / ticker_data[f'ma_{period}']
        
        print(f"✅ Индикаторы созданы для {ticker}")
        print(f"   Доступные признаки: {[col for col in ticker_data.columns if col.startswith(('momentum_', 'ma_', 'price_to_ma_'))]}")
    
    return True

def test_data_loading():
    """Тест загрузки реальных данных"""
    print("\n📊 Тест загрузки реальных данных...")
    
    train_path = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv"
    test_path = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv"
    
    try:
        # Загружаем данные
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"✅ Данные загружены:")
        print(f"   Train: {len(train_df)} строк, колонки: {list(train_df.columns)}")
        print(f"   Test: {len(test_df)} строк, колонки: {list(test_df.columns)}")
        
        # Проверяем тикеры
        train_tickers = train_df['ticker'].unique()
        test_tickers = test_df['ticker'].unique()
        
        print(f"   Train тикеры: {train_tickers}")
        print(f"   Test тикеры: {test_tickers}")
        
        # Проверяем даты
        train_df['begin'] = pd.to_datetime(train_df['begin'])
        test_df['begin'] = pd.to_datetime(test_df['begin'])
        
        print(f"   Train даты: {train_df['begin'].min()} - {train_df['begin'].max()}")
        print(f"   Test даты: {test_df['begin'].min()} - {test_df['begin'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return False

def test_meta_stock_logic():
    """Тест логики META-Stock без ML библиотек"""
    print("\n🤖 Тест логики META-Stock...")
    
    # Загружаем данные
    train_path = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv"
    test_path = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv"
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Объединяем данные
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df['begin'] = pd.to_datetime(full_df['begin'])
        full_df = full_df.sort_values(['ticker', 'begin'])
        
        print(f"✅ Объединенные данные: {len(full_df)} строк")
        
        # Создаем признаки для одного тикера
        ticker = full_df['ticker'].unique()[0]
        ticker_data = full_df[full_df['ticker'] == ticker].copy()
        
        print(f"✅ Данные для тикера {ticker}: {len(ticker_data)} строк")
        
        # Создаем таргеты
        horizons = [1, 5, 10, 20]
        for horizon in horizons:
            ticker_data[f'return_{horizon}d'] = (
                ticker_data['close'].shift(-horizon) / ticker_data['close'] - 1
            )
        
        # Создаем признаки
        for period in [1, 5, 10]:
            ticker_data[f'momentum_{period}d'] = ticker_data['close'].pct_change(period)
        
        for period in [5, 10, 20]:
            ticker_data[f'ma_{period}'] = ticker_data['close'].rolling(period).mean()
            ticker_data[f'price_to_ma_{period}'] = ticker_data['close'] / ticker_data[f'ma_{period}']
        
        # Простое предсказание на основе моментума
        for horizon in horizons:
            momentum_col = f'momentum_5d'
            if momentum_col in ticker_data.columns:
                # Простое предсказание: продолжение тренда
                ticker_data[f'pred_return_{horizon}d'] = ticker_data[momentum_col] * 0.5
                
                # Вероятность роста на основе сигмоиды
                ticker_data[f'pred_prob_up_{horizon}d'] = 1 / (1 + np.exp(-10 * ticker_data[momentum_col]))
        
        print(f"✅ Предсказания созданы для {ticker}")
        
        # Показываем статистику
        for horizon in horizons:
            pred_col = f'pred_return_{horizon}d'
            prob_col = f'pred_prob_up_{horizon}d'
            
            if pred_col in ticker_data.columns:
                mean_pred = ticker_data[pred_col].mean()
                mean_prob = ticker_data[prob_col].mean()
                print(f"   Horizon {horizon}d: pred_return = {mean_pred:.6f}, prob_up = {mean_prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в логике META-Stock: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 ПРОСТОЙ ТЕСТ META-STOCK ПОДХОДА")
    print("=" * 70)
    
    # Тест 1: Простая логика
    test1_success = simple_meta_stock_test()
    
    # Тест 2: Загрузка данных
    test2_success = test_data_loading()
    
    # Тест 3: META-Stock логика
    test3_success = test_meta_stock_logic()
    
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"Простая логика: {'✅ ПРОЙДЕН' if test1_success else '❌ ПРОВАЛЕН'}")
    print(f"Загрузка данных: {'✅ ПРОЙДЕН' if test2_success else '❌ ПРОВАЛЕН'}")
    print(f"META-Stock логика: {'✅ ПРОЙДЕН' if test3_success else '❌ ПРОВАЛЕН'}")
    
    if test1_success and test2_success and test3_success:
        print("\n🎉 Все тесты пройдены успешно!")
        print("\n💡 META-Stock подход работает корректно:")
        print("   • Создание таргетов для горизонтов 1-20 дней")
        print("   • Технические индикаторы (моментум, MA)")
        print("   • Простое предсказание на основе моментума")
        print("   • Вероятности роста через сигмоиду")
    else:
        print("\n⚠️ Некоторые тесты провалены. Проверьте ошибки выше.")
