#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы META-Stock подхода
"""

import sys
import os
sys.path.append('/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/scripts')

from baseline_solution import BaselineSolution, METAStockPredictor
import pandas as pd
import numpy as np

def test_meta_stock():
    """Тестирование META-Stock подхода"""
    print("🧪 Тестирование META-Stock подхода...")
    

    train_candles = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv"
    test_candles = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv"
    train_news = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news.csv"
    test_news = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news_2.csv"
    
   
    for path in [train_candles, test_candles]:
        if not os.path.exists(path):
            print(f" Файл не найден: {path}")
            return False
    
    try:
        baseline = BaselineSolution(window_size=5, use_ml=True)
        baseline.load_data(train_candles, test_candles, train_news, test_news)
        
        print(f" Данные загружены:")
        print(f"   Train: {len(baseline.train_df)} строк")
        print(f"   Test: {len(baseline.test_df)} строк")
        print(f"   Total: {len(baseline.full_df)} строк")
        
        print("\n Тестирование META-Stock...")
        meta_predictor = METAStockPredictor(horizons=[1, 5, 10, 20], n_splits=3)
        
        sample_df = baseline.full_df.head(1000).copy()
        predictions_df = meta_predictor.predict_returns(sample_df)
        
        print(f" META-Stock предсказания готовы:")
        print(f"   Обработано строк: {len(predictions_df)}")
        
        # Проверяем наличие предсказаний
        pred_cols = [f'pred_return_{h}d' for h in [1, 5, 10, 20]]
        for col in pred_cols:
            if col in predictions_df.columns:
                print(f"   {col}: {predictions_df[col].mean():.6f} (среднее)")
        
        return True
        
    except Exception as e:
        print(f" Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_run():
    """Тестирование полного пайплайна BaselineSolution"""
    print("\n  Тестирование полного пайплайна...")
    
    try:
        baseline = BaselineSolution(window_size=5, use_ml=True)
        
        # Запускаем с META-Stock подходом
        baseline.run(
            train_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv",
            test_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv",
            train_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news.csv",
            test_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news_2.csv",
            output_path="test_meta_stock_submission.csv",
            use_meta_stock=True
        )
        
        # Проверяем созданный файл
        if os.path.exists("test_meta_stock_submission.csv"):
            submission = pd.read_csv("test_meta_stock_submission.csv")
            print(f" Submission файл создан: {len(submission)} строк")
            print(f"   Колонки: {list(submission.columns)}")
            return True
        else:
            print(" Submission файл не создан")
            return False
            
    except Exception as e:
        print(f" Ошибка при запуске пайплайна: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print(" ТЕСТИРОВАНИЕ META-STOCK ПОДХОДА")
    print("=" * 70)
    
    # Тест 1: META-Stock компонент
    test1_success = test_meta_stock()
    
    # Тест 2: Полный пайплайн
    test2_success = test_baseline_run()
    
    print("\n" + "=" * 70)
    print(" РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"META-Stock компонент: {' ПРОЙДЕН' if test1_success else ' ПРОВАЛЕН'}")
    print(f"Полный пайплайн: {' ПРОЙДЕН' if test2_success else ' ПРОВАЛЕН'}")
    
    if test1_success and test2_success:
        print("\n Все тесты пройдены успешно!")
    else:
        print("\n Некоторые тесты провалены. Проверьте ошибки выше.")
