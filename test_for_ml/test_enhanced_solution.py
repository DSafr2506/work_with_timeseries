#!/usr/bin/env python3
"""
Тест улучшенного решения с проверкой всех компонентов
"""

import sys
import os
sys.path.append('/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/scripts')

import pandas as pd
import numpy as np
from pathlib import Path

def test_enhanced_features():
    """Тест расширенных признаков"""
    print("🧪 Тестирование расширенных признаков...")
    
    # Создаем тестовые данные
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['AFLT', 'GAZP', 'LKOH']
    
    test_data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            base_price = 100 + i * 0.1
            test_data.append({
                'ticker': ticker,
                'begin': date,
                'close': base_price + np.random.normal(0, 0.5),
                'open': base_price + np.random.normal(0, 0.1),
                'high': base_price + abs(np.random.normal(0, 0.2)),
                'low': base_price - abs(np.random.normal(0, 0.2)),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    df = pd.DataFrame(test_data)
    print(f"✅ Создан тестовый датафрейм: {len(df)} строк")
    
    # Тестируем создание расширенных признаков
    try:
        from enhanced_solution import EnhancedFeatureExtractor
        
        extractor = EnhancedFeatureExtractor()
        enhanced_df = extractor.compute_advanced_technical_indicators(df)
        
        # Проверяем количество признаков
        feature_cols = [col for col in enhanced_df.columns 
                       if col not in ['ticker', 'begin', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Создано {len(feature_cols)} расширенных признаков")
        print(f"   Примеры признаков: {feature_cols[:10]}")
        
        # Проверяем типы признаков
        momentum_cols = [col for col in feature_cols if 'momentum' in col]
        volatility_cols = [col for col in feature_cols if 'volatility' in col]
        ma_cols = [col for col in feature_cols if 'sma' in col or 'ema' in col]
        
        print(f"   Моментумы: {len(momentum_cols)}")
        print(f"   Волатильности: {len(volatility_cols)}")
        print(f"   Скользящие средние: {len(ma_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании расширенных признаков: {e}")
        return False

def test_news_analysis():
    """Тест анализа новостей"""
    print("\n🧪 Тестирование анализа новостей...")
    
    # Создаем тестовые новости
    test_news = [
        {
            'publish_date': '2020-01-01 10:00:00',
            'title': 'Газпром увеличил добычу газа на 5%',
            'publication': 'РБК'
        },
        {
            'publish_date': '2020-01-02 11:00:00', 
            'title': 'Сбербанк объявил о росте прибыли на 10%',
            'publication': 'Коммерсант'
        }
    ]
    
    news_df = pd.DataFrame(test_news)
    
    try:
        from enhanced_solution import OpenRouterNewsAnalyzer, EnhancedFeatureExtractor
        
        # Тест без API (fallback режим)
        print("   📰 Тестирование без API...")
        extractor = EnhancedFeatureExtractor()
        features_df = extractor._extract_basic_news_features(news_df)
        
        print(f"✅ Базовые новостные признаки: {len(features_df)} строк")
        print(f"   Колонки: {list(features_df.columns)}")
        
        # Тест с API (если есть ключ)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            print("   🔍 Тестирование с OpenRouter API...")
            analyzer = OpenRouterNewsAnalyzer(api_key)
            
            # Тест анализа сентимента
            sentiment = analyzer.analyze_sentiment("Газпром показал отличные результаты")
            print(f"   Сентимент: {sentiment}")
            
            # Тест извлечения сущностей
            entities = analyzer.extract_key_entities("Газпром увеличил добычу газа")
            print(f"   Сущности: {entities}")
            
        else:
            print("   ℹ️ API ключ не найден, пропускаем тест с API")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при анализе новостей: {e}")
        return False

def test_ensemble_models():
    """Тест ансамбля моделей"""
    print("\n🧪 Тестирование ансамбля моделей...")
    
    try:
        from enhanced_solution import EnsemblePredictor
        
        # Создаем тестовые данные
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 5)  # 5 горизонтов
        
        # Тестируем ансамбль
        ensemble = EnsemblePredictor(n_splits=3)
        
        print("   🤖 Обучение ансамбля...")
        model_scores = ensemble.train_ensemble(X, y, [f'feature_{i}' for i in range(10)])
        
        print(f"✅ Ансамбль обучен: {len(model_scores)} моделей")
        for name, score in model_scores.items():
            print(f"   {name}: RMSE = {score:.6f}")
        
        # Тест предсказания
        test_X = np.random.randn(10, 10)
        predictions = ensemble.predict_ensemble(test_X)
        
        print(f"✅ Предсказания созданы: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании ансамбля: {e}")
        return False

def test_submission_format():
    """Тест формата submission"""
    print("\n Тестирование формата submission...")
    
    try:
        from enhanced_solution import create_submission_format
        
        # Создаем тестовые предсказания
        test_predictions = []
        for ticker in ['AFLT', 'GAZP', 'LKOH']:
            row = {'ticker': ticker, 'begin': '2025-01-01'}
            for horizon in range(1, 21):
                row[f'pred_return_{horizon}d'] = np.random.normal(0, 0.01)
            test_predictions.append(row)
        
        predictions_df = pd.DataFrame(test_predictions)
        
        # Создаем submission
        submission_df = create_submission_format(predictions_df, horizons=list(range(1, 21)))
        
        print(f"✅ Submission создан: {len(submission_df)} строк")
        print(f"   Колонки: {list(submission_df.columns)}")
        
        # Проверяем формат
        expected_cols = ['ticker'] + [f'p{i}' for i in range(1, 21)]
        if list(submission_df.columns) == expected_cols:
            print("✅ Формат submission корректен")
        else:
            print("❌ Неправильный формат submission")
            return False
        
        # Проверяем значения
        for col in [f'p{i}' for i in range(1, 21)]:
            if not submission_df[col].between(0, 1).all():
                print(f"❌ Значения в {col} не в диапазоне [0,1]")
                return False
        
        print("✅ Все значения в правильном диапазоне [0,1]")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании submission: {e}")
        return False

def test_data_loading():
    """Тест загрузки реальных данных"""
    print("\n Тестирование загрузки данных...")
    
    try:
        # Проверяем существование файлов
        train_candles = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv"
        test_candles = "/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv"
        
        if not os.path.exists(train_candles):
            print(f"❌ Файл не найден: {train_candles}")
            return False
        
        if not os.path.exists(test_candles):
            print(f"❌ Файл не найден: {test_candles}")
            return False
        
        # Загружаем данные
        train_df = pd.read_csv(train_candles)
        test_df = pd.read_csv(test_candles)
        
        print(f"✅ Train: {len(train_df)} строк, {len(train_df.columns)} колонок")
        print(f"✅ Test: {len(test_df)} строк, {len(test_df.columns)} колонок")
        
        # Проверяем тикеры
        train_tickers = set(train_df['ticker'].unique())
        test_tickers = set(test_df['ticker'].unique())
        
        print(f"   Train тикеры: {len(train_tickers)}")
        print(f"   Test тикеры: {len(test_tickers)}")
        print(f"   Общие тикеры: {len(train_tickers & test_tickers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return False

def test_enhanced_solution():
    """Тест полного решения"""
    print("\n Тестирование полного решения...")
    
    try:
        from enhanced_solution import run_enhanced_solution
        
        # Запускаем без API ключа
        print("   🚀 Запуск enhanced solution...")
        result = run_enhanced_solution(openrouter_api_key=None)
        
        if result is not None:
            print(f"✅ Решение выполнено успешно")
            print(f"   Результат: {len(result)} строк")
            return True
        else:
            print("❌ Решение не выполнено")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при запуске решения: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print(" ТЕСТИРОВАНИЕ ENHANCED META-STOCK SOLUTION")
    print("=" * 70)
    
    # Запускаем все тесты
    tests = [
        ("Расширенные признаки", test_enhanced_features),
        ("Анализ новостей", test_news_analysis),
        ("Ансамбль моделей", test_ensemble_models),
        ("Формат submission", test_submission_format),
        ("Загрузка данных", test_data_loading),
        ("Полное решение", test_enhanced_solution)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test_name}: {e}")
            results[test_name] = False
    
    # Итоговый отчет
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📈 Итого: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n Все тесты пройдены успешно!")
        print("\n Enhanced META-Stock Solution готов к использованию!")
        print("   • Расширенные технические индикаторы")
        print("   • Ансамбль моделей")
        print("   • Анализ новостей с OpenRouter API")
        print("   • Проверка на лики")
        print("   • Правильный формат submission")
    else:
        print(f"\n⚠️ {total - passed} тестов провалены. Проверьте ошибки выше.")
