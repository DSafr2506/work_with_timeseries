#!/usr/bin/env python3
"""
Тест интеграции OpenRouter API
"""

import sys
import os
sys.path.append('/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/scripts')

def test_openai_client():
    """Тест установки OpenAI клиента"""
    print("🧪 Тестирование OpenAI клиента...")
    
    try:
        from openai import OpenAI
        print("✅ OpenAI клиент установлен")
        return True
    except ImportError as e:
        print(f"❌ OpenAI клиент не установлен: {e}")
        print("💡 Установите: pip install openai")
        return False

def test_openrouter_connection():
    """Тест подключения к OpenRouter API"""
    print("\n🧪 Тестирование подключения к OpenRouter API...")
    
    try:
        from openai import OpenAI
        
        # Ваш API ключ
        api_key = "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"
        
        # Инициализация клиента
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Простой тест запроса
        print("   🔄 Отправка тестового запроса...")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://finam-hack.com",
                "X-Title": "Finam Hack",
            },
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Привет! Это тест подключения к OpenRouter API."
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        response = completion.choices[0].message.content
        print(f"✅ API работает! Ответ: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения к API: {e}")
        return False

def test_news_analyzer():
    """Тест анализатора новостей"""
    print("\n🧪 Тестирование анализатора новостей...")
    
    try:
        from enhanced_solution import OpenRouterNewsAnalyzer
        
        # Инициализация анализатора
        api_key = "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"
        analyzer = OpenRouterNewsAnalyzer(
            api_key=api_key,
            model="meta-llama/llama-3.1-8b-instruct"
        )
        
        # Тест анализа сентимента
        print("   🔍 Тестирование анализа сентимента...")
        test_text = "Газпром показал отличные результаты в четвертом квартале"
        sentiment = analyzer.analyze_sentiment(test_text)
        
        print(f"✅ Сентимент проанализирован: {sentiment}")
        
        # Тест извлечения сущностей
        print("   🔍 Тестирование извлечения сущностей...")
        entities = analyzer.extract_key_entities(test_text)
        
        print(f"✅ Сущности извлечены: {entities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в анализаторе новостей: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_solution_with_api():
    """Тест полного решения с API"""
    print("\n🧪 Тестирование полного решения с OpenRouter API...")
    
    try:
        from enhanced_solution import run_enhanced_solution
        
        # Запуск с API
        api_key = "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"
        model = "meta-llama/llama-3.1-8b-instruct"
        
        print("   🚀 Запуск enhanced solution с OpenRouter API...")
        result = run_enhanced_solution(api_key, model)
        
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

def test_different_models():
    """Тест разных моделей OpenRouter"""
    print("\n🧪 Тестирование разных моделей OpenRouter...")
    
    models_to_test = [
        ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B"),
        ("anthropic/claude-3-haiku", "Claude 3 Haiku"),
        ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo"),
        ("google/gemini-pro", "Gemini Pro")
    ]
    
    api_key = "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"
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        for model_id, model_name in models_to_test:
            try:
                print(f"   🔄 Тестирование {model_name}...")
                
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://finam-hack.com",
                        "X-Title": "Finam Hack",
                    },
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": "Проанализируй сентимент: 'Акции компании выросли на 5%'"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
                
                response = completion.choices[0].message.content
                print(f"      ✅ {model_name}: {response[:50]}...")
                
            except Exception as e:
                print(f"      ❌ {model_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании моделей: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ OPENROUTER API")
    print("=" * 70)
    
    # Запускаем все тесты
    tests = [
        ("OpenAI клиент", test_openai_client),
        ("Подключение к API", test_openrouter_connection),
        ("Анализатор новостей", test_news_analyzer),
        ("Разные модели", test_different_models),
        ("Полное решение", test_enhanced_solution_with_api)
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
        print("\n🎉 Все тесты пройдены успешно!")
        print("\n💡 OpenRouter API интегрирован и готов к использованию!")
        print("   • OpenAI клиент работает")
        print("   • Подключение к OpenRouter API успешно")
        print("   • Анализатор новостей функционирует")
        print("   • Различные модели доступны")
        print("   • Полное решение работает с API")
    else:
        print(f"\n⚠️ {total - passed} тестов провалены. Проверьте ошибки выше.")
        
        if not results.get("OpenAI клиент", False):
            print("\n💡 Для решения проблем:")
            print("   1. Установите OpenAI клиент: pip install openai")
            print("   2. Проверьте интернет-соединение")
            print("   3. Убедитесь в правильности API ключа")
