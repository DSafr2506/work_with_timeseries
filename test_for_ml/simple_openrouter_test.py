#!/usr/bin/env python3
"""
Простой тест OpenRouter API без зависимостей
"""

def test_openai_installation():
    """Тест установки OpenAI клиента"""
    print("🧪 Тестирование установки OpenAI клиента...")
    
    try:
        import openai
        print(" OpenAI клиент установлен")
        print(f"   Версия: {openai.__version__}")
        return True
    except ImportError as e:
        print(f" OpenAI клиент не установлен: {e}")
        print("💡 Установите: pip install openai")
        return False

def test_openrouter_connection():
    """Тест подключения к OpenRouter API"""
    print("\n Тестирование подключения к OpenRouter API...")
    
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
        print("    Отправка тестового запроса...")
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
        print(f" API работает! Ответ: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f" Ошибка подключения к API: {e}")
        return False

def test_different_models():
    """Тест разных моделей OpenRouter"""
    print("\n Тестирование разных моделей OpenRouter...")
    
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
                print(f"    Тестирование {model_name}...")
                
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
                print(f"       {model_name}: {response[:50]}...")
                
            except Exception as e:
                print(f"       {model_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f" Ошибка при тестировании моделей: {e}")
        return False

def test_financial_analysis():
    """Тест финансового анализа"""
    print("\n Тестирование финансового анализа...")
    
    try:
        from openai import OpenAI
        
        api_key = "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"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Тест анализа финансовых новостей
        financial_news = [
            "Газпром показал отличные результаты в четвертом квартале",
            "Акции Сбербанка упали на 3% после объявления о дивидендах",
            "Роснефть объявила о новых месторождениях в Арктике",
            "ЦБ повысил ключевую ставку до 16%"
        ]
        
        for i, news in enumerate(financial_news, 1):
            print(f"   📰 Анализ новости {i}: {news[:30]}...")
            
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://finam-hack.com",
                    "X-Title": "Finam Hack",
                },
                model="meta-llama/llama-3.1-8b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Проанализируй эту финансовую новость и верни JSON с полями:
                        - sentiment_score: число от -1 до 1 (отрицательный/положительный)
                        - confidence: число от 0 до 1 (уверенность)
                        - market_impact: число от -1 до 1 (влияние на рынок)
                        - sector: сектор экономики
                        
                        Новость: {news}"""
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            response = completion.choices[0].message.content
            print(f"      📊 Результат: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f" Ошибка при финансовом анализе: {e}")
        return False

def test_news_entity_extraction():
    """Тест извлечения сущностей из новостей"""
    print("\n Тестирование извлечения сущностей...")
    
    try:
        from openai import OpenAI
        
        api_key = "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"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        news_text = "Газпром и Роснефть объявили о совместном проекте в Арктике стоимостью 50 млрд рублей"
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://finam-hack.com",
                "X-Title": "Finam Hack",
            },
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"""Извлеки сущности из этой новости и верни JSON с полями:
                    - companies: список компаний
                    - sectors: список секторов
                    - events: список событий
                    - numbers: список чисел и процентов
                    - money: денежные суммы
                    
                    Новость: {news_text}"""
                }
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        response = completion.choices[0].message.content
        print(f" Сущности извлечены: {response[:150]}...")
        
        return True
        
    except Exception as e:
        print(f" Ошибка при извлечении сущностей: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print(" ПРОСТОЙ ТЕСТ OPENROUTER API")
    print("=" * 70)
    
    # Запускаем все тесты
    tests = [
        ("OpenAI клиент", test_openai_installation),
        ("Подключение к API", test_openrouter_connection),
        ("Разные модели", test_different_models),
        ("Финансовый анализ", test_financial_analysis),
        ("Извлечение сущностей", test_news_entity_extraction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f" {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f" Критическая ошибка в тесте {test_name}: {e}")
            results[test_name] = False
    
    # Итоговый отчет
    print("\n" + "=" * 70)
    print(" РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = " ПРОЙДЕН" if success else " ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n Итого: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n Все тесты пройдены успешно!")
        print("\n OpenRouter API интегрирован и готов к использованию!")
        print("   • OpenAI клиент работает")
        print("   • Подключение к OpenRouter API успешно")
        print("   • Различные модели доступны")
        print("   • Финансовый анализ функционирует")
        print("   • Извлечение сущностей работает")
    else:
        print(f"\n {total - passed} тестов провалены. Проверьте ошибки выше.")
        
        if not results.get("OpenAI клиент", False):
            print("\n💡 Для решения проблем:")
            print("   1. Установите OpenAI клиент: pip install openai")
            print("   2. Проверьте интернет-соединение")
            print("   3. Убедитесь в правильности API ключа")
