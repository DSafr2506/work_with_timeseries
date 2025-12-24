
import os
import sys
import subprocess
import platform

def check_python_version():
    """Проверка версии Python"""
    print(" Проверка версии Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f" Требуется Python 3.8+, у вас {version.major}.{version.minor}")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Установка зависимостей"""
    print("\n  Установка зависимостей...")
    
    # Основные зависимости
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Основные зависимости установлены")
    except subprocess.CalledProcessError:
        print(" Ошибка установки основных зависимостей")
        return False
    
    # OpenAI клиент
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        print(" OpenAI клиент установлен")
    except subprocess.CalledProcessError:
        print(" OpenAI клиент не установлен (опционально)")
    
    # Дополнительные зависимости
    optional_deps = ["catboost", "xgboost", "lightgbm"]
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f" {dep} установлен")
        except subprocess.CalledProcessError:
            print(f" {dep} не установлен (опционально)")
    
    return True

def check_data_files():
    """Проверка наличия данных"""
    print("\n Проверка данных...")
    
    required_files = [
        "data/raw/participants/candles.csv",
        "data/raw/participants/news.csv",
        "data/raw/participants/candles_2.csv",
        "data/raw/participants/news_2.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f" {file}")
    
    if missing_files:
        print(f" Отсутствуют файлы: {missing_files}")
        return False
    
    return True

def run_tests():
    """Запуск тестов"""
    print("\n Запуск тестов...")
    
    try:
        print("   🔄 Тестирование OpenRouter API...")
        result = subprocess.run([sys.executable, "simple_openrouter_test.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("    OpenRouter API тест пройден")
        else:
            print("    OpenRouter API тест не пройден (возможно, нет API ключа)")
    except Exception as e:
        print(f"    Ошибка теста OpenRouter API: {e}")
    
    # Тест META-Stock
    try:
        print("   🔄 Тестирование META-Stock...")
        result = subprocess.run([sys.executable, "test_meta_stock.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("    META-Stock тест пройден")
        else:
            print("    META-Stock тест не пройден")
    except Exception as e:
        print(f"    Ошибка теста META-Stock: {e}")

def run_solutions():
    """Запуск решений"""
    print("\n Запуск решений...")
    
    solutions = [
        ("Enhanced Solution", "scripts/enhanced_solution.py"),
        ("Baseline Solution", "scripts/baseline_solution.py"),
        ("Advanced Solution", "scripts/advanced_solution.py")
    ]
    
    for name, script in solutions:
        print(f"\n Запуск {name}...")
        try:
            result = subprocess.run([sys.executable, script], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f" {name} выполнен успешно")
            else:
                print(f"    {name} завершился с ошибкой")
                print(f"   Ошибка: {result.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {name} превысил время выполнения")
        except Exception as e:
            print(f"    Ошибка запуска {name}: {e}")

def show_results():
    """Показ результатов"""
    print("\n Результаты..")
    
    result_files = [
        "enhanced_submission.csv",
        "baseline_submission.csv", 
        "advanced_submission.csv"
    ]
    
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f" {file} ({size} байт)")
        else:
            print(f" {file} не найден")

def show_next_steps():
    """Показ следующих шагов"""
    print("\n  Следующие шаги:")
    print("1. Проверьте результаты в CSV файлах")
    print("2. Сравните качество разных решений")
    print("3. Настройте OpenRouter API для лучших результатов")
    print("4. Экспериментируйте с параметрами")
    print("\n Рекомендации:")
    print("- Enhanced Solution: лучшее качество с OpenRouter API")
    print("- Baseline Solution: хорошее качество без API")
    print("- Advanced Solution: быстрое решение")

def main():
    """Главная функция"""
    print_header()
    
    # Проверка Python
    if not check_python_version():
        return
    
    # Установка зависимостей
    if not install_requirements():
        print("❌ Не удалось установить зависимости")
        return
    
    # Проверка данных
    if not check_data_files():
        print("❌ Отсутствуют необходимые файлы данных")
        return
    
    # Запуск тестов
    run_tests()
    
    # Запуск решений
    run_solutions()
    
    # Показ результатов
    show_results()
    
    # Следующие шаги
    show_next_steps()
    
    print("\n Quick Start завершен!")
    print("=" * 70)

if __name__ == "__main__":
    main()
