#!/usr/bin/env python3
import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_lm_studio_connection(url="http://localhost:1234/v1", retries=3, delay=2):
    """Проверка подключения к LM-Studio с retry-механизмом."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{url}/models", timeout=2)
            if response.status_code == 200:
                print(f"[OK] Подключение к LM-Studio успешно на попытке {attempt}")
                return True
            else:
                print(f"[ERROR] Не удалось подключиться к LM-Studio: {response.status_code} (попытка {attempt}/{retries})")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Ошибка подключения к LM-Studio: {e} (попытка {attempt}/{retries})")
        
        if attempt < retries:
            print(f"Повторная попытка через {delay} секунд...")
            time.sleep(delay)
    
    print(f"[ERROR] Все попытки ({retries}) исчерпаны")
    return False

def get_valid_vault_path():
    """Запрашивает у пользователя путь к Vault и проверяет его существование."""
    # Пытаемся получить путь из конфигурации
    default_path = ""
    config_path = Path(os.path.dirname(os.path.abspath(__file__))) / "config.yaml"
    
    try:
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                default_path = config.get('vault_path', '')
                if default_path:
                    print(f"[INFO] Найден путь к хранилищу в конфигурации: {default_path}")
    except Exception as e:
        print(f"[INFO] Не удалось прочитать путь из config.yaml: {e}")
    
    while True:
        if default_path:
            vault_path = input(f"1. Где находится Obsidian Vault? (укажите полный путь или нажмите Enter для {default_path}): ").strip()
            # Если пользователь просто нажал Enter, используем значение по умолчанию
            if not vault_path:
                vault_path = default_path
                print(f"[INFO] Используется путь из конфигурации: {vault_path}")
        else:
            vault_path = input("1. Где находится Obsidian Vault? (укажите полный путь): ").strip()
        
        if os.path.exists(vault_path) and os.path.isdir(vault_path):
            return vault_path
        else:
            print(f"Ошибка: Директория '{vault_path}' не существует или не является директорией. Попробуйте снова.")

def get_valid_api_choice():
    """Запрашивает у пользователя выбор API."""
    while True:
        print("\n2. Какой API вы хотите использовать?")
        print("   a. LM-Studio (локальная модель)")
        print("   b. OpenAI (облачная модель)")
        api_choice = input("   Ваш выбор (a/b): ").strip().lower()
        
        if api_choice == 'a':
            return 'lm-studio'
        elif api_choice == 'b':
            return 'openai'
        else:
            print("   Пожалуйста, выберите 'a' или 'b'.")

def try_connect_to_api(api_choice):
    """Пытается подключиться к выбранному API с возможностью повторных попыток."""
    while True:
        if api_choice == 'a' or api_choice == 'lm-studio':  # LM-Studio - поддерживаем оба значения
            if check_lm_studio_connection():
                return True
            
            retry = input("\nНе удалось подключиться к LM-Studio. Хотите попробовать снова? (y/n): ").strip().lower()
            if retry != 'y':
                print("\n2. Какой API вы хотите использовать?")
                print("   a. LM-Studio (локальная модель)")
                api_choice = input("   Ваш выбор (a): ").strip().lower()
                if not api_choice:
                    api_choice = 'a'
                continue
        else:
            # Другие API в будущем
            pass

def get_yes_no_input(prompt, default='y'):
    """Запрашивает у пользователя ответ да/нет с проверкой ввода."""
    while True:
        response = input(prompt).strip().lower()
        if not response:
            return default
        if response in ['y', 'n']:
            return response
        print("Пожалуйста, введите 'y' для да или 'n' для нет.")

def check_openai_connection(config_path="config.yaml"):
    """Проверка подключения к OpenAI API."""
    try:
        import yaml
        
        # Проверка существования файла конфигурации
        if not os.path.exists(config_path):
            print(f"[ERROR] Файл конфигурации не найден: {os.path.abspath(config_path)}")
            return False
            
        print(f"[INFO] Чтение конфигурации из файла: {os.path.abspath(config_path)}")
        
        # Пытаемся загрузить API ключ из конфигурации
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Поиск по нескольким возможным именам ключа
            api_key = config.get('api_key', config.get('openai_api_key', ''))
            
            if not api_key:
                print("[ERROR] API ключ не найден в конфигурации. Проверьте наличие параметра 'api_key' в файле.")
                return False
            
            # Логирование для отладки
            key_preview = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 12 else "***"
            print(f"[INFO] Найден API ключ: {key_preview}")
            
            # Пробуем выполнить тестовый запрос
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            print("[INFO] Отправка тестового запроса к OpenAI API...")
            response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
            
            if response.status_code == 200:
                print("[OK] Подключение к OpenAI API успешно")
                return True
            else:
                print(f"[ERROR] Ошибка подключения к OpenAI API: {response.status_code}")
                print(f"[ERROR] Ответ сервера: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке подключения к OpenAI API: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Интерактивный запуск процесса анализа Obsidian Vault."""
    print("\n--- Obsidian Knowledge Manager Setup ---\n")
    
    # Запрос пути к хранилищу
    vault_path = get_valid_vault_path()
    
    # Запрос выбора API
    api_choice = get_valid_api_choice()
    
    # Определяем путь к конфигурационному файлу
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = script_dir / "config.yaml"
    
    print(f"[INFO] Используем конфигурацию: {os.path.abspath(config_path)}")
    
    # Проверка подключения к API
    if api_choice == 'openai':
        print("\nПроверка подключения к OpenAI API...")
        if not check_openai_connection(config_path):
            print("⚠️ Не удалось подключиться к OpenAI API. Проверьте api_key в config.yaml.")
            retry = input("Попробовать использовать LM-Studio вместо OpenAI? (y/n): ").strip().lower()
            if retry == 'y':
                api_choice = 'lm-studio'
                print("Выбран LM-Studio API.")
                # Проверяем подключение к LM-Studio с возможностью пропуска
                print("\nПроверка подключения к LM-Studio...")
                skip_check = input("Пропустить проверку подключения к LM-Studio? (y/n, по умолчанию: n): ").strip().lower()
                if skip_check != 'y':
                    try_connect_to_api(api_choice)
            else:
                print("Продолжаем с OpenAI API, но могут возникнуть проблемы.")
    else:
        # Проверка подключения LM-Studio с возможностью пропуска
        print("\nПроверка подключения к LM-Studio...")
        skip_check = input("Пропустить проверку подключения к LM-Studio? (y/n, по умолчанию: n): ").strip().lower()
        if skip_check != 'y':
            try_connect_to_api(api_choice)
    
    # Запрос о необходимости повторной индексации
    reindex = get_yes_no_input("\n3. Нужно ли проводить индексацию заново? (y/n, по умолчанию y): ", default='y')
    skip_indexing_flag = "--skip-indexing" if reindex == 'n' else ""
    
    # Расширенный запрос о GPU
    print("\n4. Использование GPU может значительно ускорить обработку данных (особенно векторизацию),")
    print("   но требует наличия совместимой видеокарты и установленных драйверов CUDA/ROCm.")
    use_gpu = get_yes_no_input("   Использовать GPU для обработки? (y/n, по умолчанию n): ", default='n')
    
    # Если пользователь выбрал GPU, дополнительно спрашиваем о VRAM
    if use_gpu == 'y':
        print("\n   Информация: Для корректной работы с GPU требуется минимум 4GB VRAM.")
        print("   При недостаточном объеме VRAM возможны ошибки Out of Memory.")
        confirm_gpu = get_yes_no_input("   Вы уверены, что хотите использовать GPU? (y/n): ", default='y')
        use_gpu_flag = "--use-gpu" if confirm_gpu == 'y' else ""
    else:
        use_gpu_flag = ""
    
    # Формирование команды для запуска
    while True:
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        main_script = script_dir / "main.py"
        
        if not main_script.exists():
            print(f"\nОшибка: Скрипт main.py не найден в директории {script_dir}")
            response = input("Хотите указать другой путь к скрипту main.py? (y/n): ").strip().lower()
            
            if response == 'y':
                custom_path = input("Введите полный путь к скрипту main.py: ").strip()
                main_script = Path(custom_path)
                if not main_script.exists():
                    print(f"Ошибка: Скрипт не найден по указанному пути: {custom_path}")
                    continue
            else:
                print("Запуск невозможен без main.py. Проверьте наличие файла и попробуйте снова.")
                continue
        
        # Команда готова, выходим из цикла
        break
    
    while True:
        # Формирование команды для запуска с учетом выбранного API
        command = [sys.executable, str(main_script), "-v", vault_path, "--api", api_choice]
        
        if skip_indexing_flag:
            command.append(skip_indexing_flag)
        
        if use_gpu_flag:
            command.append(use_gpu_flag)
        
        # Вывод команды для отладки
        print("\nЗапуск команды:")
        print(" ".join(command))
        
        # Запуск скрипта
        try:
            print("\nЗапускаем процесс...")
            subprocess.run(command, check=True)
            # Успешно выполнено, выходим из цикла
            break
        except subprocess.CalledProcessError as e:
            print(f"\nОшибка при выполнении процесса: {e}")
            retry = input("Хотите попробовать запустить скрипт снова? (y/n): ").strip().lower()
            if retry != 'y':
                print("Программа завершена пользователем.")
                return
            # Спрашиваем, хочет ли пользователь изменить параметры
            change_params = input("Хотите изменить параметры запуска? (y/n): ").strip().lower()
            if change_params == 'y':
                # Возвращаемся к запросу параметров
                reindex = get_yes_no_input("\n3. Нужно ли проводить индексацию заново? (y/n, по умолчанию y): ", default='y')
                skip_indexing_flag = "--skip-indexing" if reindex == 'n' else ""
                
                use_gpu = get_yes_no_input("\n4. Использовать GPU для обработки? (y/n, по умолчанию n): ", default='n')
                use_gpu_flag = "--use-gpu" if use_gpu == 'y' else ""
                
                # Также предлагаем выбрать другой Vault
                change_vault = input("Хотите выбрать другой Vault? (y/n): ").strip().lower()
                if change_vault == 'y':
                    vault_path = get_valid_vault_path()
                
                # Также предлагаем выбрать другой API
                change_api = input("Хотите выбрать другой API? (y/n): ").strip().lower()
                if change_api == 'y':
                    api_choice = get_valid_api_choice()
        
        except KeyboardInterrupt:
            print("\nПроцесс был прерван пользователем.")
            retry = input("Хотите попробовать запустить скрипт снова? (y/n): ").strip().lower()
            if retry != 'y':
                print("Программа завершена пользователем.")
                return

if __name__ == "__main__":
    main()
