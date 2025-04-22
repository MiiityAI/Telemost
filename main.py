#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
import importlib.util
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='main.log', filemode='a')
logger = logging.getLogger(__name__)

def import_from_file(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import modules
script_dir = Path(__file__).parent
try:
    obsidian_processor = import_from_file("obsidian_vault_processor", script_dir / "obsidian_vault_processor.py")
    smart_search = import_from_file("smart_search", script_dir / "smart_search.py")
    insight_finder = import_from_file("insight_finder", script_dir / "insight_finder.py")
    
    # Импортируем API модули с разными именами
    openai_api_gpt_path = script_dir / "openai-api_gpt.py"
    openai_api_v2_path = script_dir / "openai-api_v2.py"
    
    # Проверяем наличие файлов
    if openai_api_gpt_path.exists():
        openai_api_gpt = import_from_file("openai_api_gpt", openai_api_gpt_path)
    else:
        logger.warning("openai-api_gpt.py не найден")
        
    if openai_api_v2_path.exists():
        openai_api_v2 = import_from_file("openai_api_v2", openai_api_v2_path)
    else:
        logger.warning("openai-api_v2.py не найден")

except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    print(f"Error importing required modules: {e}")
    print("Make sure all component scripts are in the same directory.")
    sys.exit(1)

def setup_pipeline(vault_path, config_path, skip_indexing=False, use_gpu=False, api_provider="lm-studio"):
    """Set up the processing pipeline."""
    config = None
    try:
        config = obsidian_processor.load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        print(f"Loaded configuration: cosinus_index={config.get('cosinus_index', 'N/A')}, top_results={config.get('top_results', 'N/A')}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        print(f"Warning: Could not load config from {config_path}, using defaults")
        config = {"cache_dir": ".cache", "cosinus_index": 0.5, "top_results": 10}
    
    cache_dir = config.get("cache_dir", ".cache")
    cache_path = Path(cache_dir)
    cache_exists = cache_path.exists() and os.path.exists(os.path.join(cache_dir, "vector_cache.pkl"))
    
    if not skip_indexing or not cache_exists:
        print("Indexing vault...")
        processor = obsidian_processor.VaultProcessor(use_gpu=use_gpu or config.get("use_gpu", False), config_path=config_path)
        processor.process_vault(vault_path)
        print("Indexing complete.")
    else:
        print("Using existing cache.")
    
    searcher = smart_search.SmartSearcher(config_path, vault_path)
    if 'cosinus_index' in config:
        searcher.cosinus_index = config['cosinus_index']
    if 'top_results' in config:
        searcher.top_results = config['top_results']
    
    # Создаем API клиент в зависимости от выбора пользователя
    api_client = None
    
    # Выбор API клиента строго по выбору пользователя
    if api_provider == "openai":
        # Используем новую версию API для OpenAI
        if 'openai_api_gpt' in globals():
            try:
                # Используем расширенную версию API с отключенной автопроверкой
                api_client = openai_api_gpt.OpenAI_API(
                    config_path=config_path,
                    auto_check_connection=False
                )
                print("Используется OpenAI API (расширенная версия).")
            except Exception as e:
                logger.error(f"Ошибка при инициализации расширенной версии OpenAI API: {e}")
                print(f"Ошибка при инициализации OpenAI API: {e}")
                sys.exit(1)
        else:
            logger.error("Модуль OpenAI API не найден")
            print("Ошибка: Модуль OpenAI API не найден")
            sys.exit(1)
    else:
        # Используем стандартный LM-Studio API
        if 'openai_api_v2' in globals():
            try:
                api_client = openai_api_v2.OpenAI_API(
                    system_message="You are a helpful assistant specializing in organizing and structuring information."
                )
                print("Используется LM-Studio API.")
                
                # Проверка соединения (только для LM-Studio)
                try:
                    if not api_client.check_connection():
                        print("Warning: Could not connect to LM-Studio API. Some features may not work.")
                except Exception as e:
                    logger.error(f"Ошибка при проверке соединения с LM-Studio: {e}")
                    print(f"Warning: Error checking connection to LM-Studio: {e}")
            except Exception as e:
                logger.error(f"Ошибка при инициализации LM-Studio API: {e}")
                print(f"Ошибка при инициализации LM-Studio API: {e}")
                sys.exit(1)
        else:
            logger.error("Модуль LM-Studio API не найден")
            print("Ошибка: Модуль LM-Studio API не найден")
            sys.exit(1)
    
    return {"config": config, "searcher": searcher, "api_client": api_client, "vault_path": vault_path, "config_path": config_path}

def search_vault(query, components):
    """Search the vault and return only the structured response from LM Studio."""
    searcher = components["searcher"]
    api_client = components["api_client"]
    
    results = searcher.search(query)
    if not results:
        return "No relevant information found."
    
    formatted_results = searcher.format_search_results(query, results)
    parts = formatted_results.split("Total tokens:")
    formatted_text = parts[0].strip()
    token_count = int(parts[1].strip()) if len(parts) > 1 else len(formatted_results) // 4
    
    max_tokens = 20000
    
    if token_count > max_tokens:
        max_chars = max_tokens * 4
        formatted_text = formatted_text[:max_chars]
        warning = f"Warning: Text was truncated to {max_tokens} tokens due to exceeding the limit."
    else:
        warning = ""
    
    try:
        prompt = f"""Summarize the following information about: {query}

{formatted_text}

Structure your response with:
1. SUMMARY (1-2 sentences)
2. KEY POINTS (3-5 bullet points)
3. DETAILS (1-2 paragraphs)
"""
        structured_response = api_client.query(prompt)
        if warning:
            return f"{warning}\n\n{structured_response}"
        return structured_response
    except Exception as e:
        logger.error(f"API error: {e}")
        return f"Error: Could not structure response due to API error: {str(e)}"

def run_insight_finder(components):
    """Run the insight finder with API integration."""
    vault_path = components["vault_path"]
    config_path = components["config_path"]
    api_client = components["api_client"]
    
    generator = insight_finder.InsightGenerator(vault_path=vault_path, config_path=config_path)
    insights = generator.find_insights()
    
    if not insights:
        return "No insights found."
    
    print(f"\nFound {len(insights)} potential connections:")
    for i, (file1, file2, data) in enumerate(insights, 1):
        name1 = Path(file1).stem
        name2 = Path(file2).stem
        score = data.get('score', 0)
        print(f"{i}. {name1} <--> {name2} (similarity: {score:.2f})")
    
    while True:  # Цикл для повторного запроса при неверном вводе
        choice = input("\nSelect insight to create (or 'all' for all, 'q' to quit): ")
        
        if choice.lower() == 'q':
            return "Operation cancelled."
            
        if choice.lower() == 'all':
            created = []
            for file1, file2, data in insights:
                path = generator.create_insight_file(file1, file2, data, api_client)
                if path:
                    created.append(path)
            return f"Created {len(created)} insight files."
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(insights):
                file1, file2, data = insights[idx]
                path = generator.create_insight_file(file1, file2, data, api_client)
                if path:
                    return f"Created insight file: {path}"
                else:
                    return "Error creating insight file."
            # Если индекс вне диапазона, цикл продолжится
        except ValueError:
            # При неверном вводе цикл продолжится
            continue

def main_loop(components):
    """Main interactive loop with [clear] and [save] commands."""
    print("\nCommands:")
    print("  [exit] - Exit the program")
    print("  [merge] - Find and merge related notes")
    print("  [clear] - Clear conversation history")  # Новая команда
    print("  [save] - Save conversation history")    # Новая команда
    print("  Any other input is treated as a search query")
    
    while True:
        try:
            user_input = input("\nEnter command or query: ").strip()
            if user_input.lower() == '[exit]':
                print("Exiting...")
                break
            elif user_input.lower() == '[merge]':
                result = run_insight_finder(components)
                print(result)
            elif user_input.lower() == '[clear]':  # Обработка [clear]
                components["api_client"].clear_history()
                print("Conversation history cleared.")
            elif user_input.lower() == '[save]':   # Обработка [save]
                history = components["api_client"].message_history
                # Используем нужную версию save_dialog в зависимости от API типа
                if components.get("api_type") == "openai":
                    file_path = openai_api_gpt.save_dialog(history)
                else:
                    # Проверяем, есть ли функция save_dialog в модуле v2
                    if hasattr(openai_api_v2, 'save_dialog'):
                        file_path = openai_api_v2.save_dialog(history)
                    else:
                        # Если функции нет в модуле v2, используем из gpt
                        file_path = openai_api_gpt.save_dialog(history)
                print(f"Dialog saved to {file_path}")
            else:
                print("Searching...")
                result = search_vault(user_input, components)
                print("\nResult:")
                print(result)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"An error occurred: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Obsidian Knowledge Manager")
    parser.add_argument("-v", "--vault", required=True, help="Path to Obsidian vault")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("-s", "--skip-indexing", action="store_true", help="Skip vault indexing")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for processing")
    parser.add_argument("-m", "--max-results", type=int, help="Override max results in config")
    parser.add_argument("-t", "--threshold", type=float, help="Override threshold in config")
    parser.add_argument("--api", choices=["lm-studio", "openai"], default="lm-studio", 
                      help="API провайдер (lm-studio или openai)")
    
    args = parser.parse_args()
    
    vault_path = Path(args.vault)
    if not vault_path.exists() or not vault_path.is_dir():
        print(f"Error: Vault directory not found: {vault_path}")
        return 1
    
    try:
        # Выбор API - просто выводим информацию
        api_choice = args.api
        if api_choice == 'openai':
            print("\nВыбран OpenAI API. Инициализация...")
        else:
            print("\nВыбран LM-Studio API. Инициализация...")
        
        # Инициализация компонентов
        components = setup_pipeline(str(vault_path), args.config, args.skip_indexing, args.use_gpu, args.api)
        
        # Сохраняем тип API в компонентах
        components["api_type"] = api_choice
        
        if args.max_results is not None:
            components["searcher"].top_results = args.max_results
            print(f"Overriding max results from command line: {args.max_results}")
        if args.threshold is not None:
            components["searcher"].cosinus_index = args.threshold
            print(f"Overriding threshold from command line: {args.threshold}")
        main_loop(components)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"A fatal error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())