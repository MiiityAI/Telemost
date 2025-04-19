from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from llama_cpp import Llama
import uuid
import time
from collections import deque
import logging
import os
from datetime import datetime
import sys
import subprocess
import shlex
import re
import requests
import argparse
import json

def get_api_keys():
    """Ищет API ключи в системных переменных и файле config.yaml."""
    api_keys = {}
    
    # Проверяем системные переменные
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        api_keys["openai"] = openai_key
    if anthropic_key:
        api_keys["anthropic"] = anthropic_key
    
    # Проверяем файл config.yaml
    config_file = "config.yaml"
    if os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                
            if "OPENAI_API_KEY" in config and "openai" not in api_keys:
                api_keys["openai"] = config["OPENAI_API_KEY"]
            if "ANTHROPIC_API_KEY" in config and "anthropic" not in api_keys:
                api_keys["anthropic"] = config["ANTHROPIC_API_KEY"]
            
            if "openai_model" in config:
                api_keys["openai_model"] = config["openai_model"]
        except Exception as e:
            print(f"Ошибка при чтении config.yaml: {str(e)}")
    
    return api_keys

def choose_model():
    """Интерактивный выбор модели при запуске."""
    print("\n" + "=" * 60)
    print("Выберите модель для работы:")
    print("1. gpt (OpenAI GPT-4o)")
    print("2. claude (Anthropic Claude)")
    print("3. deepseek (DeepSeek-R1 локальная модель)")
    print("=" * 60)
    
    choice = input("Введите номер (1-3): ").strip()
    
    # Получаем API ключи
    api_keys = get_api_keys()
    
    if choice == "1":
        if "openai" not in api_keys:
            print("Ключ OpenAI API не найден. Используем значение по умолчанию.")
            api_keys["openai"] = "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"
        return "openai", api_keys
    elif choice == "2":
        if "anthropic" not in api_keys:
            print("Ключ Anthropic API не найден. Используем значение по умолчанию.")
            api_keys["anthropic"] = "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"
        return "anthropic", api_keys
    else:  # По умолчанию выбираем локальную модель
        return "local", api_keys

class CommandIntentDetector:
    """
    Класс для определения намерений пользователя по выполнению команд и автоматической обработки.
    """
    def __init__(self):
        self.file_commands = {
            "показать файлы": ["ls", "dir"],
            "список файлов": ["ls", "dir"],
            "файлы в папке": ["ls", "dir"],
            "содержимое директории": ["ls", "dir"],
            "содержимое папки": ["ls", "dir"],
            "показать директорию": ["ls", "dir"],
        }
        
        self.system_commands = {
            "текущий путь": ["pwd"],
            "текущая директория": ["pwd"],
            "текущая папка": ["pwd"],
            "путь к файлу": ["pwd"],
            "показать дату": ["date"],
            "текущая дата": ["date"],
            "текущее время": ["date"],
            "показать время": ["date"],
            "кто я": ["whoami"],
            "имя пользователя": ["whoami"],
            "идентификация пользователя": ["whoami"],
            "проверить сеть": ["ping -c 4 google.com"],
            "пинг": ["ping -c 4 google.com"],
        }
        
        # Объединяем команды для более эффективного поиска
        self.all_commands = {**self.file_commands, **self.system_commands}
    
    def detect_command_intent(self, user_text):
        """
        Определяет, содержит ли запрос пользователя намерение выполнить системную команду.
        
        Args:
            user_text (str): Запрос пользователя.
            
        Returns:
            dict: Информация о намерении выполнить команду.
                {
                    "has_command_intent": bool,
                    "command": str,
                    "command_type": str,
                    "confidence": float,
                    "original_text": str
                }
        """
        user_text_lower = user_text.lower()
        result = {
            "has_command_intent": False,
            "command": None,
            "command_type": None,
            "confidence": 0.0,
            "original_text": user_text
        }
        
        # Проверяем наличие ключевых фраз в запросе
        for intent_phrase, commands in self.all_commands.items():
            if intent_phrase in user_text_lower:
                result["has_command_intent"] = True
                result["command"] = commands[0]  # Берем первую команду из списка
                
                # Определяем тип команды
                if intent_phrase in self.file_commands:
                    result["command_type"] = "file"
                else:
                    result["command_type"] = "system"
                
                # Простая оценка уверенности - чем короче запрос, тем выше уверенность
                # (меньше шансов ложного срабатывания)
                words_count = len(user_text_lower.split())
                if words_count <= 3:
                    result["confidence"] = 0.9
                elif words_count <= 6:
                    result["confidence"] = 0.8
                elif words_count <= 10:
                    result["confidence"] = 0.7
                else:
                    result["confidence"] = 0.6
                
                # Если нашли совпадение, прекращаем поиск
                break
                
        return result
    
    def should_execute_automatically(self, intent_info, auto_execute_threshold=0.7):
        """
        Определяет, следует ли автоматически выполнить команду на основе уверенности.
        
        Args:
            intent_info (dict): Информация о намерении.
            auto_execute_threshold (float): Порог уверенности для автоматического выполнения.
            
        Returns:
            bool: True, если команду следует выполнить автоматически.
        """
        return (
            intent_info["has_command_intent"] and 
            intent_info["confidence"] >= auto_execute_threshold
        )

class ApiModelHandler:
    """
    Класс для обработки запросов к внешним API моделям (OpenAI и Anthropic).
    """
    def __init__(self, api_keys=None):
        """
        Инициализирует обработчик API с ключами.
        
        Args:
            api_keys (dict): Словарь с API ключами {'openai': 'key', 'anthropic': 'key'}
        """
        self.api_keys = api_keys or {}
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        self.anthropic_api_url = "https://api.anthropic.com/v1/messages"
        
    def format_openai_request(self, system_text, user_text, history, temperature, max_tokens):
        """
        Форматирует запрос для OpenAI API.
        """
        messages = [{"role": "system", "content": system_text}]
        
        # Добавляем историю
        for user_msg, model_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": model_msg})
        
        # Добавляем текущий запрос
        messages.append({"role": "user", "content": user_text})
        
        # Используем модель из конфигурации, если она указана
        model = "gpt-4o"
        if "openai_model" in self.api_keys:
            model = self.api_keys["openai_model"]
        
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    def format_anthropic_request(self, system_text, user_text, history, temperature, max_tokens):
        """
        Форматирует запрос для Anthropic API.
        """
        messages = []
        
        # Добавляем историю
        for user_msg, model_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": model_msg})
        
        # Добавляем текущий запрос пользователя
        messages.append({"role": "user", "content": user_text})
        
        return {
            "model": "claude-3-7-sonnet-20250219",
            "messages": messages,
            "system": system_text,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    def call_openai_api(self, system_text, user_text, history, temperature=0.7, max_tokens=1024):
        """
        Вызывает API OpenAI для получения ответа.
        """
        if 'openai' not in self.api_keys:
            return {
                "status": "error",
                "error_message": "OpenAI API key not provided."
            }
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json"
        }
        
        data = self.format_openai_request(
            system_text, user_text, history, temperature, max_tokens
        )
        
        try:
            response = requests.post(
                self.openai_api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "status": "success",
                    "response": response_data["choices"][0]["message"]["content"],
                    "model": data["model"],
                    "usage": response_data.get("usage", {})
                }
            else:
                return {
                    "status": "error",
                    "error_message": f"API Error: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Request failed: {str(e)}"
            }
    
    def call_anthropic_api(self, system_text, user_text, history, temperature=0.7, max_tokens=1024):
        """
        Вызывает API Anthropic для получения ответа.
        """
        if 'anthropic' not in self.api_keys:
            return {
                "status": "error",
                "error_message": "Anthropic API key not provided."
            }
        
        headers = {
            "x-api-key": self.api_keys['anthropic'],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = self.format_anthropic_request(
            system_text, user_text, history, temperature, max_tokens
        )
        
        try:
            response = requests.post(
                self.anthropic_api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "status": "success",
                    "response": response_data["content"][0]["text"],
                    "model": "claude-3-7-sonnet-20250219",
                    "usage": {
                        "prompt_tokens": response_data.get("usage", {}).get("input_tokens", 0),
                        "completion_tokens": response_data.get("usage", {}).get("output_tokens", 0)
                    }
                }
            else:
                return {
                    "status": "error",
                    "error_message": f"API Error: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Request failed: {str(e)}"
            }

class MCPCompliantModel:
    def __init__(
        self,
        model_type: str = "local",  # "local", "openai" или "anthropic"
        model_path: str = "/home/dima/.cache/gpt4all/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        api_keys: dict = None,
        max_context_length: int = 32768,
        temperature: float = 0.7,
        input_tokens_limit: int = 2048,
        output_tokens_limit: int = 1024,
        top_p: float = 1.0,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_sequences: list = ["<|im_end|>", "<|im_start|>"],
        auto_execute_commands: bool = True,
        auto_execute_threshold: float = 0.7,
        **kwargs
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop_sequences = stop_sequences
        
        # Настройки автоматического выполнения команд
        self.auto_execute_commands = auto_execute_commands
        self.auto_execute_threshold = auto_execute_threshold
        self.command_detector = CommandIntentDetector()

        # Хранилище диалога
        self.dialog_history = deque(maxlen=10)

        # Инициализация модели в зависимости от типа
        if model_type == "local":
            temp_out = StringIO()
            temp_err = StringIO()
            with redirect_stdout(temp_out), redirect_stderr(temp_err):
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.max_context_length,
                    n_gpu_layers=24,
                    verbose=False,
                    **kwargs
                )
            self.api_handler = None
        else:
            self.llm = None
            self.api_handler = ApiModelHandler(api_keys)
        
        # Настройка логирования
        self.setup_logging()
    
    def setup_logging(self):
        """
        Настраивает систему логирования для MCPCompliantModel.
        """
        # Создаем директорию для логов, если её нет
        log_dir = os.path.join(os.path.expanduser("~"), ".mcp_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Имя файла лога с датой
        log_file = os.path.join(log_dir, f"mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Настройка форматирования
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Настройка файлового хендлера
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Настройка консольного хендлера
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Только предупреждения и ошибки в консоль
        console_handler.setFormatter(formatter)
        
        # Получаем логгер
        self.logger = logging.getLogger('mcp_agent')
        self.logger.setLevel(logging.INFO)
        
        # Добавляем хендлеры
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("MCP Agent initialized")
        self.logger.info(f"Model type: {self.model_type}")
        if self.model_type == "local":
            self.logger.info(f"Model path: {self.model_path}")
        self.logger.info(f"Max context length: {self.max_context_length}")

    def log_conversation(self, user_text, response):
        """
        Логирует разговор с пользователем.
        
        Args:
            user_text (str): Запрос пользователя.
            response (dict): Ответ модели.
        """
        if hasattr(self, 'logger'):
            self.logger.info(f"User: {user_text}")
            if "choices" in response and response["choices"]:
                model_response = response["choices"][0]["text"]
                self.logger.info(f"AI: {model_response}")
            else:
                self.logger.warning("No response from model")

    def log_code_execution(self, code, result):
        """
        Логирует выполнение Python кода.
        
        Args:
            code (str): Исполняемый код.
            result (dict): Результат выполнения.
        """
        if hasattr(self, 'logger'):
            self.logger.info(f"Executing Python code: {code[:100]}...")
            if result["status"] == "success":
                self.logger.info("Code execution successful")
                self.logger.debug(f"Output: {result['output']}")
            else:
                self.logger.error(f"Code execution failed: {result.get('error_message', '')}")

    def log_command_execution(self, command, result):
        """
        Логирует выполнение системной команды.
        
        Args:
            command (str): Исполняемая команда.
            result (dict): Результат выполнения.
        """
        if hasattr(self, 'logger'):
            self.logger.info(f"Executing system command: {command}")
            if result["status"] == "success":
                self.logger.info("Command execution successful")
                self.logger.debug(f"Output: {result['output']}")
            else:
                self.logger.error(f"Command execution failed: {result.get('error_message', result.get('error', ''))}")

    def format_prompt(self, system_text, user_text):
        """
        Улучшенное форматирование промпта с расширенным контекстом.
        """
        # Базовый системный промпт
        full_prompt = f"<|im_start|>system\n{system_text}\n\nYou have access to these tools:\n"
        
        # Добавляем инструкции по доступным инструментам
        full_prompt += "1. Python Interpreter: You can run Python code. Wrap code in ```python and ```\n"
        full_prompt += "2. System Commands: You can run system commands. Wrap commands in ```cmd and ```\n"
        full_prompt += "3. File Operations: You can read files using file operations in Python\n\n"
        
        # Добавляем правила безопасности
        full_prompt += "IMPORTANT RULES:\n"
        full_prompt += "- Always provide code when appropriate to solve user's problems\n"
        full_prompt += "- Never execute dangerous commands or code that could harm the system\n"
        full_prompt += "- When providing commands, explain what they do before suggesting them\n"
        full_prompt += "<|im_end|>\n"
        
        # Добавление истории диалога с контекстом выполнения команд
        for pair in self.dialog_history:
            # Проверка корректности формата записи в истории
            if isinstance(pair, tuple) and len(pair) == 2:
                user_msg, model_msg = pair
                full_prompt += (
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{model_msg}<|im_end|>\n"
                )
            else:
                # Логируем ошибку, но продолжаем работу
                if hasattr(self, 'logger'):
                    self.logger.error(f"Invalid dialog history entry: {pair}")
        
        # Добавление текущего запроса пользователя
        full_prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        
        return full_prompt

    def handle_request(self, request_str):
        try:
            request = json.loads(request_str)
            req_id = request.get("id", str(uuid.uuid4()))
            created = int(time.time())

            if "prompt" not in request:
                return self.error_response("Missing 'prompt' parameter", req_id)
            user_text = request["prompt"]

            # Обработка фидбэка
            feedback = request.get("feedback", "positive")
            if feedback == "negative":
                self.presence_penalty += 0.1

            system_text = request.get("system", "You are a helpful assistant.")
            temperature = float(request.get("temperature", self.temperature))
            max_tokens = int(request.get("max_tokens", self.output_tokens_limit))
            top_p = float(request.get("top_p", self.top_p))
            top_k = int(request.get("top_k", self.top_k))
            stop = request.get("stop", self.stop_sequences)
            presence_penalty = float(request.get("presence_penalty", self.presence_penalty))
            frequency_penalty = float(request.get("frequency_penalty", self.frequency_penalty))
            input_tokens_limit = int(request.get("input_tokens", self.input_tokens_limit))

            # Проверки параметров
            if not 0.0 <= temperature <= 2.0:
                return self.error_response("Temperature must be between 0.0 and 2.0", req_id)
            if not 0 <= top_p <= 1.0:
                return self.error_response("Top_p must be between 0.0 and 1.0", req_id)
                
            # Проверка на намерение выполнить команду
            if self.auto_execute_commands:
                intent_info = self.command_detector.detect_command_intent(user_text)
                should_execute = self.command_detector.should_execute_automatically(
                    intent_info, self.auto_execute_threshold
                )
                
                if should_execute:
                    # Выполняем команду и получаем результат
                    command_result = self.execute_system_command(intent_info["command"])
                    
                    # Логируем выполнение команды
                    self.log_command_execution(intent_info["command"], command_result)
                    
                    # Формируем новый запрос с результатом выполнения команды
                    if command_result["status"] == "success":
                        result_text = command_result["output"]
                        new_user_text = (
                            f"{user_text}\n\n"
                            f"Результат выполнения команды '{intent_info['command']}':\n"
                            f"{result_text}"
                        )
                        user_text = new_user_text

            # Генерация ответа в зависимости от типа модели
            if self.model_type == "local":
                # Обработка с использованием локальной модели
                formatted_prompt = self.format_prompt(system_text, user_text)
                try:
                    self.validate_prompt(formatted_prompt)
                except ValueError as e:
                    return self.error_response(f"Invalid prompt format: {str(e)}", req_id)

                input_tokens_used = self.count_tokens(formatted_prompt)

                if input_tokens_used > input_tokens_limit:
                    return self.error_response(
                        f"Input tokens {input_tokens_used} exceed allowed {input_tokens_limit}",
                        req_id
                    )

                if input_tokens_used + max_tokens > self.max_context_length:
                    return self.error_response(
                        f"Context length {input_tokens_used + max_tokens} exceeds maximum {self.max_context_length}",
                        req_id
                    )

                # Генерация ответа
                response = self.llm.create_completion(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty
                )

                # Обработка ответа
                choices = []
                total_completion = 0
                for choice in response["choices"]:
                    text = choice["text"]
                    completion_tokens = self.count_tokens(text)
                    total_completion += completion_tokens
                    choices.append({
                        "text": text,
                        "index": choice["index"],
                        "logprobs": None,
                        "finish_reason": choice["finish_reason"]
                    })

                # Обновление истории диалога (перед возвратом ответа)
                model_response = response["choices"][0]["text"] if response["choices"] else ""
                self.dialog_history.append((user_text, model_response))  # пункт 4
                
                # Формирование финального ответа
                final_response = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": "DeepSeek-R1-Distill-Qwen-7B-GGUF",
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": input_tokens_used,
                        "completion_tokens": total_completion,
                        "total_tokens": input_tokens_used + total_completion
                    }
                }
                
            else:
                # Обработка с использованием API
                if self.model_type == "openai":
                    api_response = self.api_handler.call_openai_api(
                        system_text, user_text, self.dialog_history, temperature, max_tokens
                    )
                elif self.model_type == "anthropic":
                    api_response = self.api_handler.call_anthropic_api(
                        system_text, user_text, self.dialog_history, temperature, max_tokens
                    )
                else:
                    return self.error_response(f"Unsupported model type: {self.model_type}", req_id)
                
                if api_response["status"] == "error":
                    return self.error_response(api_response["error_message"], req_id)
                
                # Получаем ответ от API
                model_response = api_response["response"]
                
                # Обновление истории диалога
                self.dialog_history.append((user_text, model_response))
                
                # Формирование финального ответа
                final_response = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": api_response["model"],
                    "choices": [{
                        "text": model_response,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }],
                    "usage": api_response.get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    })
                }
            
            return final_response
        except Exception as e:
            return self.error_response(f"Error: {str(e)}", req_id)

    def validate_prompt(self, prompt):
        required_tags = [
            "<|im_start|>system",
            "<|im_end|>",
            "<|im_start|>user",
            "<|im_start|>assistant"
        ]
        
        for tag in required_tags:
            if tag not in prompt:
                raise ValueError(f"Отсутствует обязательный тег: {tag}")
        
        return True

    def execute_code(self, code: str):
        """
        Выполняет Python код и возвращает результат выполнения.
        
        Args:
            code (str): Python код для выполнения.
            
        Returns:
            dict: Результат выполнения кода.
        """
        import sys
        from io import StringIO
        
        # Создаем безопасное окружение для выполнения
        safe_globals = {
            "__builtins__": __builtins__,
            "print": print,
            "input": input
        }
        
        # Перенаправляем стандартный вывод
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            # Выполняем код
            exec(code, safe_globals)
            output = redirected_output.getvalue()
            result = {
                "status": "success",
                "output": output
            }
        except Exception as e:
            result = {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        finally:
            # Восстанавливаем стандартный вывод
            sys.stdout = old_stdout
            
        return result

    def execute_system_command(self, command: str):
        """
        Безопасно выполняет системную команду и возвращает результат.
        
        Args:
            command (str): Системная команда для выполнения.
            
        Returns:
            dict: Результат выполнения команды.
        """
        import subprocess
        import shlex
        
        # Список разрешенных команд
        allowed_commands = ['ls', 'dir', 'echo', 'pwd', 'whoami', 'date', 'time', 'ping']
        
        # Получаем первое слово команды для проверки
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return {"status": "error", "error_message": "Empty command"}
        
        base_cmd = cmd_parts[0]
        
        # Проверяем, разрешена ли команда
        if base_cmd not in allowed_commands:
            return {
                "status": "error",
                "error_message": f"Command '{base_cmd}' is not allowed for security reasons."
            }
        
        try:
            # Выполняем команду с таймаутом
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error_message": "Command execution timed out after 10 seconds"
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

    def chat_cli(self):
        """
        Запускает интерактивную командную строку для общения с агентом.
        Улучшенная версия с поддержкой команд и подсказками.
        """
        import os
        import readline  # Поддержка истории команд и автодополнения
        
        # Определение модели в зависимости от типа
        model_name = "Local DeepSeek-R1"
        if self.model_type == "openai":
            model_name = "OpenAI GPT-4o"
        elif self.model_type == "anthropic":
            model_name = "Anthropic Claude-3-7-Sonnet"
        
        print("\n" + "=" * 60)
        print(f"MCP Agent Command Interface - {model_name}")
        print("=" * 60)
        print("Available commands:")
        print("  !help       - Show this help message")
        print("  !exit       - Exit the program")
        print("  !run <code> - Execute Python code")
        print("  !cmd <cmd>  - Execute system command")
        print("  !clear      - Clear the screen")
        print("  !history    - Show chat history")
        print("  !model      - Show current model info")
        print("  !model <type> - Switch model type (local, openai, anthropic)")
        print("  !auto <on/off> - Enable/disable automatic command execution")
        print("=" * 60 + "\n")

        # История команд для интерфейса
        command_history = []

        while True:
            try:
                user_input = input("\n\033[1;32mYou:\033[0m ")  # Зеленый цвет для пользователя
                
                # Сохраняем в историю
                if user_input and user_input not in command_history:
                    command_history.append(user_input)
                
                # Обработка специальных команд
                if user_input.lower() == "!exit":
                    print("Exiting MCP Agent. Goodbye!")
                    break
                
                elif user_input.lower() == "!help":
                    print("\n" + "=" * 60)
                    print("MCP Agent Command Help")
                    print("=" * 60)
                    print("Available commands:")
                    print("  !help       - Show this help message")
                    print("  !exit       - Exit the program")
                    print("  !run <code> - Execute Python code")
                    print("  !cmd <cmd>  - Execute system command")
                    print("  !clear      - Clear the screen")
                    print("  !history    - Show chat history")
                    print("  !model      - Show current model info")
                    print("  !model <type> - Switch model type (local, openai, anthropic)")
                    print("  !auto <on/off> - Enable/disable automatic command execution")
                    print("=" * 60)
                    continue
                
                elif user_input.lower() == "!clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                    
                elif user_input.lower() == "!history":
                    print("\n--- Chat History ---")
                    for i, (user_msg, model_msg) in enumerate(self.dialog_history):
                        print(f"\n[{i+1}] User: {user_msg}")
                        print(f"    AI: {model_msg[:100]}{'...' if len(model_msg) > 100 else ''}")
                    continue
                    
                elif user_input.lower() == "!model":
                    print("\n--- Current Model Info ---")
                    print(f"Model type: {self.model_type}")
                    if self.model_type == "local":
                        print(f"Local model path: {self.model_path}")
                    else:
                        print(f"API model: {'GPT-4o' if self.model_type == 'openai' else 'Claude-3-7-Sonnet'}")
                    print(f"Automatic command execution: {'Enabled' if self.auto_execute_commands else 'Disabled'}")
                    print(f"Auto-execute threshold: {self.auto_execute_threshold}")
                    continue
                
                elif user_input.lower().startswith("!model "):
                    model_type = user_input[7:].strip().lower()
                    if model_type in ["local", "openai", "anthropic"]:
                        # Проверка доступности API ключей для API моделей
                        if model_type in ["openai", "anthropic"] and (not self.api_handler or model_type not in self.api_handler.api_keys):
                            print(f"\n\033[1;31mError:\033[0m {model_type.capitalize()} API key not provided. Cannot switch model.")
                            continue
                        
                        # Переключение модели
                        prev_model = self.model_type
                        self.model_type = model_type
                        print(f"\nSwitched model from {prev_model} to {model_type}")
                        
                        # Инициализация локальной модели при необходимости
                        if model_type == "local" and self.llm is None:
                            print("Initializing local model...")
                            temp_out = StringIO()
                            temp_err = StringIO()
                            with redirect_stdout(temp_out), redirect_stderr(temp_err):
                                self.llm = Llama(
                                    model_path=self.model_path,
                                    n_ctx=self.max_context_length,
                                    n_gpu_layers=24,
                                    verbose=False
                                )
                            print("Local model initialized successfully.")
                    else:
                        print(f"\n\033[1;31mError:\033[0m Invalid model type. Use 'local', 'openai', or 'anthropic'.")
                    continue
                
                elif user_input.lower().startswith("!auto "):
                    auto_setting = user_input[6:].strip().lower()
                    if auto_setting in ["on", "true", "yes", "1"]:
                        self.auto_execute_commands = True
                        print("Automatic command execution enabled.")
                    elif auto_setting in ["off", "false", "no", "0"]:
                        self.auto_execute_commands = False
                        print("Automatic command execution disabled.")
                    else:
                        print(f"\n\033[1;31mError:\033[0m Invalid setting. Use 'on' or 'off'.")
                    continue
                    
                elif user_input.lower().startswith("!run "):
                    code_to_execute = user_input[5:].strip()
                    if code_to_execute:
                        print(f"\nExecuting Python code:")
                        print("-" * 40)
                        print(code_to_execute)
                        print("-" * 40)
                        
                        result = self.execute_code(code_to_execute)
                        self.log_code_execution(code_to_execute, result)
                            
                        if result["status"] == "success":
                            print(f"\n\033[1;36mCode output:\033[0m")  # Голубой цвет для вывода
                            print(result['output'])
                        else:
                            print(f"\n\033[1;31mError:\033[0m {result.get('error_type', '')} - {result.get('error_message', '')}")
                    continue
                    
                elif user_input.lower().startswith("!cmd "):
                    command = user_input[5:].strip()
                    if command:
                        print(f"\nExecuting system command: {command}")
                        
                        result = self.execute_system_command(command)
                        self.log_command_execution(command, result)
                            
                        if result["status"] == "success":
                            print(f"\n\033[1;36mCommand output:\033[0m")  # Голубой цвет для вывода
                            print(result['output'])
                        else:
                            print(f"\n\033[1;31mError:\033[0m {result.get('error_message', result.get('error', 'Unknown error'))}")
                    continue

                # Обычный запрос к модели
                system_prompt = (
                    "You are a helpful AI assistant that can write Python code and suggest system commands. "
                    "When providing code, use ```python and ``` markers. "
                    "When suggesting system commands, use ```cmd and ``` markers. "
                    "You can execute commands yourself to help users complete tasks."
                )
                
                request = {
                    "prompt": user_input,
                    "system": system_prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.output_tokens_limit
                }

                response = self.handle_request(json.dumps(request))
                self.log_conversation(user_input, response)

                if "error" in response:
                    print(f"\n\033[1;31mError:\033[0m {response['error']['message']}")
                else:
                    ai_response = response['choices'][0]['text']
                    print(f"\n\033[1;34mAI:\033[0m {ai_response}")  # Синий цвет для ИИ
                    
                    # Извлечение кода Python и системных команд из ответа
                    code_blocks = re.findall(r'```python(.*?)```', ai_response, re.DOTALL)
                    cmd_blocks = re.findall(r'```cmd(.*?)```', ai_response, re.DOTALL)
                    
                    # Предложение выполнить код
                    if code_blocks:
                        user_choice = input("\nFound Python code. Execute? (y/n): ").lower()
                        if user_choice == 'y':
                            for i, code_block in enumerate(code_blocks):
                                code = code_block.strip()
                                print(f"\nExecuting code block {i+1}:")
                                print("-" * 40)
                                print(code)
                                print("-" * 40)
                                
                                result = self.execute_code(code)
                                self.log_code_execution(code, result)
                                    
                                if result["status"] == "success":
                                    print(f"\n\033[1;36mResult:\033[0m")
                                    print(result['output'])
                                else:
                                    print(f"\n\033[1;31mError:\033[0m {result.get('error_type', '')} - {result.get('error_message', '')}")
                    
                    # Предложение выполнить команды
                    if cmd_blocks:
                        user_choice = input("\nFound system commands. Execute? (y/n): ").lower()
                        if user_choice == 'y':
                            for i, cmd_block in enumerate(cmd_blocks):
                                cmd = cmd_block.strip()
                                print(f"\nExecuting command: {cmd}")
                                
                                result = self.execute_system_command(cmd)
                                self.log_command_execution(cmd, result)
                                    
                                if result["status"] == "success":
                                    print(f"\n\033[1;36mResult:\033[0m")
                                    print(result['output'])
                                else:
                                    print(f"\n\033[1;31mError:\033[0m {result.get('error_message', result.get('error', 'Unknown error'))}")
            
            except KeyboardInterrupt:
                print("\nOperation interrupted. Type !exit to quit.")
            except Exception as e:
                print(f"\n\033[1;31mError:\033[0m An unexpected error occurred: {str(e)}")
                self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)

    def run_tests(self):
        # Тест 1: проверка ограничения длины очереди
        for i in range(15):
            self.dialog_history.append((f"test_user_{i}", f"test_model_{i}"))
        assert len(self.dialog_history) == 10, "Очередь должна содержать максимум 10 элементов"

        # Тест 2: проверка формата промпта
        prompt = self.format_prompt("system message", "test user message")
        assert self.validate_prompt(prompt), "Неверный формат промпта"

        # Тест 3: проверка обработки фидбэка
        initial_penalty = self.presence_penalty
        self.handle_request(json.dumps({"prompt": "test", "feedback": "negative"}))
        assert self.presence_penalty > initial_penalty, "presence_penalty не изменился после отрицательного фидбэка"

        print("Все тесты пройдены успешно!")

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.llm.tokenize(text.encode('utf-8')))
        except Exception as e:
            raise ValueError(f"Tokenization failed: {str(e)}") from e

    def error_response(self, message, req_id):
        return {
            "id": req_id,
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        }

    def start_server(self, port=8000):
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/generate', methods=['POST'])
        def generate():
            response = self.handle_request(request.data.decode('utf-8'))
            return jsonify(response)

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "ready": True,
                "model_id": "DeepSeek-R1-Distill-Qwen-7B-GGUF",
                "max_context_length": self.max_context_length
            })

        print(f"MCP Server running on port {port}")
        app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    # Получаем выбор модели от пользователя
    model_type, api_keys = choose_model()
    
    # Установка пути к локальной модели
    model_path = "/home/dima/.cache/gpt4all/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    
    # Создание экземпляра MCP с выбранной моделью
    mcp = MCPCompliantModel(
        model_type=model_type,
        model_path=model_path,
        api_keys=api_keys,
        max_context_length=32768,
        temperature=0.7,
        input_tokens_limit=4500,
        output_tokens_limit=2048,
        top_p=0.9,
        presence_penalty=0.2,
        auto_execute_commands=True
    )
    
    # Запуск интерфейса
    mcp.chat_cli()