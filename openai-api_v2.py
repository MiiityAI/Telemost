#!/usr/bin/env python3
import requests
import json
import time
import os
import logging
import re
import yaml
import random  # Добавлен импорт для случайной выборки блоков
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Iterator, Optional, Union

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAI_API:
    """API class for OpenAI-compatible endpoints like LM Studio."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio",
                 max_history_length: int = 10,
                 system_message: str = "You are a helpful assistant.",
                 config_path: str = "config.yaml"):
        """
        Initialize OpenAI API connection.
        
        Args:
            base_url: URL for API (default: "http://localhost:1234/v1")
            api_key: API key (default: "lm-studio")
            max_history_length: Maximum number of messages in history
            system_message: System message to use
            config_path: Path to configuration file
        """
        self.base_url = base_url
        self.api_key = api_key
        self.max_history_length = max_history_length
        
        # Загрузка настроек из config.yaml
        self.config = self.load_config(config_path)
        self.temperature = self.config.get('temperature', 0.2)  # По умолчанию 0.2
        
        # Инициализация истории с системным сообщением
        self.message_history = [
            {"role": "system", "content": system_message}
        ]
        
        # Проверка соединения при инициализации
        self.check_connection()
    
    def load_config(self, config_path: str) -> dict:
        """Загружает конфигурацию из YAML файла."""
        config = {'temperature': 0.2}  # Значение по умолчанию
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and isinstance(yaml_config, dict):
                        config.update(yaml_config)
                logger.info(f"Конфигурация загружена из {config_path}")
            else:
                logger.warning(f"Файл конфигурации {config_path} не найден. Используются значения по умолчанию.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
        
        return config
        
    def check_connection(self) -> bool:
        """
        Check if connection to API is working.
        
        Returns:
            bool: True if connection works, False otherwise
        """
        try:
            models_url = f"{self.base_url}/models"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.get(models_url, headers=headers, timeout=600)
            
            if response.status_code == 200:
                models = response.json().get("data", [])
                logger.info(f"Connected to API. Available models: {len(models)}")
                return True
            else:
                logger.warning(f"API connection check failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error checking API connection: {str(e)}")
            return False
    
    def _truncate_messages_to_fit(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Обрезает сообщения, чтобы они поместились в ограничение токенов.
        Сохраняет системное сообщение и последний запрос пользователя.
        
        Args:
            messages: Список сообщений
            max_tokens: Максимальное количество токенов
            
        Returns:
            List[Dict]: Обрезанный список сообщений
        """
        # Простая эвристика: примерно 4 символа на токен
        chars_per_token = 4
        total_chars = sum(len(msg["content"]) for msg in messages)
        estimated_tokens = total_chars / chars_per_token
        
        # Если вместимся в лимит, возвращаем как есть
        if estimated_tokens <= max_tokens:
            return messages
        
        # Выделяем системное сообщение и последнее сообщение пользователя
        system_msg = None
        last_user_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg
            elif msg["role"] == "user" and last_user_msg is None:
                # Берем только самое последнее сообщение пользователя
                last_user_msg = msg
            else:
                other_msgs.append(msg)
        
        # Оцениваем сколько символов у нас уже занято критичными сообщениями
        critical_chars = 0
        if system_msg:
            critical_chars += len(system_msg["content"])
        if last_user_msg:
            critical_chars += len(last_user_msg["content"])
        
        # Вычисляем сколько символов осталось для остальных сообщений
        max_chars = int(max_tokens * chars_per_token)
        remaining_chars = max(0, max_chars - critical_chars)
        
        # Сортируем остальные сообщения от новых к старым
        other_msgs.reverse()
        
        # Добавляем сообщения, пока влезают
        included_msgs = []
        for msg in other_msgs:
            msg_len = len(msg["content"])
            if msg_len <= remaining_chars:
                included_msgs.append(msg)
                remaining_chars -= msg_len
            else:
                # Если сообщение слишком большое, можно обрезать, но лучше пропустить
                break
        
        # Собираем финальный список сообщений в правильном порядке
        truncated_messages = []
        if system_msg:
            truncated_messages.append(system_msg)
        truncated_messages.extend(reversed(included_msgs))  # Возвращаем исходный порядок
        if last_user_msg:
            truncated_messages.append(last_user_msg)
        
        logger.info(f"Обрезано сообщений: было {len(messages)}, стало {len(truncated_messages)}")
        return truncated_messages
    
    def _extract_context_limit(self, error_message: str) -> int:
        """
        Извлекает максимальный размер контекста из сообщения об ошибке.
        
        Args:
            error_message: Сообщение об ошибке
            
        Returns:
            int: Максимальный размер контекста или 4000 по умолчанию
        """
        logger.info(f"Анализ сообщения об ошибке: {error_message[:200]}")
        
        # Типичный формат ошибки LM Studio
        match = re.search(r'model is loaded with context length of only (\d+) tokens', error_message)
        if match:
            limit = int(match.group(1))
            logger.info(f"Найден лимит контекста (формат 1): {limit}")
            return max(1000, int(limit * 0.9))
        
        # Альтернативный формат
        match = re.search(r'context length of only (\d+) tokens', error_message)
        if match:
            limit = int(match.group(1))
            logger.info(f"Найден лимит контекста (формат 2): {limit}")
            return max(1000, int(limit * 0.9))
        
        # Формат "Trying to keep X tokens"
        match = re.search(r'Trying to keep the first (\d+) tokens', error_message)
        if match:
            current = int(match.group(1))
            
            # Ищем упоминание о лимите модели
            limit_match = re.search(r'context length of only (\d+) tokens', error_message)
            if limit_match:
                limit = int(limit_match.group(1))
                logger.info(f"Найдено: текущие токены = {current}, лимит модели = {limit}")
                return max(1000, int(limit * 0.9))
            else:
                # Если лимит не найден, используем 70% от текущего размера
                logger.info(f"Найдено только текущее количество токенов: {current}")
                return max(1000, int(current * 0.7))
        
        logger.warning("Не удалось определить лимит из сообщения об ошибке, используем 4000")
        return 4000
    
    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """
        Оценивает количество токенов в сообщениях.
        
        Args:
            messages: Список сообщений
        
        Returns:
            int: Примерное количество токенов
        """
        # Примерное соотношение: 1 токен ~= 4 символа
        chars_per_token = 4
        
        # Добавляем базовые токены для метаданных сообщений (роли и т.д.)
        base_tokens = 4 * len(messages)
        
        # Считаем символы в сообщениях
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        
        # Оцениваем общее количество токенов
        return base_tokens + (total_chars // chars_per_token)

    def smart_truncate_chunk(self, chunk_text, target_length, min_block_size=50):
        """
        Интеллектуально обрезает текст, сохраняя начало, конец и случайную выборку из середины.
        
        Args:
            chunk_text: Исходный текст для обрезки
            target_length: Целевая длина текста
            min_block_size: Минимальный размер блока для рассмотрения
        
        Returns:
            str: Обрезанный текст
        """
        # Разбиваем на абзацы или предложения
        blocks = re.split(r'\n\n|\. ', chunk_text)
        blocks = [b for b in blocks if len(b) >= min_block_size]
        
        if not blocks or sum(len(b) for b in blocks) <= target_length:
            return chunk_text[:target_length]
        
        # Всегда сохраняем начало и конец
        intro = blocks[0]
        outro = blocks[-1]
        
        # Сколько места осталось на середину
        remaining_length = target_length - len(intro) - len(outro) - 20  # 20 для маркеров обрезки
        
        if remaining_length <= 0:
            # Если места не хватает, возвращаем только начало
            return intro[:target_length-10] + "..."
        
        # Выбираем случайные блоки из середины
        middle_blocks = blocks[1:-1]
        selected_blocks = []
        current_length = 0
        
        # Случайно перемешиваем блоки
        random.shuffle(middle_blocks)
        
        for block in middle_blocks:
            if current_length + len(block) <= remaining_length:
                selected_blocks.append(block)
                current_length += len(block)
        
        # Собираем финальный текст
        result = intro + " ... " + " ".join(selected_blocks) + " ... " + outro
        return result
    
    def _handle_context_overflow(self, response, api_url, headers, payload):
        """
        Интеллектуальная обработка ошибки превышения контекста.
        Использует точный расчет коэффициента обрезки и интеллектуальную выборку текста.
        """
        error_msg = response.text
        logger.info(f"Обнаружено превышение контекста: {error_msg[:200]}")
        
        # Извлекаем лимит токенов из сообщения об ошибке
        max_tokens = self._extract_context_limit(error_msg)
        logger.info(f"Обнаружен лимит контекста: {max_tokens} токенов")
        
        # Извлекаем сообщения
        original_messages = payload["messages"]
        system_msg = None
        last_user_msg = None
        
        for msg in original_messages:
            if msg["role"] == "system":
                system_msg = msg
            elif msg["role"] == "user":
                last_user_msg = msg
        
        # Проверка на наличие необходимых сообщений
        if not last_user_msg:
            return "Error: No user message found in context."
        
        # Оцениваем текущее количество токенов
        current_tokens = self._estimate_tokens(original_messages)
        logger.info(f"Оценка текущего размера контекста: {current_tokens} токенов")
        
        # Вычисляем коэффициент обрезки
        required_reduction_ratio = max_tokens / current_tokens if current_tokens > 0 else 0.5
        
        # Ограничиваем коэффициент от 0.1 до 0.9
        required_reduction_ratio = max(0.1, min(0.9, required_reduction_ratio))
        logger.info(f"Рассчитанный коэффициент обрезки: {required_reduction_ratio:.2f}")
        
        # Определяем доступное количество токенов для контента
        metadata_tokens = 50  # Tokens for message metadata
        available_tokens = int(max_tokens * required_reduction_ratio) - metadata_tokens
        
        # Формируем новый список сообщений
        new_messages = []
        
        # Определяем долю токенов для системного сообщения (если оно есть)
        system_tokens = 0
        if system_msg:
            # Максимум 10% для системного сообщения
            max_system_tokens = min(int(available_tokens * 0.1), 200)
            max_system_chars = max_system_tokens * 4  # Примерно 4 символа на токен
            
            # Копируем и обрезаем системное сообщение, если нужно
            system_copy = dict(system_msg)
            if len(system_copy["content"]) > max_system_chars:
                system_copy["content"] = self.smart_truncate_chunk(system_copy["content"], max_system_chars)
            
            new_messages.append(system_copy)
            system_tokens = max_system_tokens
        
        # Оставшиеся токены для сообщения пользователя
        user_tokens = available_tokens - system_tokens
        user_chars = user_tokens * 4  # Примерно 4 символа на токен
        
        # Проверяем, это запрос для объединения файлов?
        content = last_user_msg["content"]
        is_file_merge = "Файл 1:" in content and "Файл 2:" in content
        
        if is_file_merge:
            # Обработка слияния файлов
            intro_end = content.find("Файл 1:")
            file1_start = intro_end
            file2_start = content.find("Файл 2:")
            
            # Выделяем части
            intro = content[:intro_end].strip()
            file1_content = content[file1_start:file2_start].strip()
            file2_content = content[file2_start:].strip()
            
            # Распределяем доступные символы
            intro_chars = min(len(intro), int(user_chars * 0.1))
            remaining_chars = user_chars - intro_chars
            
            # Распределяем оставшиеся символы между файлами (пропорционально)
            file1_len = len(file1_content)
            file2_len = len(file2_content)
            total_files_len = file1_len + file2_len
            
            if total_files_len > 0:
                file1_ratio = file1_len / total_files_len
            else:
                file1_ratio = 0.5
            
            file1_chars = min(file1_len, int(remaining_chars * file1_ratio))
            file2_chars = min(file2_len, int(remaining_chars * (1 - file1_ratio)))
            
            # Интеллектуально обрезаем части
            intro_trimmed = intro[:intro_chars]
            file1_trimmed = self.smart_truncate_chunk(file1_content, file1_chars)
            file2_trimmed = self.smart_truncate_chunk(file2_content, file2_chars)
            
            # Формируем новый контент
            new_content = f"{intro_trimmed}\n\nФайл 1:{file1_trimmed}\n\nФайл 2:{file2_trimmed}\n\n[Внимание: файлы были интеллектуально обрезаны из-за ограничений контекста]"
        else:
            # Обычное сообщение - интеллектуальная обрезка
            new_content = self.smart_truncate_chunk(content, user_chars)
        
        # Создаем копию сообщения пользователя с обрезанным содержимым
        user_copy = dict(last_user_msg)
        user_copy["content"] = new_content
        new_messages.append(user_copy)
        
        # Обновляем payload
        payload["messages"] = new_messages
        
        # Оцениваем размер обрезанного контекста
        estimated_tokens = self._estimate_tokens(new_messages)
        logger.info(f"Оценка размера обрезанного контекста: {estimated_tokens} токенов (лимит: {max_tokens})")
        
        # Отправляем запрос с обрезанным контекстом
        try:
            logger.info(f"Отправка интеллектуально обрезанного запроса ({len(new_messages)} сообщений)")
            retry_response = requests.post(api_url, headers=headers, json=payload, timeout=600)
            
            if retry_response.status_code == 200:
                result = retry_response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Добавляем предупреждение
                warning = f"⚠️ ВНИМАНИЕ: Контент был интеллектуально обрезан из-за ограничений модели. Сохранены ключевые части сообщения.\n\n"
                return warning + content
            else:
                # Если интеллектуальная обрезка не помогла, делаем экстремальное обрезание
                logger.warning(f"Интеллектуальная обрезка не удалась: {retry_response.status_code} - {retry_response.text[:100]}")
                
                # Экстремальное обрезание
                if system_msg:
                    system_copy = {"role": "system", "content": "Помоги с анализом."}
                    extreme_messages = [system_copy]
                else:
                    extreme_messages = []
                
                if is_file_merge:
                    user_content = "Объедини информацию из двух файлов (они были обрезаны из-за ограничений контекста)."
                else:
                    user_content = "Сообщение было слишком длинным. Помоги с ответом на основе этой информации: " + content[:200]
                
                user_copy = {"role": "user", "content": user_content}
                extreme_messages.append(user_copy)
                
                payload["messages"] = extreme_messages
                
                # Последняя попытка с экстремально обрезанным контекстом
                extreme_response = requests.post(api_url, headers=headers, json=payload, timeout=600)
                
                if extreme_response.status_code == 200:
                    result = extreme_response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Добавляем предупреждение о сильной обрезке
                    warning = f"⚠️⚠️⚠️ ВНИМАНИЕ: Контент был экстремально обрезан из-за ограничений модели. Сохранена только базовая информация.\n\n"
                    return warning + content
                else:
                    return f"Error: Не удалось обработать запрос даже после экстремальной обрезки контекста. Попробуйте уменьшить размер исходных файлов. Код ошибки: {extreme_response.status_code}"
        except Exception as e:
            logger.error(f"Ошибка при отправке обрезанного запроса: {str(e)}")
            return f"Error: Произошла ошибка при обработке запроса: {str(e)}"

    def query(self, messages: Union[List[Dict], str], 
              model: str = "local-model", 
              temperature: float = None,
              stream: bool = False,
              maintain_history: bool = True) -> str:
        """Send query to API."""
        api_url = f"{self.base_url}/chat/completions"
        
        # Используем температуру из параметров или из конфигурации
        if temperature is None:
            temperature = self.temperature
        
        # Convert string to message dict if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Prepare messages for API
        if maintain_history:
            # Add only new user message to history
            if len(messages) > 0 and messages[-1]["role"] == "user":
                self.message_history.append(messages[-1])
            
            messages_to_send = self.message_history
        else:
            # Use only provided messages
            messages_to_send = messages
        
        # Prepare payload
        payload = {
            "model": model,
            "messages": messages_to_send,
            "temperature": temperature,
            "stream": stream
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.info(f"Sending request to {api_url}")
            response = requests.post(api_url, headers=headers, json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Add response to history if maintaining it
                if maintain_history:
                    self.message_history.append({"role": "assistant", "content": content})
                    
                    # Trim history if needed
                    if len(self.message_history) > self.max_history_length + 1:  # +1 for system
                        system_msg = self.message_history[0]
                        self.message_history = [system_msg] + self.message_history[-(self.max_history_length):]
                
                return content
            elif response.status_code == 400 and "context" in response.text and "tokens" in response.text:
                # Обработка проблемы с превышением контекста
                content = self._handle_context_overflow(response, api_url, headers, payload)
                
                # Обновляем историю сообщений, если нужно
                if maintain_history and not content.startswith("Error:"):
                    # Убираем предупреждение из сообщения для истории
                    content_without_warning = re.sub(r'^⚠️ WARNING:[^\n]*\n\n', '', content)
                    self.message_history.append({"role": "assistant", "content": content_without_warning})
                    
                    # Обновляем историю на основе payload, который был использован для успешного запроса
                    self.message_history = payload["messages"] + [{"role": "assistant", "content": content_without_warning}]
                
                return content
            else:
                error_msg = f"API Error: {response.status_code} - {response.text[:200]}"
                logger.error(error_msg)
                return f"Error: Cannot connect to API ({error_msg})"
                
        except Exception as e:
            error_msg = f"Error: {str(e)[:200]}"
            logger.error(error_msg)
            return f"Error: Cannot connect to API ({error_msg})"
    
    def query_stream(self, messages: Union[List[Dict], str], 
                    model: str = "local-model",
                    temperature: float = None,
                    maintain_history: bool = True) -> Iterator[str]:
        """Stream responses from API."""
        api_url = f"{self.base_url}/chat/completions"
        
        if temperature is None:
            temperature = self.temperature
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if maintain_history:
            if len(messages) > 0 and messages[-1]["role"] == "user":
                self.message_history.append(messages[-1])
            
            messages_to_send = self.message_history
        else:
            messages_to_send = messages
        
        payload = {
            "model": model,
            "messages": messages_to_send,
            "temperature": temperature,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, stream=True)
            
            if response.status_code == 200:
                full_content = ""
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and data['choices']:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        full_content += content
                                        yield content
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {line}")
            
            if maintain_history and full_content:
                self.message_history.append({"role": "assistant", "content": full_content})
                
                if len(self.message_history) > self.max_history_length + 1:
                    system_msg = self.message_history[0]
                    self.message_history = [system_msg] + self.message_history[-(self.max_history_length):]
            elif response.status_code == 400 and "context" in response.text and "tokens" in response.text:
            # При ошибке контекста для стриминга, возвращаем предупреждение
                yield "⚠️ WARNING: Context too large. Only kept system and latest query.\n\n"
            
            # Извлекаем лимит токенов
            match = re.search(r'context length of only (\d+) tokens', response.text)
            max_tokens = int(int(match.group(1)) * 0.9 if match else 4000)
            
            # Оставляем только критичные сообщения
            new_messages = []
            system_msg = None
            last_user_msg = None
            
            for msg in messages_to_send:
                if msg["role"] == "system":
                    system_msg = msg
                elif msg["role"] == "user":
                    last_user_msg = msg
            
            if system_msg:
                new_messages.append(system_msg)
            if last_user_msg:
                new_messages.append(last_user_msg)
            
            # Проверка на превышение лимита
            total_chars = sum(len(msg["content"]) for msg in new_messages)
            if total_chars / 4 > max_tokens and new_messages:
                # Удаляем системное сообщение если нужно
                if len(new_messages) > 1:
                    new_messages = [new_messages[1]]  # Оставляем только сообщение пользователя
                
                # Если даже это не помогает, обрезаем сообщение пользователя
                if len(new_messages[0]["content"]) / 4 > max_tokens:
                    max_chars = int(max_tokens * 4 * 0.8)
                    new_messages[0]["content"] = new_messages[0]["content"][:max_chars]
            
            # Обновляем payload и пробуем снова
            payload["messages"] = new_messages
            payload["stream"] = True
            
            retry_response = requests.post(api_url, headers=headers, json=payload, stream=True)
            
            if retry_response.status_code == 200:
                full_content = ""
                
                for line in retry_response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and data['choices']:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        full_content += content
                                        yield content
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {line}")
                
                if maintain_history and full_content:
                    self.message_history = new_messages + [{"role": "assistant", "content": full_content}]
            else:
                yield f"Error: Still failed after truncating context: {retry_response.status_code}"
            
        except Exception as e:
            yield f"Error: {str(e)[:200]}"
    
    def query_with_retry(self, messages: Union[List[Dict], str], 
                         model: str = "local-model",
                         temperature: float = None,
                         max_retries: int = 3, 
                         maintain_history: bool = True) -> str:
        """
        Try to query API with retries on failure.
        
        Args:
            messages: Either a string or list of message dicts
            model: Model identifier
            temperature: Temperature for generation (default from config or 0.2)
            max_retries: Maximum number of retries
            maintain_history: Whether to maintain conversation history
            
        Returns:
            str: Response from model
        """
        # Используем температуру из параметров или из конфигурации
        if temperature is None:
            temperature = self.temperature
        
        for attempt in range(max_retries):
            try:
                response = self.query(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    stream=False,
                    maintain_history=maintain_history
                )
                
                # Проверка на контекстное предупреждение
                if "⚠️ WARNING: The context was too large" in response:
                    # Предупреждение о контексте, но не ошибка - возвращаем результат
                    return response
                
                # Check if error
                if response.startswith("Error:"):
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {response}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                
                return response
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return "Error: Could not get response after multiple attempts"
    
    def clear_history(self):
        """Clear conversation history, keeping only the system message."""
        system_message = self.message_history[0]
        self.message_history = [system_message]
        logger.info("Conversation history cleared")
    
    def set_system_message(self, message: str):
        """
        Set a new system message.
        
        Args:
            message: New system message
        """
        if self.message_history and self.message_history[0]["role"] == "system":
            self.message_history[0]["content"] = message
        else:
            self.message_history.insert(0, {"role": "system", "content": message})
            
        logger.info(f"System message set to: {message[:50]}...")

    def list_available_models(self) -> List[str]:
        """
        List available models from the API.
        
        Returns:
            List[str]: List of model IDs
        """
        try:
            models_url = f"{self.base_url}/models"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.get(models_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                model_ids = [model.get("id", "unknown") for model in models_data]
                return model_ids
            else:
                logger.warning(f"Failed to list models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

def save_dialog(dialog_history: List[Dict], output_dir: Optional[Path] = None):
    """
    Save dialog history to a file.
    
    Args:
        dialog_history: List of dialog entries
        output_dir: Optional output directory (default: "./dialogues")
    """
    if output_dir is None:
        output_dir = Path("./dialogues")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_dialog.md"
    file_path = output_dir / filename
    
    content = f"# История диалога\n"
    content += f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, msg in enumerate(dialog_history):
        if msg["role"] == "system":
            continue  # Пропускаем системные сообщения
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        content += f"## Сообщение {i}\n"
        content += f"**{role}:** {msg['content']}\n\n"
        content += "---\n\n"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"История диалога сохранена в {file_path}")
    return file_path

def main():
    """Main function to run interactive CLI with OpenAI API."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI API CLI")
    parser.add_argument("--url", default="http://localhost:1234/v1", help="API base URL")
    parser.add_argument("--key", default="lm-studio", help="API key")
    parser.add_argument("--system", default="You are an algorithm, that structurizes texts and finds new senses in documents", 
                       help="System message")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (default from config.yaml or 0.2)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    # Create API instance
    api = OpenAI_API(
        base_url=args.url,
        api_key=args.key,
        system_message=args.system,
        config_path=args.config
    )
    
    # Перезаписываем температуру из командной строки, если она указана
    if args.temperature is not None:
        api.temperature = args.temperature
    
    # Check connection
    if not api.check_connection():
        print("⚠️ Failed to connect to API. Make sure LM Studio is running.")
        retry = input("Retry connection? (y/n): ").lower()
        if retry != 'y':
            return
        
        if not api.check_connection():
            print("⚠️ Still can't connect. Exiting.")
            return
    
    # Get available models
    models = api.list_available_models()
    if models:
        print(f"Available models: {', '.join(models)}")
        model = models[0]  # Use first available model
    else:
        print("⚠️ No models found. Using default model name.")
        model = "local-model"
    
    print("\n🤖 LM Studio Chat")
    print(f"System: \"{args.system}\"")
    print(f"Temperature: {api.temperature}")
    print("\nCommands:")
    print("  [save] - Save dialog history")
    print("  [clear] - Clear conversation history")
    print("  [exit] - Exit chat")
    print("  [merge] - Run insight generator")
    
    dialog_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Handle commands
        if user_input.lower() == "[exit]":
            print("Goodbye! 👋")
            break
            
        elif user_input.lower() == "[clear]":
            api.clear_history()
            dialog_history = []
            print("Conversation history cleared.")
            continue
            
        elif user_input.lower() == "[save]":
            file_path = save_dialog(dialog_history)
            print(f"Dialog saved to {file_path}")
            continue
            
        elif user_input.lower() == "[merge]":
            print("\n--- Запуск генератора инсайтов ---")
            
            # Подготовка компонентов для insight_finder
            components = {
                "vault_path": os.getcwd(),  # Текущая директория
                "config_path": args.config,
                "api_client": api  # Передаем текущий экземпляр API
            }
            
            # Перед выполнением run_insight_finder
            print("Подготовка к отправке запроса в LM-Studio...\n")
            
            # Запускаем генератор инсайтов
            from insight_finder import run_insight_finder
            result = run_insight_finder(components)
            
            # Показываем результат
            print("\n--- Результат генерации инсайтов ---")
            print(result)
            continue
        
        # Send query to API
        print("\nAssistant: ", end="", flush=True)
        
        # Use streaming for better user experience
        full_response = ""
        for chunk in api.query_stream(
            messages=user_input,
            model=model,
            temperature=api.temperature
        ):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()  # Add newline after response
        
        # Add to dialog history
        dialog_history.append({
            "user": user_input,
            "assistant": full_response,
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    main()