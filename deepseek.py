##код для работы с локальной LM-Studio моделью, надо включить для её работы LM -Studio##
import requests
import time

class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url.rstrip("/")  # Убираем слэш в конце если есть
        self.chat_endpoint = f"{self.base_url}/api/v0/chat/completions"
        self.conversation_history = []
        self.max_history = 5  # Хранить историю из 5 последних обменов
        self.default_model = "deepseek-coder-v2-lite-instruct"

    def generate_response(self, messages, model="deepseek-coder-v2-lite-instruct", 
                        temperature=0.7, max_tokens=-1, stream=False):
        """
        Генерирует ответ от модели LM Studio
        
        :param messages: Список сообщений в формате [{"role": "user", "content": "text"}, ...]
        :param model: Название модели (по умолчанию deepseek-coder-v2-lite-instruct)
        :param temperature: Температура генерации (0.0-1.0)
        :param max_tokens: Максимальное количество токенов в ответе (-1 = без ограничений)
        :param stream: Включить потоковую передачу (SSE)
        :return: Текст ответа модели
        """
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.chat_endpoint,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30  # Таймаут 30 секунд
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"Ошибка {response.status_code}: {response.text}"
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка подключения: {str(e)}")
            
    def chat_mode(self):
        """Интерактивный режим чата с памятью последних сообщений"""
        print("\n=== Режим чата с LM-Studio (Deepseek) ===")
        print("Введите 'выход' чтобы вернуться в меню\n")
        
        while True:
            user_input = input("Вы: ")
            if user_input.lower() == "выход":
                break
                
            # Создаем сообщения с учетом истории
            messages = []
            for exchange in self.conversation_history[-self.max_history:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
                
            # Добавляем текущий запрос
            messages.append({"role": "user", "content": user_input})
            
            # Отправляем запрос с историей
            response_text = self.generate_response(
                messages=messages,
                model=self.default_model,
                temperature=0.7
            )
            
            # Сохраняем обмен в историю
            self.conversation_history.append({
                "user": user_input,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            print(f"Deepseek: {response_text}\n")
            
    def request_mode(self):
        """Режим единичных запросов без сохранения контекста"""
        print("\n=== Режим запросов к LM-Studio (Deepseek) ===")
        print("Введите 'выход' чтобы вернуться в меню\n")
        
        while True:
            user_input = input("Запрос: ")
            if user_input.lower() == "выход":
                break
                
            messages = [{"role": "user", "content": user_input}]
            response = self.generate_response(messages)
            print(f"Ответ: {response}\n")
            
            continue_prompt = input("Еще запрос? (да/нет): ")
            if continue_prompt.lower() != "да":
                break

if __name__ == "__main__":
    # Пример использования
    client = LMStudioClient()
    
    # Пример использования обоих режимов
    # client.chat_mode()
    # client.request_mode()
    
    # Стандартный режим для обратной совместимости
    messages = [{"role": "user", "content": "What day is it today?"}]
    response = client.generate_response(messages)
    print("Ответ модели:")
    print(response)