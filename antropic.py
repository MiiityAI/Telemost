### код, который используется для взаимодействия с Anthropic API
import requests
import json
import time

class AnthropicClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = "claude-3-7-sonnet-20250219"
        self.conversation_history = []
        self.max_history = 5  # Хранить историю из 5 последних обменов
        
    def send_message(self, prompt, max_tokens=4000, temperature=0.9):
        """Режим единичного запроса без сохранения истории"""
        # Set up the headers for authentication and content type
        headers = {
            "Content-Type": "application/json", 
            "x-api-key": self.api_key, 
            "anthropic-version": "2023-06-01"
        }
        # Create the message content
        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        } 
        # Send the request to Anthropic
        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload
        )
        # Parse the JSON response
        result = response.json()        
        return result["content"][0]["text"]
        
    def chat_mode(self):
        """Интерактивный режим чата с памятью последних сообщений"""
        print("\n=== Режим чата с Claude ===")
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
            
            # Set up the headers for authentication and content type
            headers = {
                "Content-Type": "application/json", 
                "x-api-key": self.api_key, 
                "anthropic-version": "2023-06-01"
            }
            
            # Create the message content with history
            payload = {
                "model": self.default_model,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.9
            } 
            
            # Send the request to Anthropic
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            )
            
            # Parse the JSON response
            result = response.json()
            response_text = result["content"][0]["text"]
            
            # Сохраняем обмен в историю
            self.conversation_history.append({
                "user": user_input,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            print(f"Claude: {response_text}\n")
            
    def request_mode(self):
        """Режим единичных запросов без сохранения контекста"""
        print("\n=== Режим запросов к Claude ===")
        print("Введите 'выход' чтобы вернуться в меню\n")
        
        while True:
            user_input = input("Запрос: ")
            if user_input.lower() == "выход":
                break
                
            response = self.send_message(user_input)
            print(f"Ответ: {response}\n")
            
            continue_prompt = input("Еще запрос? (да/нет): ")
            if continue_prompt.lower() != "да":
                break

if __name__ == "__main__":
    anthropic_api_key = " "
    client = AnthropicClient(anthropic_api_key)
    
    # Пример использования обоих режимов
    # client.chat_mode()
    # client.request_mode()
    
    # Стандартный режим для обратной совместимости
    response = client.send_message("Hello, Claude!")
    print(response)
