##код для взаимодействия с GPT4o ##
from openai import OpenAI
import time

class GPTClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.max_history = 5  # Хранить историю из 5 последних обменов
        
    def generate_response(self, prompt):
        """Режим единичного запроса без сохранения истории"""
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
        
    def chat_mode(self):
        """Интерактивный режим чата с памятью последних сообщений"""
        print("\n=== Режим чата с GPT-4o ===")
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
            
            # Отправляем запрос к API
            completion = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )
            response = completion.choices[0].message.content
            
            # Сохраняем обмен в историю
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": time.time()
            })
            
            print(f"GPT-4o: {response}\n")
            
    def request_mode(self):
        """Режим единичных запросов без сохранения контекста"""
        print("\n=== Режим запросов к GPT-4o ===")
        print("Введите 'выход' чтобы вернуться в меню\n")
        
        while True:
            user_input = input("Запрос: ")
            if user_input.lower() == "выход":
                break
                
            response = self.generate_response(user_input)
            print(f"Ответ: {response}\n")
            
            continue_prompt = input("Еще запрос? (да/нет): ")
            if continue_prompt.lower() != "да":
                break

if __name__ == "__main__":
    openai_api_key = " "
    client = GPTClient(openai_api_key)
    
    # Пример использования обоих режимов
    # client.chat_mode()
    # client.request_mode()
    
    # Стандартный режим для обратной совместимости
    response = client.generate_response("Write a one-sentence bedtime story about the alien (movie).")
    print(response)
