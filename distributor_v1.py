# моя задача - создать программу, которая будет задавать вопрос о выборе модели для генерации текста,
# и можно будет выбрать , режим чата или отправлять туда запросы
import os
from model_clients import antropic, deepseek, gpt4

class ModelDistributor:
    def __init__(self):
        self.anthropic_api_key = "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"
        self.openai_api_key = "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"
        
        # Используем словарь с понятными именами и соответствующими им объектами клиентов
        self.models = {
            "Claude-3.7-sonnet": antropic.AnthropicClient(self.anthropic_api_key),
            "GPT-4o": gpt4.GPTClient(self.openai_api_key),
            "LM-Studio (Deepseek)": deepseek.LMStudioClient()
        }

    def choose_model(self):
        while True:
            print("Выберите модель:")
            # Создаем список имен моделей
            model_names = list(self.models.keys())
            
            # Печатаем имена с номерами
            for idx, name in enumerate(model_names, 1):
                print(f"{idx}. {name}")
            
            try:
                choice = int(input("Введите номер модели: "))
                
                if 1 <= choice <= len(model_names):
                    # Получаем выбранное имя модели
                    selected_name = model_names[choice - 1]
                    # Возвращаем соответствующий объект модели
                    return self.models[selected_name]
                else:
                    print(f"Ошибка: выберите число от 1 до {len(model_names)}")
            
            except ValueError:
                print("Ошибка: введите числовой номер модели")

    def select_mode(self):
        print("Выберите режим:")
        print("1. Чат")
        print("2. Запрос")
        choice = int(input("Введите номер режима: "))
        return choice
    
    def chat_mode(self, model):
    # Просто вызываем встроенный метод модели
        model.chat_mode()
    
    def request_mode(self, model):
    # Просто вызываем встроенный метод модели
        model.request_mode()

    def main(self):
        while True:
            model = self.choose_model()
            mode = self.select_mode()
            if mode == 1:
                self.chat_mode(model)
            elif mode == 2:
                self.request_mode(model)
                
if __name__ == "__main__":
    distributor = ModelDistributor()
    distributor.main()