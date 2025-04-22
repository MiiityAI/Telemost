#!/usr/bin/env python3
import os
import networkx as nx
import pickle
import logging
import re
from pathlib import Path
import argparse
import yaml

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Значения по умолчанию
DEFAULT_CONFIG = {
    "semantic_threshold": 0.7,
    "path_threshold": 6,
    "output_dir": "_ai_merged",
    "cache_dir": ".cache",
    "merge_cosinus_threshold": 0.75
}

def load_config(config_path="config.yaml"):
    """Загружает конфигурацию из файла или возвращает значения по умолчанию."""
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if yaml_config and isinstance(yaml_config, dict):
                    # Обновляем только существующие ключи
                    for key in config:
                        if key in yaml_config:
                            config[key] = yaml_config[key]
            logging.info(f"Загружена конфигурация из {config_path}")
        except Exception as e:
            logging.error(f"Ошибка при загрузке конфигурации: {e}")
    
    return config

class InsightGenerator:
    """Класс для генерации инсайтов на основе анализа графа документов."""
    
    def __init__(self, vault_path, output_dir=None, merge_cosinus_threshold=None,
                 path_threshold=None, config_path="config.yaml"):
        
        # Загружаем конфигурацию
        self.config = load_config(config_path)
        
        self.vault_path = Path(vault_path)
        
        # Используем параметры из аргументов или из конфигурации
        self.output_dir = output_dir if output_dir else Path(self.vault_path) / self.config["output_dir"]
        self.merge_cosinus_threshold = merge_cosinus_threshold if merge_cosinus_threshold is not None else self.config["merge_cosinus_threshold"]
        self.path_threshold = path_threshold if path_threshold is not None else self.config["path_threshold"]
        self.cache_dir = self.config.get("cache_dir", ".cache")
        
        # Пути к файлам в кэш-директории
        self.similarity_path = os.path.join(self.cache_dir, "doc_similarities.pkl")
        self.connections_path = os.path.join(self.cache_dir, "wikilinks_connections.pkl")
        
        # Загружаем данные
        self.doc_similarities = self.load_data(self.similarity_path)
        self.wikilinks = self.load_data(self.connections_path)
        
        # Создаем директорию для инсайтов, если её нет
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Строим граф вики-ссылок
        self.wiki_graph = self.build_wikilinks_graph()
        
        # Загружаем данные о тегах из БД
        self.document_tags = self.load_document_tags()
        
        logging.info(f"InsightGenerator инициализирован с порогом сходства {self.merge_cosinus_threshold} и порогом пути {self.path_threshold}")
        logging.info(f"Директория для инсайтов: {self.output_dir}")

    def load_data(self, path):
        """Загружает данные из pickle-файла."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Загружены данные из {path}: {len(data)} записей")
            return data
        except Exception as e:
            logging.error(f"Ошибка загрузки данных из {path}: {e}")
            return {}
    
    def load_document_tags(self):
        """Загружает информацию о тегах документов из БД."""
        db_path = os.path.join(self.cache_dir, "chunks.db")
        doc_tags = {}
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Получаем все теги для каждого файла
            cursor.execute("""
                SELECT c.file_path, GROUP_CONCAT(t.name, ',') as tags
                FROM chunks c
                JOIN chunk_tags ct ON c.id = ct.chunk_id
                JOIN tags t ON ct.tag_id = t.id
                GROUP BY c.file_path
            """)
            
            for file_path, tags_str in cursor.fetchall():
                doc_tags[file_path] = set(tags_str.split(','))
            
            conn.close()
            logging.info(f"Загружена информация о тегах для {len(doc_tags)} документов")
            return doc_tags
        
        except Exception as e:
            logging.error(f"Ошибка при загрузке тегов из БД: {e}")
            return {}
    
    def build_wikilinks_graph(self):
        """Строит граф на основе вики-ссылок."""
        G = nx.DiGraph()
        
        # Добавляем все уникальные файлы как вершины
        all_files = set()
        for source, target in self.wikilinks.keys():
            all_files.add(source)
            all_files.add(target)
        
        for file_path in all_files:
            G.add_node(file_path, label=Path(file_path).stem)
        
        # Добавляем направленные ребра между связанными файлами
        for (source, target), data in self.wikilinks.items():
            G.add_edge(source, target, weight=data["count"])
        
        logging.info(f"Построен граф с {G.number_of_nodes()} вершинами и {G.number_of_edges()} рёбрами")
        return G
    
    def calculate_tag_similarity(self, file1, file2):
        """Вычисляет сходство между тегами двух документов."""
        tags1 = self.document_tags.get(file1, set())
        tags2 = self.document_tags.get(file2, set())
        
        if not tags1 or not tags2:
            return 0.0
        
        # Вычисляем сходство Жаккара
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_insights(self):
        """Находит потенциальные инсайты."""
        insights = []
        
        # Перебираем все пары документов с высоким семантическим сходством
        for (file1, file2), data in self.doc_similarities.items():
            # Проверяем порог косинусного сходства
            if data["score"] >= self.merge_cosinus_threshold:
                # Вычисляем сходство по тегам
                tag_similarity = self.calculate_tag_similarity(file1, file2)
                
                # Проверяем путь между документами в графе вики-ссылок
                try:
                    if file1 in self.wiki_graph.nodes and file2 in self.wiki_graph.nodes:
                        # Если оба документа в графе, проверяем путь
                        try:
                            path_length = nx.shortest_path_length(self.wiki_graph, file1, file2)
                            if path_length > self.path_threshold:
                                # Добавляем информацию о тегах к данным
                                insight_data = data.copy()
                                insight_data["tag_similarity"] = tag_similarity
                                insights.append((file1, file2, insight_data))
                        except nx.NetworkXNoPath:
                            # Если пути нет, считаем расстояние бесконечным
                            insight_data = data.copy()
                            insight_data["tag_similarity"] = tag_similarity
                            insights.append((file1, file2, insight_data))
                    else:
                        # Если хотя бы один документ не в графе, считаем связь потенциальным инсайтом
                        insight_data = data.copy()
                        insight_data["tag_similarity"] = tag_similarity
                        insights.append((file1, file2, insight_data))
                except Exception as e:
                    logging.error(f"Ошибка при анализе пути между {file1} и {file2}: {e}")
        
        # Сортируем инсайты по убыванию сходства
        insights.sort(key=lambda x: x[2]["score"] + x[2]["tag_similarity"], reverse=True)
        
        logging.info(f"Найдено {len(insights)} потенциальных инсайтов")
        return insights
    
    def extract_tags_from_content(self, content):
        """Извлекает теги из содержимого файла."""
        # Поиск тегов формата #tag (без пробела)
        hashtags = set(re.findall(r'#([a-zA-Z0-9_-]+)', content))
        
        # Поиск тегов формата # tag (с пробелом)
        header_tags = set(re.findall(r'#\s+([a-zA-Z0-9_-]+)', content))
        
        # Поиск тегов формата [[tag]] (Obsidian-ссылки)
        wikilinks = set(re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content))
        
        # Объединяем все найденные теги
        all_tags = hashtags.union(header_tags).union(wikilinks)
        return all_tags
    
    def create_insight_file(self, file1, file2, insight_data, api_client=None):
        """Создает файл с инсайтом, объединяющим содержимое двух документов."""
        try:
            # Проверяем существование файлов
            if not os.path.exists(file1) or not os.path.exists(file2):
                logging.error(f"Один из файлов не существует: {file1} или {file2}")
                return None
                
            # Получаем имена файлов без пути и расширения
            name1 = Path(file1).stem
            name2 = Path(file2).stem
            
            # Читаем содержимое файлов с обработкой кодировок
            try:
                with open(file1, 'r', encoding='utf-8') as f:
                    content1 = f.read()
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                encodings = ['latin-1', 'cp1251', 'windows-1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file1, 'r', encoding=encoding) as f:
                            content1 = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logging.error(f"Не удалось прочитать файл {file1}, перепробованы кодировки: utf-8, {encodings}")
                    return None
            
            try:
                with open(file2, 'r', encoding='utf-8') as f:
                    content2 = f.read()
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                encodings = ['latin-1', 'cp1251', 'windows-1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file2, 'r', encoding=encoding) as f:
                            content2 = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logging.error(f"Не удалось прочитать файл {file2}, перепробованы кодировки: utf-8, {encodings}")
                    return None
            
            # Проверяем, что файлы не пустые
            if not content1.strip() or not content2.strip():
                logging.warning(f"Пропускаем пустые файлы: {file1} или {file2}")
                return None
            
            # Создаем имя файла с инсайтом
            insight_filename = f"ai_merged_{name1}_{name2}.md"
            insight_path = os.path.join(self.output_dir, insight_filename)
            
            # Исправление проблемы 4: Проверяем и создаем директорию вывода
            os.makedirs(os.path.dirname(insight_path), exist_ok=True)
            
            # Содержимое инсайта
            if api_client:
                # Формируем запрос к API
                prompt = f"""Объедините информацию из двух файлов:
                
                Файл 1: {content1}
                
                Файл 2: {content2}
                
                Сформируйте краткое описание связей между ними."""
                
                # Перед отправкой запроса к API
                print(f"Отправка запроса в ИИ для файлов:\n- {Path(file1).stem}\n- {Path(file2).stem}")
                
                # Отправка запроса
                structured_response = api_client.query(prompt)
                
                # Показать ответ перед сохранением
                print("\nОтвет от ИИ:")
                print("-" * 60)
                
                # Показываем больше текста с сохранением форматирования
                preview_length = 1000
                response_preview = structured_response[:preview_length]
                
                # Добавляем индикатор, если текст был обрезан
                if len(structured_response) > preview_length:
                    response_preview += f"\n... [обрезано {len(structured_response) - preview_length} символов]"
                
                # Сохраняем форматирование и разделители
                print(response_preview)
                print("-" * 60)
                
                # Запросить подтверждение перед сохранением
                save_confirm = input("\nСохранить этот инсайт? (y/n): ").lower().strip()
                if save_confirm != 'y':
                    print("Инсайт не сохранен.")
                    return None
                
                # Формируем содержимое файла
                insight_content = f"[[ai-merged]], [[{name1}]], [[{name2}]]\n\n{structured_response}"
            else:
                # Находим вики-линки или теги для первого файла
                wikilinks1 = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content1)
                hashtags1 = re.findall(r'#([\w-]+)', content1, re.UNICODE)
                
                # Находим вики-линки или теги для второго файла
                wikilinks2 = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content2)
                hashtags2 = re.findall(r'#([\w-]+)', content2, re.UNICODE)
                
                # Определяем вики-линки для использования
                link1 = name1  # По умолчанию - имя файла
                if wikilinks1:
                    link1 = wikilinks1[0]  # Используем первый вики-линк
                elif hashtags1:
                    link1 = hashtags1[0]   # Или первый хэштег
                    
                link2 = name2  # По умолчанию - имя файла
                if wikilinks2:
                    link2 = wikilinks2[0]  # Используем первый вики-линк
                elif hashtags2:
                    link2 = hashtags2[0]   # Или первый хэштег
                
                # Формируем первую строку с тегами
                first_line = f"[[ai-merged]], [[{link1}]], [[{link2}]]"
                
                # Формируем содержимое файла с инсайтом - очень простой формат
                insight_content = f"{first_line}\n{content1}\n\n{content2}"
            
            # Записываем файл
            try:
                with open(insight_path, 'w', encoding='utf-8') as f:
                    f.write(insight_content)
                
                logging.info(f"Создан файл с инсайтом: {insight_path}")
                return insight_path
            except Exception as e:
                logging.error(f"Ошибка при записи файла инсайта: {e}")
                print(f"Error writing insight file: {e}")
                return None
            
        except Exception as e:
            logging.error(f"Ошибка при создании файла с инсайтом для {file1} и {file2}: {e}")
            return None
    
    def create_all_insights(self):
        """Создает файлы со всеми найденными инсайтами."""
        insights = self.find_insights()
        created_files = []
        
        for file1, file2, data in insights:
            insight_path = self.create_insight_file(file1, file2, data)
            if insight_path:
                created_files.append(insight_path)
        
        logging.info(f"Создано {len(created_files)} файлов с инсайтами")
        return created_files
    
    def interactive_insights(self):
        """Интерактивный режим создания инсайтов."""
        insights = self.find_insights()
        
        if not insights:
            print("Инсайты не найдены.")
            return
        
        print(f"\nНайдено {len(insights)} потенциальных инсайтов:")
        
        for i, (file1, file2, data) in enumerate(insights):
            name1 = Path(file1).stem
            name2 = Path(file2).stem
            print(f"{i+1}. {name1} <--> {name2} (сходство: {data['score']:.2f}, теги: {data.get('tag_similarity', 0):.2f})")
        
        while True:
            try:
                choice = input("\nВыберите номер инсайта для создания (или 'all' для всех, 'q' для выхода): ")
                
                if choice.lower() == 'q':
                    break
                
                if choice.lower() == 'all':
                    created_files = self.create_all_insights()
                    print(f"Создано {len(created_files)} файлов с инсайтами в папке {self.output_dir}")
                    break
                
                idx = int(choice) - 1
                if 0 <= idx < len(insights):
                    file1, file2, data = insights[idx]
                    insight_path = self.create_insight_file(file1, file2, data)
                    if insight_path:
                        print(f"Создан файл с инсайтом: {insight_path}")
                    else:
                        print("Ошибка при создании файла.")
                else:
                    print("Неверный номер. Попробуйте снова.")
            
            except ValueError:
                print("Некорректный ввод. Попробуйте снова.")
            except Exception as e:
                print(f"Ошибка: {e}")

def run_insight_finder(components):
    """Run the insight finder with API integration."""
    vault_path = components["vault_path"]
    config_path = components["config_path"]
    api_client = components["api_client"]
    
    # Исправление проблемы 1: Правильный вызов InsightGenerator без ссылки на модуль
    generator = InsightGenerator(vault_path=vault_path, config_path=config_path)
    
    # Проверка и создание выходной директории (Исправление проблемы 4)
    if not os.path.exists(generator.output_dir):
        os.makedirs(generator.output_dir, exist_ok=True)
        print(f"Created output directory: {generator.output_dir}")
    
    insights = generator.find_insights()
    
    if not insights:
        return "No insights found."
    
    print(f"\nFound {len(insights)} potential connections:")
    for i, (file1, file2, data) in enumerate(insights, 1):
        name1 = Path(file1).stem
        name2 = Path(file2).stem
        score = data.get('score', 0)
        tag_similarity = data.get('tag_similarity', 0)
        print(f"{i}. {name1} <--> {name2} (similarity: {score:.2f}, tag similarity: {tag_similarity:.2f})")
    
    choice = input("\nSelect insight to create (or 'all' for all, 'q' to quit): ")
    if choice.lower() == 'q':
        return "Operation cancelled."
    
    if choice.lower() == 'all':
        print("\n⚠️ ВНИМАНИЕ: Для каждого инсайта будет запрошено подтверждение сохранения.")
        confirm_all = input("Продолжить с просмотром всех инсайтов? (y/n): ").strip().lower()
        if confirm_all != 'y':
            return "Operation cancelled."
            
        created = []
        for i, (file1, file2, data) in enumerate(insights, 1):
            name1 = Path(file1).stem
            name2 = Path(file2).stem
            print(f"\nProcessing insight {i}/{len(insights)}: {name1} <--> {name2}")
            
            # Исправление проблемы 3: Добавляем обратную связь
            print("Sending request to AI...")
            
            path = generator.create_insight_file(file1, file2, data, api_client)
            if path:
                print(f"✅ Success! Created insight file: {path}")
                
                # Показываем краткое содержимое созданного файла
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print("\nPreview of insight content:")
                    print("-" * 40)
                    print(content[:200] + "..." if len(content) > 200 else content)
                    print("-" * 40)
                except Exception as e:
                    print(f"Could not read created file: {e}")
                
                created.append(path)
            else:
                print("❌ Failed to create insight file.")
        
        if created:
            return f"Created {len(created)} insight files in {generator.output_dir}"
        else:
            return "No insights were saved."
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(insights):
            file1, file2, data = insights[idx]
            name1 = Path(file1).stem
            name2 = Path(file2).stem
            
            # Исправление проблемы 3: Добавляем обратную связь
            print(f"Creating insight between {name1} and {name2}...")
            print("Sending request to AI...")
            
            path = generator.create_insight_file(file1, file2, data, api_client)
            if path:
                print(f"✅ Success! Created insight file: {path}")
                
                # Показываем содержимое созданного файла
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print("\nInsight content:")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)
                except Exception as e:
                    print(f"Could not read created file: {e}")
                
                return f"Created insight file: {path}"
            else:
                print("❌ Failed to create insight file.")
                return "Insight file was not saved."
        else:
            return "Invalid selection."
    except ValueError:
        return "Invalid input."

def main():
    parser = argparse.ArgumentParser(description="Генерация инсайтов на основе анализа графа документов")
    parser.add_argument("--vault", required=True, help="Путь к хранилищу документов (Obsidian vault)")
    parser.add_argument("--output", default=None, help="Директория для сохранения инсайтов")
    parser.add_argument("--threshold", type=float, default=None, help="Порог косинусного сходства")
    parser.add_argument("--path", type=int, default=None, help="Порог расстояния между документами")
    parser.add_argument("--config", default="config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--process", action="store_true", help="Перестроить векторный индекс перед поиском инсайтов")
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Если нужно перестроить индекс
    if args.process:
        from obsidian_vault_processor import VaultProcessor
        
        logging.info("Пересоздание векторного индекса...")
        processor = VaultProcessor(config_path=args.config)
        processor.process_vault(args.vault)
        
        logging.info("Индекс перестроен. Вычисляем сходства между документами...")
        
        # Вычисляем семантическое сходство между документами
        threshold = args.threshold if args.threshold is not None else config.get("merge_cosinus_threshold", 0.75)
        processor.save_semantic_similarities(threshold=threshold, top_k=3)
        
        # Получаем связи на основе вики-ссылок
        processor.save_wikilinks_connections()
        
        logging.info("Индекс и связи перестроены.")
    
    # Инициализируем генератор инсайтов
    generator = InsightGenerator(
        vault_path=args.vault,
        output_dir=args.output,
        merge_cosinus_threshold=args.threshold,
        path_threshold=args.path,
        config_path=args.config
    )
    
    print("\n===== AI Insight Generator =====")
    print(f"Хранилище: {args.vault}")
    print(f"Директория для инсайтов: {generator.output_dir}")
    print(f"Порог косинусного сходства: {generator.merge_cosinus_threshold}")
    print(f"Порог пути: {generator.path_threshold}")
    
    choice = input("\nНайти инсайты? (y/n): ")
    if choice.lower() == 'y':
        generator.interactive_insights()
    else:
        print("Выход.")

if __name__ == "__main__":
    main()