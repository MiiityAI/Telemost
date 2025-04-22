#!/usr/bin/env python3
import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
import pickle
import argparse
import yaml
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Функция для загрузки конфигурации
def load_config(config_path="config.yaml"):
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация загружена из {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        # Возвращаем конфигурацию по умолчанию
        return {
            "semantic_threshold": 0.5,
            "path_threshold": 6,
            "output_dir": "_ai_merged",
            "use_gpu": False,
            "cache_dir": ".cache",
            "db_path": "chunks.db",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "timeout": 6000,
            "max_workers": 12,
            "batch_size": 100,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "language_specific_normalization": False
        }

# Класс для работы с чанками текста
class ChunkMaster:
    def __init__(self, db_path=None, chunk_size=None, chunk_overlap=None, config_path="config.yaml"):
        # Загружаем конфигурацию
        self.config = load_config(config_path)
        
        # Используем параметры из конфигурации или переданные явно
        self.db_path = db_path or self.config["db_path"]
        self.chunk_size = chunk_size or self.config["chunk_size"]
        self.chunk_overlap = chunk_overlap or self.config["chunk_overlap"]
        
        # Инициализируем базу данных
        self._init_database()
        
        # Текущий обрабатываемый файл (для извлечения тегов)
        self.current_file = None

    def _init_database(self):
        """Инициализирует базу данных, создавая необходимые таблицы."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицы, если они не существуют
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            file_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            tag_id INTEGER,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        )
        """)
        
        conn.commit()
        conn.close()
        logging.info("База данных инициализирована")

    def split_text(self, text, chunk_size=None):
        """Разбивает текст на чанки с учетом перекрытия."""
        chunk_size = chunk_size or self.chunk_size
        overlap = self.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Добавляем чанк
            chunks.append(text[start:min(end, len(text))])
            
            # Двигаемся вперед с учетом перекрытия
            start = end - overlap
        
        logging.info(f"Текст разбит на {len(chunks)} чанков.")
        return chunks

    def save_chunks(self, chunks, file_path):
        """Сохраняет чанки в базу данных."""
        if not chunks:
            logging.warning(f"Нет чанков для сохранения из {file_path}")
            return

        # Преобразуем file_path в строку
        file_path_str = str(file_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Извлекаем теги из текста (для всего файла)
            all_file_text = " ".join(chunks)
            tags = self.extract_tags(all_file_text)
            
            # Сохраняем теги, если они есть
            tag_ids = []
            for tag in tags:
                try:
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    tag_ids.append(tag_id)
                except Exception as e:
                    logging.error(f"Ошибка при сохранении тега {tag}: {e}")
            
            # Сохраняем чанки
            for chunk in chunks:
                try:
                    cursor.execute(
                        "INSERT INTO chunks (content, file_path) VALUES (?, ?)",
                        (chunk, file_path_str)  # Используем строковую версию пути
                    )
                    chunk_id = cursor.lastrowid
                    
                    # Связываем чанк с тегами
                    for tag_id in tag_ids:
                        cursor.execute(
                            "INSERT INTO chunk_tags (chunk_id, tag_id) VALUES (?, ?)",
                            (chunk_id, tag_id)
                        )
                except Exception as e:
                    logging.error(f"Ошибка при сохранении чанка для {file_path}: {e}")
            
            conn.commit()
            logging.info(f"Сохранено {len(chunks)} чанков и {len(tags)} тегов для {file_path}")
        except Exception as e:
            conn.rollback()
            logging.error(f"Ошибка при сохранении данных: {e}")
        finally:
            conn.close()

    def extract_tags(self, text):
        """Извлекает теги всех поддерживаемых форматов."""
        tags = set()
        
        # Проверка на наличие YAML-метаданных
        yaml_match = re.match(r'^---\n(.*?)\n---', text, re.DOTALL)
        if yaml_match:
            try:
                import yaml
                yaml_text = yaml_match.group(1)
                yaml_data = yaml.safe_load(yaml_text)
                if yaml_data and 'tags' in yaml_data:
                    if isinstance(yaml_data['tags'], list):
                        tags.update(yaml_data['tags'])
                    elif isinstance(yaml_data['tags'], str):
                        tags.add(yaml_data['tags'])
            except Exception as e:
                logging.warning(f"Ошибка при извлечении тегов из YAML: {e}")
        
        # Поиск тегов формата #tag (без пробела)
        hashtags = re.findall(r'#([\w-]+)', text, re.UNICODE)
        tags.update(hashtags)
        
        # Поиск тегов формата # tag (с пробелом)
        header_tags = re.findall(r'#\s+([\w-]+)', text, re.UNICODE)
        tags.update(header_tags)
        
        # Поиск тегов формата [[tag]] (Obsidian-ссылки)
        wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', text)
        tags.update(wikilinks)
        
        # Добавляем искусственные теги по названию файла, если нет других тегов
        if not tags and hasattr(self, 'current_file'):
            file_name = os.path.basename(self.current_file)
            if file_name.endswith('.md'):
                file_name = file_name[:-3]  # Убираем расширение .md
            tags.add(file_name.replace(' ', '_'))
        
        logging.info(f"Найдено тегов: {list(tags)}")
        return list(tags)

# Класс для работы с векторами
class VectorMaster:
    def __init__(self, use_gpu=None, cache_dir=None, config_path="config.yaml"):
        # Загружаем конфигурацию
        self.config = load_config(config_path)
        
        # Используем параметры из конфигурации или переданные явно
        self.use_gpu = use_gpu if use_gpu is not None else self.config["use_gpu"]
        self.cache_dir = cache_dir or self.config["cache_dir"]
        
        # Настройка GPU
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Загружаем модель для создания эмбеддингов
        self.model = self._load_model()
        
        # Инициализируем пустой индекс
        self.index = None
        self.index_to_text = {}
        self.index_to_file = {}
        self.file_embeddings = {}
        self.wikilinks_connections = {}
        
        # Инициализируем кэш векторов
        self.vector_cache = {}
        
        # Инициализируем индекс, если директория кэша существует
        self.initialize_index()

    def _load_model(self):
        """Загружает модель для создания эмбеддингов."""
        try:
            # Многоязычная модель вместо "all-MiniLM-L6-v2"
            model_name = self.config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
            return SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            raise

    def initialize_index(self):
        """Инициализирует FAISS индекс."""
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        logging.info(f"FAISS индекс (косинусное сходство) инициализирован с размерностью {self.index.d}")

    def generate_embeddings(self, chunks):
        logging.info(f"Начинаю генерацию эмбеддингов для {len(chunks)} чанков.")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        logging.info(f"Сгенерировано {len(embeddings)} эмбеддингов с размерностью {embeddings.shape}.")
        return embeddings

    def extract_wikilinks(self, text, source_file):
        """Извлекает вики-ссылки из текста и сохраняет информацию о связях."""
        wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', text, re.UNICODE)
        
        if wikilinks and source_file not in self.wikilinks_connections:
            self.wikilinks_connections[source_file] = set(wikilinks)
        elif wikilinks:
            self.wikilinks_connections[source_file].update(wikilinks)
            
        return wikilinks
    
    def add_vectors(self, embeddings, chunks, file_path):
        """Добавляет векторы в индекс и сохраняет связь с файлом."""
        # Преобразуем file_path в строку
        file_path_str = str(file_path)
        
        # Добавляем в индекс
        self.index.add(embeddings)
        
        # Сохраняем связь между файлом и его эмбеддингами
        if file_path_str not in self.file_embeddings:
            self.file_embeddings[file_path_str] = []
        
        # Сохраняем связь с файлом и соответствующие chunks для дальнейшего использования
        start_idx = self.index.ntotal - len(embeddings)
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            idx = start_idx + i
            self.file_embeddings[file_path_str].append((idx, embedding))
            
            # Извлекаем вики-ссылки из чанка
            self.extract_wikilinks(chunk, file_path_str)
        
        # Сохраняем связь между индексом и текстом/файлом
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.index_to_text[idx] = chunk
            self.index_to_file[idx] = file_path_str
        
        logging.info(f"Добавлено {len(embeddings)} векторов для файла {file_path}")
        return embeddings

    def compute_chunk_similarities(self, threshold=0.7, top_k=3):
        """
        Вычисляет семантическое сходство между документами на основе
        наиболее похожих чанков (а не усреднения векторов).
        
        Args:
            threshold: минимальный порог сходства для создания связи
            top_k: количество наиболее похожих чанков для сравнения
            
        Returns:
            dict: {(file1, file2): {"score": float, "chunks": [(chunk1, chunk2, score), ...]}}
        """
        if not self.file_embeddings:
            logging.warning("Нет данных о документах и их эмбеддингах")
            return {}
            
        # Создаем список всех файлов
        files = list(self.file_embeddings.keys())
        logging.info(f"Начинаю вычисление семантического сходства между {len(files)} документами")
        
        # Словарь для хранения результатов сходства между документами
        similarities = {}
        
        # Для каждой пары документов
        for i in range(len(files)):
            for j in range(i+1, len(files)):
                file1, file2 = files[i], files[j]
                
                # Собираем чанки и их эмбеддинги для обоих файлов
                chunks1, vectors1 = zip(*self.file_embeddings[file1])
                chunks2, vectors2 = zip(*self.file_embeddings[file2])
                
                # Преобразуем списки в numpy массивы
                vectors1 = np.array(vectors1)
                vectors2 = np.array(vectors2)
                
                # Вычисляем матрицу сходства между всеми парами чанков
                # (используем косинусное сходство)
                similarity_matrix = np.matmul(vectors1, vectors2.T)
                
                # Нормализуем векторы для получения косинусного сходства
                norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
                norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
                similarity_matrix = similarity_matrix / (norms1 @ norms2.T)
                
                # Находим top_k наиболее похожих пар чанков
                top_pairs = []
                flat_indices = np.argsort(similarity_matrix.flatten())[-top_k:]
                for flat_idx in flat_indices:
                    i_idx, j_idx = np.unravel_index(flat_idx, similarity_matrix.shape)
                    chunk1, chunk2 = chunks1[i_idx], chunks2[j_idx]
                    score = similarity_matrix[i_idx, j_idx]
                    
                    # Добавляем пару чанков, только если сходство выше порога
                    if score >= threshold:
                        top_pairs.append((chunk1, chunk2, float(score)))
                
                # Если есть похожие пары чанков, сохраняем информацию о сходстве документов
                if top_pairs:
                    # Вычисляем общий скор как среднее по top_k парам
                    avg_score = sum(score for _, _, score in top_pairs) / len(top_pairs)
                    
                    similarities[(file1, file2)] = {
                        "score": avg_score,
                        "chunks": top_pairs
                    }
        
        # Выводим статистику о найденных сходствах
        logging.info(f"Найдено {len(similarities)} семантически близких пар документов")
        for (file1, file2), data in list(similarities.items())[:5]:  # выводим примеры первых 5 пар
            logging.info(f"Сходство между {os.path.basename(file1)} и {os.path.basename(file2)}: {data['score']:.3f}")
            
        return similarities

    def get_wikilinks_connections(self):
        """
        Создает словарь связей между документами на основе вики-ссылок.
        
        Returns:
            dict: словарь связей в формате {(источник, цель): {"count": число ссылок}}
        """
        # Преобразуем имена ссылок в пути к файлам
        filename_to_path = {}
        for file_path in self.file_embeddings.keys():
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            filename_to_path[base_name] = file_path
        
        # Создаем словарь связей
        connections = {}
        link_count = 0
        
        for source_file, links in self.wikilinks_connections.items():
            for link in links:
                if link in filename_to_path:
                    target_file = filename_to_path[link]
                    connection_key = (source_file, target_file)
                    
                    if connection_key not in connections:
                        connections[connection_key] = {"count": 1}
                    else:
                        connections[connection_key]["count"] += 1
                    
                    link_count += 1
        
        logging.info(f"Найдено {link_count} вики-ссылок между {len(connections)} парами документов")
        return connections

    def save_vector_cache(self):
        """Сохраняет кэш векторных представлений."""
        try:
            # Создаем директорию кэша, если её нет
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Сохраняем все необходимые данные в кэш
            cache_data = {
                'index_to_text': self.index_to_text,
                'index_to_file': self.index_to_file,
                'file_embeddings': self.file_embeddings,
                'wikilinks_connections': self.wikilinks_connections
            }
            
            # Сохранение кэша
            cache_path = os.path.join(self.cache_dir, "vector_cache.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Если используется FAISS индекс, сохраняем его отдельно
            if self.index is not None:
                index_path = os.path.join(self.cache_dir, "faiss_index.bin")
                faiss.write_index(self.index, index_path)
            
            logging.info(f"Векторный кэш сохранен в {self.cache_dir}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении векторного кэша: {e}")

    def load_vector_cache(self):
        """Загружает кэш векторных представлений."""
        try:
            cache_path = os.path.join(self.cache_dir, "vector_cache.pkl")
            index_path = os.path.join(self.cache_dir, "faiss_index.bin")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.index_to_text = cache_data['index_to_text']
                self.index_to_file = cache_data['index_to_file']
                self.file_embeddings = cache_data['file_embeddings']
                self.wikilinks_connections = cache_data['wikilinks_connections']
                
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
                
                logging.info(f"Векторный кэш загружен из {self.cache_dir}")
                return True
        except Exception as e:
            logging.error(f"Ошибка при загрузке векторного кэша: {e}")
        return False

# Класс для обработки Vault
class VaultProcessor:
    def __init__(self, batch_size=None, max_workers=None, use_gpu=None, config_path="config.yaml"):
        # Загружаем конфигурацию
        self.config = load_config(config_path)
        
        # Используем параметры из конфигурации или переданные явно
        self.batch_size = batch_size or self.config["batch_size"]
        self.max_workers = max_workers or self.config["max_workers"]
        self.use_gpu = use_gpu if use_gpu is not None else self.config["use_gpu"]
        
        # Создаем директорию кэша, если её нет
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        
        # Перемещаем базу данных в директорию кэша
        db_path = os.path.join(self.config["cache_dir"], os.path.basename(self.config["db_path"]))
        
        self.chunk_master = ChunkMaster(db_path=db_path, config_path=config_path)
        self.vector_master = VectorMaster(use_gpu=self.use_gpu, config_path=config_path)

    def process_file(self, file_path):
        """Обрабатывает отдельный файл."""
        try:
            # Преобразуем file_path в строку для открытия файла
            file_path_str = str(file_path)
            
            # Пропускаем файлы, которые не являются текстовыми
            if not file_path_str.endswith(('.md', '.txt')):
                return
                
            # Расширяем список кодировок для попыток открытия файла
            encodings = ['utf-8', 'latin-1', 'cp1251', 'windows-1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path_str, "r", encoding=encoding) as f:
                        text = f.read()
                        break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                logging.error(f"Не удалось прочитать файл {file_path_str}, перепробованы кодировки: {encodings}")
                return
                
            # Разбиваем текст на чанки
            chunks = self.chunk_master.split_text(text)
            
            # Устанавливаем текущий файл для извлечения тегов
            self.chunk_master.current_file = file_path_str
            
            # Сохраняем чанки в базу данных
            self.chunk_master.save_chunks(chunks, file_path_str)
            
            # Генерируем эмбеддинги для чанков
            embeddings = self.vector_master.generate_embeddings(chunks)
            
            # Добавляем векторы в индекс
            self.vector_master.add_vectors(embeddings, chunks, file_path_str)
            
            logging.info(f"Обработан файл: {file_path}")
        except Exception as e:
            logging.error(f"Ошибка при обработке {file_path}: {e}")

    def process_vault(self, vault_path, file_extensions=None):
        """Обрабатывает все файлы в хранилище."""
        if file_extensions is None:
            file_extensions = [".md", ".txt"]
        
        vault_path = Path(vault_path)
        logging.info(f"Обработка хранилища {vault_path} для расширений {file_extensions}")
        
        # 1. Удаляем файл базы данных и принудительно пересоздаем его
        db_path = Path(self.chunk_master.db_path)
        if db_path.exists():
            try:
                db_path.unlink()
                logging.info(f"Старая база данных удалена: {db_path}")
            except Exception as e:
                logging.error(f"Ошибка при удалении базы данных: {e}")
        
        # 2. Повторно инициализируем базу данных
        self.chunk_master._init_database()
        
        # 3. Очищаем кэш вектормастера и переинициализируем индекс
        if hasattr(self.vector_master, 'index') and self.vector_master.index is not None:
            self.vector_master.index = None
        self.vector_master.initialize_index()
        self.vector_master.index_to_text = {}
        self.vector_master.index_to_file = {}
        self.vector_master.file_embeddings = {}
        self.vector_master.wikilinks_connections = {}
        
        # Найдем все файлы с указанными расширениями
        all_files = []
        for extension in file_extensions:
            all_files.extend(vault_path.glob(f"**/*{extension}"))
        
        # Отфильтруем файлы в директории _ai_merged, если такая существует
        all_files = [f for f in all_files if "_ai_merged" not in str(f)]
        
        # Количество найденных файлов
        total_files = len(all_files)
        logging.info(f"Найдено {total_files} файлов (исключая _ai_merged)")
        
        if total_files == 0:
            logging.warning(f"Файлы с расширениями {file_extensions} не найдены в {vault_path}")
            return
        
        # Определяем размер пакета для обработки
        batch_size = min(self.batch_size, total_files)
        
        # Обрабатываем файлы пакетами
        for i in range(0, total_files, batch_size):
            batch = all_files[i:i + batch_size]
            
            # Используем ThreadPoolExecutor для параллельной обработки
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(self.process_file, batch)
            
            logging.info(f"Обработано {min(i + batch_size, total_files)}/{total_files} файлов")
        
        # Сохраняем векторный кэш
        self.vector_master.save_vector_cache()
        
        # Вычисляем и сохраняем семантические сходства
        self.save_semantic_similarities()
        
        # Получаем и сохраняем информацию о вики-ссылках
        self.save_wikilinks_connections()
        
        logging.info(f"Обработка хранилища завершена, обработано {total_files} файлов")

    def save_semantic_similarities(self, threshold=0.7, top_k=3):
        """Вычисляет и сохраняет семантические сходства между документами."""
        try:
            # Получаем семантические сходства
            doc_similarities = self.vector_master.compute_chunk_similarities(threshold=threshold, top_k=top_k)
            
            # Сохраняем результаты в кэш-директорию
            similarities_path = os.path.join(self.config["cache_dir"], "doc_similarities.pkl")
            with open(similarities_path, "wb") as f:
                pickle.dump(doc_similarities, f)
            
            logging.info(f"Семантические сходства сохранены в {similarities_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении семантических сходств: {e}")

    def save_wikilinks_connections(self):
        """Получает и сохраняет информацию о вики-ссылках."""
        try:
            # Получаем информацию о вики-ссылках
            wikilinks_connections = self.vector_master.get_wikilinks_connections()
            
            # Сохраняем результаты в кэш-директорию
            connections_path = os.path.join(self.config["cache_dir"], "wikilinks_connections.pkl")
            with open(connections_path, "wb") as f:
                pickle.dump(wikilinks_connections, f)
            
            logging.info(f"Информация о вики-ссылках сохранена в {connections_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении информации о вики-ссылках: {e}")


def main():
    """Основная функция скрипта."""
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Obsidian Vault Processor")
    
    parser.add_argument("-v", "--vault", help="Путь к Obsidian Vault для обработки")
    parser.add_argument("-c", "--config", default="config.yaml", help="Путь к файлу конфигурации (по умолчанию: config.yaml)")
    parser.add_argument("--use-gpu", action="store_true", help="Использовать GPU для обработки (если доступен)")
    
    args = parser.parse_args()
    
    # Если указан путь к хранилищу, обрабатываем его
    if args.vault:
        start_time = time.time()
        
        # Инициализируем процессор
        processor = VaultProcessor(
            use_gpu=args.use_gpu,
            config_path=args.config
        )
        
        # Обрабатываем хранилище
        processor.process_vault(args.vault)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Обработка завершена за {elapsed_time:.2f} секунд")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
