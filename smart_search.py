#!/usr/bin/env python3
import os
import argparse
import logging
import faiss
import numpy as np
import pickle
import re
from pathlib import Path
import yaml
from sentence_transformers import SentenceTransformer

# Настройка логирования
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'smart_search.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Отключаем логи SentenceTransformer
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

def load_config(config_path):
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
            "cosinus_index": 0.5,
            "cache_dir": ".cache",
            "top_results": 13,
            "use_gpu": False
        }

class SmartSearcher:
    def __init__(self, config_path, vault_path=None):
        """Инициализация поисковика."""
        self.config = load_config(config_path)
        self.vault_path = vault_path
        self.cache_dir = self.config.get("cache_dir", ".cache")
        self.semantic_threshold = self.config.get("semantic_threshold", 0.5)
        self.cosinus_index = self.config.get("cosinus_index", 0.5)
        self.top_results = self.config.get("top_results", 13)
        self.use_gpu = self.config.get("use_gpu", False)
        
        # Настройка GPU
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Загружаем модель для создания эмбеддингов
        self.model = self._load_model()
        
        # Загружаем кэш и индекс
        self._load_cache()
        
    def _load_model(self):
        """Загружает модель для создания эмбеддингов."""
        try:
            model_name = self.config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
            model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Модель {model_name} загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def _load_cache(self):
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
                self.wikilinks_connections = cache_data.get('wikilinks_connections', {})
                
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
                    logger.info(f"Загружен FAISS индекс с {self.index.ntotal} векторами и размерностью {self.index.d}")
                else:
                    logger.error(f"Индекс не найден: {index_path}")
                    raise FileNotFoundError(f"Индекс не найден: {index_path}")
                
                logger.info(f"Векторный кэш загружен из {self.cache_dir}")
                return True
            else:
                logger.error(f"Кэш не найден: {cache_path}")
                raise FileNotFoundError(f"Кэш не найден: {cache_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторного кэша: {e}")
            raise
    
    def keyword_search(self, query):
        """
        Выполняет поиск чанков, содержащих ключевые слова из запроса.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            List: Список найденных чанков с их метаданными
        """
        logger.info(f"Выполняется поиск по ключевым словам для запроса: '{query}'")
        
        # Извлекаем ключевые слова (длиннее 3 символов)
        keywords = [word.lower() for word in re.findall(r'\b\w{3,}\b', query)]
        if not keywords:
            logger.warning(f"Не удалось извлечь ключевые слова из запроса: '{query}'")
            return []
            
        logger.info(f"Извлеченные ключевые слова: {keywords}")
        
        # Результаты поиска по ключевым словам
        keyword_results = []
        
        # Ищем ключевые слова в чанках
        for idx, chunk_text in self.index_to_text.items():
            chunk_text_lower = chunk_text.lower()
            
            # Считаем, сколько ключевых слов найдено в чанке
            matches = sum(1 for keyword in keywords if keyword in chunk_text_lower)
            
            # Если найдено хотя бы одно ключевое слово, добавляем в результаты
            if matches > 0:
                file_path = self.index_to_file.get(idx, "Unknown")
                
                # Рассчитываем релевантность как отношение найденных ключевых слов к общему количеству
                relevance = matches / len(keywords)
                
                keyword_results.append({
                    "chunk_id": idx,
                    "text": chunk_text,
                    "relevance": relevance,
                    "file_path": file_path,
                    "matches": matches
                })
        
        # Сортируем по убыванию релевантности
        keyword_results.sort(key=lambda x: x["relevance"], reverse=True)
        
        logger.info(f"Найдено {len(keyword_results)} чанков, содержащих ключевые слова")
        return keyword_results
    
    def semantic_search(self, query, reference_chunks=None):
        """
        Выполняет поиск по семантической близости.
        
        Args:
            query: Поисковый запрос
            reference_chunks: Список чанков для уточнения поиска
            
        Returns:
            List: Список найденных чанков с их метаданными
        """
        logger.info(f"Выполняется семантический поиск для запроса: '{query}' (threshold {self.cosinus_index})")
        
        try:
            # Генерируем эмбеддинг запроса
            query_embedding = self.model.encode([query])[0]
            
            # Нормализуем запрос для косинусного сходства
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Если есть референсные чанки, учитываем их семантику
            if reference_chunks and len(reference_chunks) > 0:
                # Выбираем тексты для топ-3 найденных чанков
                reference_texts = [chunk["text"] for chunk in reference_chunks[:3]]
                
                # Генерируем эмбеддинги для референсных текстов
                reference_embeddings = self.model.encode(reference_texts)
                
                # Нормализуем
                for i in range(len(reference_embeddings)):
                    reference_embeddings[i] = reference_embeddings[i] / np.linalg.norm(reference_embeddings[i])
                
                # Объединяем запрос с семантикой референсных текстов
                combined_embedding = query_embedding.flatten() + np.mean(reference_embeddings, axis=0)
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                combined_embedding = combined_embedding.reshape(1, -1).astype('float32')
                
                # Используем комбинированный эмбеддинг для поиска
                search_embedding = combined_embedding
            else:
                search_embedding = query_embedding
            
            # Поиск с помощью FAISS
            k = min(100, self.index.ntotal)
            distances, indices = self.index.search(search_embedding, k)
            
            # Создаем результаты
            semantic_results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx not in self.index_to_text:
                    continue
                
                # В IndexFlatIP (Inner Product) большие значения = лучшее сходство
                similarity = float(distances[0][i])
                
                # Проверяем порог косинусного сходства
                if similarity >= self.cosinus_index:
                    file_path = self.index_to_file.get(idx, "Unknown")
                    chunk_text = self.index_to_text.get(idx, "")
                    
                    # Исключаем чанки, найденные по ключевым словам
                    if reference_chunks and any(chunk["chunk_id"] == idx for chunk in reference_chunks):
                        continue
                    
                    semantic_results.append({
                        "chunk_id": idx,
                        "text": chunk_text,
                        "similarity": similarity,
                        "file_path": file_path
                    })
            
            # Сортируем по убыванию сходства
            semantic_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Найдено {len(semantic_results)} семантически близких чанков")
            return semantic_results
        
        except Exception as e:
            logger.error(f"Ошибка при семантическом поиске: {e}")
            return []
    
    def search(self, query):
        """
        Выполняет двухэтапный поиск: сначала по ключевым словам, затем по семантической близости.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            List: Список найденных чанков с их метаданными
        """
        logger.info(f"Выполняется двухэтапный поиск для запроса: '{query}'")
        
        # Шаг 1: Поиск по ключевым словам (ограничен параметром top_results)
        keyword_results = self.keyword_search(query)
        
        # Ограничиваем количество результатов по ключевым словам
        top_keyword_results = keyword_results[:self.top_results]
        
        logger.info(f"Отобрано топ-{len(top_keyword_results)} результатов по ключевым словам из {len(keyword_results)}")
        
        # Шаг 2: Поиск по семантической близости с учетом найденных чанков
        # Семантический поиск основывается на top_keyword_results
        semantic_results = self.semantic_search(query, top_keyword_results)
        
        # Объединяем результаты
        combined_results = []
        
        # Добавляем результаты поиска по ключевым словам
        for result in top_keyword_results:
            combined_results.append({
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "score": result["relevance"] * 1.5,  # Повышаем вес результатов по ключевым словам
                "file_path": result["file_path"],
                "match_type": "keyword"
            })
        
        # Добавляем результаты семантического поиска (без ограничения числом)
        for result in semantic_results:
            combined_results.append({
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "score": result["similarity"],
                "file_path": result["file_path"],
                "match_type": "semantic"
            })
        
        # Сортируем объединенные результаты
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Итоговые результаты: {len(combined_results)} чанков (из них {len(top_keyword_results)} по ключевым словам)")
        return combined_results

    def format_search_results(self, query, results):
        """
        Форматирует результаты поиска в требуемый вид.
        
        Args:
            query: Поисковый запрос
            results: Результаты поиска
            
        Returns:
            str: Форматированная строка с результатами поиска
        """
        output = f"Query: {query}\nContext:\n"
        
        for i, result in enumerate(results, 1):
            output += f"Chunk{i}\n{result['text']}\n\n"
        
        # Примерная оценка количества токенов (4 символа ~ 1 токен)
        token_count = len(output) // 4
        
        output += f"Total tokens: {token_count}"
        
        return output

def main():
    """Основная функция скрипта."""
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Smart Search for Obsidian Vault")
    
    parser.add_argument("-query", "--query", required=True, help="Поисковый запрос")
    parser.add_argument("-c", "--config", default="config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("-v", "--vault", help="Путь к Obsidian Vault")
    parser.add_argument("-n", "--num-results", type=int, help="Количество результатов по ключевым словам")
    parser.add_argument("-t", "--threshold", type=float, help="Порог косинусного сходства для семантического поиска")
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        config = load_config(args.config)
        
        # Если указаны аргументы командной строки, они имеют приоритет
        if args.num_results is not None:
            config["top_results"] = args.num_results
        
        if args.threshold is not None:
            config["cosinus_index"] = args.threshold
        
        # Инициализируем поисковик
        searcher = SmartSearcher(args.config, args.vault)
        
        # Обновляем параметры из командной строки
        if args.num_results is not None:
            searcher.top_results = args.num_results
            
        if args.threshold is not None:
            searcher.cosinus_index = args.threshold
        
        # Выполняем поиск
        results = searcher.search(args.query)
        
        if not results:
            logger.warning(f"Не найдено результатов для запроса: '{args.query}'")
            print(f"Query: {args.query}\nContext:\nNo results found.\n\nTotal tokens: {len(args.query) // 4 + 10}")
            return
        
        # Форматируем результаты
        output = searcher.format_search_results(args.query, results)
        
        # Выводим результаты в stdout
        print(output)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()