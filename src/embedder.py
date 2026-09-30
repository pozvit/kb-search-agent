import hashlib
import os
import re
from typing import Any, Dict, List

import numpy as np

# Реальная модель импортируется лениво: в offline-режиме (RAG_FAKE_EMBEDDINGS=1)
# sentence-transformers и torch не нужны вообще.
FAKE_EMBEDDINGS = os.environ.get("RAG_FAKE_EMBEDDINGS") == "1"

# Телеметрию библиотек Hugging Face отключаем до их импорта.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# RAG_EMBEDDINGS_ALLOW_DOWNLOAD=0 — модель берётся только из локального кэша;
# если её там нет, загрузка завершается ошибкой, обращений к Hugging Face нет вообще.
ALLOW_DOWNLOAD = os.environ.get("RAG_EMBEDDINGS_ALLOW_DOWNLOAD", "1") != "0"

if not FAKE_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer
else:  # pragma: no cover - служебный режим
    SentenceTransformer = None


class HashingModel:
    """
    Детерминированный заменитель модели для смоук-теста пайплайна без сети:
    hashing trick по словам + L2-нормировка. Качество поиска у него слабое,
    он нужен ТОЛЬКО чтобы проверить связку chunk -> vector -> Qdrant -> retrieval
    там, где нельзя скачать веса. Для реальной работы не использовать.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False,
               convert_to_numpy: bool = True, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimensions, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


def _build_model(model_name: str):
    if FAKE_EMBEDDINGS:
        print(f"ВНИМАНИЕ: RAG_FAKE_EMBEDDINGS=1 — вместо '{model_name}' используется "
              f"HashingModel (только для смоук-теста, качество поиска нерелевантно).")
        return HashingModel()
    return _load_sentence_transformer(model_name)


def _load_sentence_transformer(model_name: str):
    """
    Загрузка по имени модели (SentenceTransformer("intfloat/...")) при каждом запуске
    обращается к Hugging Face и проверяет, не вышла ли новая версия файлов, даже если
    модель уже в кэше. Тексты запросов и документов при этом не отправляются —
    эмбеддинги считаются локально, — но сам запрос (имя модели, версии библиотек, IP) уходит.

    Поэтому сначала пробуем только локальный кэш (local_files_only=True, без сети).
    В сеть идём, только если модели в кэше нет (первый запуск) и загрузка разрешена.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        # очень старые версии sentence-transformers без параметра local_files_only
        return SentenceTransformer(model_name)
    except (OSError, ValueError) as exc:   # LocalEntryNotFoundError наследует оба
        if not ALLOW_DOWNLOAD:
            raise OSError(
                f"Модель '{model_name}' не найдена в локальном кэше Hugging Face, а загрузка запрещена "
                f"(RAG_EMBEDDINGS_ALLOW_DOWNLOAD=0). Скачайте модель один раз без этой переменной."
            ) from exc
        print(f"Модель '{model_name}' не найдена в локальном кэше — однократная загрузка с Hugging Face.")
        return SentenceTransformer(model_name)


def get_model_dimension(model) -> int:
    """
    Размерность модели. В разных версиях sentence-transformers метод называется
    по-разному (get_sentence_embedding_dimension / get_embedding_dimension),
    поэтому проверяем оба и в крайнем случае считаем по тестовому вектору.
    """
    for attr in ("get_sentence_embedding_dimension", "get_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            dim = getter()
            if dim:
                return int(dim)
    return int(len(model.encode("dimension probe", convert_to_numpy=True)))


class Embedder:
    """
    Обёртка над sentence-transformers.

    passage_prefix нужен моделям семейства E5 (intfloat/multilingual-e5-*),
    которые обучены на префиксах 'passage: ' для документов и 'query: ' для запросов.
    Для моделей без префиксов оставьте пустую строку.
    """

    def __init__(self, model_name: str, batch_size: int = 32, max_tokens: int = None,
                 passage_prefix: str = "", normalize: bool = True):
        self.model = _build_model(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.passage_prefix = passage_prefix or ""
        self.normalize = normalize
        self.dimensions = get_model_dimension(self.model)

    def embed_texts(self, texts: List[str], prefix: str = None) -> List[List[float]]:
        prefix = self.passage_prefix if prefix is None else prefix
        prepared = [f"{prefix}{t}" for t in texts]

        if self.max_tokens:
            # грубая отсечка по символам: для русского ~2 символа на токен,
            # берём запас x4, чтобы не резать полезный текст без необходимости
            limit = self.max_tokens * 4
            prepared = [t[:limit] for t in prepared]

        embeddings = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=len(prepared) > 32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Принимает список чанков, возвращает список с добавленным полем 'embedding'.
        """
        vectors = self.embed_texts([chunk['text'] for chunk in chunks])

        result = []
        for chunk, emb in zip(chunks, vectors):
            new_chunk = chunk.copy()
            new_chunk['embedding'] = emb
            result.append(new_chunk)
        return result
