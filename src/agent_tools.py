"""
Tools агента.

search_knowledge_base(query, top_k=4, category=None)
    Семантический поиск по Qdrant через существующий Retriever (e5 + query-префикс).
list_documents()
    Уникальные документы коллекции (scroll по payload Qdrant, без векторов).

Каждый tool:
- принимает явные параметры и проверяет их (тип, диапазон, допустимые значения);
- выполняет реальную операцию над индексом;
- возвращает структурированный dict: {"status": "success", ...} или
  {"status": "error", "error_code": ..., "error": ...}. Исключения наружу не выходят.

Коды ошибок:
    invalid_argument  — некорректный параметр (тип, диапазон, неизвестная категория)
    index_unavailable — индекс не собран, коллекции нет, хранилище занято другим процессом,
                        не установлена/не скачана модель эмбеддингов
    unknown_tool      — LLM запросила несуществующий tool
    internal_error    — прочие непредвиденные ошибки
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_models import Fragment

ToolResult = Dict[str, Any]


def _error(code: str, message: str, **extra: Any) -> ToolResult:
    return {"status": "error", "error_code": code, "error": message, **extra}


# ---------------------------------------------------------------------------
# Адаптер к индексу: единый интерфейс для реального Qdrant и для тестового индекса
# ---------------------------------------------------------------------------

class QdrantRetrieverAdapter:
    """Обёртка над src/retriever.py: поиск с фильтром по category и список документов."""

    def __init__(self, rag_config: Dict[str, Any]):
        from retriever import Retriever  # ленивый импорт: тесты работают без qdrant/torch
        self._retriever = Retriever(rag_config)
        self._documents: Optional[List[Dict[str, Any]]] = None

    def search(self, query: str, top_k: int, category: Optional[str] = None) -> List[Dict[str, Any]]:
        query_filter = None
        if category:
            from qdrant_client.http import models
            query_filter = models.Filter(must=[
                models.FieldCondition(key="category", match=models.MatchValue(value=category))
            ])
        return self._retriever.retrieve(query, top_k=top_k, query_filter=query_filter)

    def list_documents(self) -> List[Dict[str, Any]]:
        if self._documents is not None:
            return self._documents
        client = self._retriever.client
        collection = self._retriever.collection_name
        seen: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection, limit=512, offset=offset,
                with_payload=["document_name", "category", "file_name", "version"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                key = payload.get("file_name") or payload.get("document_name") or str(point.id)
                entry = seen.setdefault(key, {
                    "document_name": payload.get("document_name", key),
                    "category": payload.get("category", "n/a"),
                    "file_name": payload.get("file_name", key),
                    "version": payload.get("version", "n/a"),
                    "chunks": 0,
                })
                entry["chunks"] += 1
            if offset is None:
                break
        self._documents = sorted(seen.values(), key=lambda d: d["file_name"])
        return self._documents

    def close(self) -> None:
        self._retriever.close()


RetrieverFactory = Callable[[], Any]


# ---------------------------------------------------------------------------
# Реестр tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, retriever_factory: RetrieverFactory, tool_config: Dict[str, Any]):
        self._factory = retriever_factory
        self._retriever = None
        search_cfg = tool_config.get("search", {})
        self.default_top_k = int(search_cfg.get("default_top_k", 4))
        self.max_top_k = int(search_cfg.get("max_top_k", 10))
        self.max_query_chars = int(search_cfg.get("max_query_chars", 500))
        self.max_documents_listed = int(tool_config.get("list_documents", {}).get("max_items", 50))
        self._handlers = {
            "search_knowledge_base": self.search_knowledge_base,
            "list_documents": self.list_documents,
        }

    @property
    def names(self) -> List[str]:
        return list(self._handlers)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Tuple[ToolResult, int]:
        """Выполняет tool; возвращает (результат, латентность в мс). Никогда не бросает исключение."""
        started = time.monotonic()
        handler = self._handlers.get(name)
        if handler is None:
            result = _error("unknown_tool", f"Неизвестный инструмент '{name}'. Доступно: {', '.join(self.names)}.")
        elif not isinstance(arguments, dict):
            result = _error("invalid_argument", f"Аргументы должны быть объектом, получено {type(arguments).__name__}.")
        else:
            try:
                result = handler(**{k: v for k, v in arguments.items()
                                    if k in ("query", "top_k", "category")})
            except TypeError as exc:
                result = _error("invalid_argument", f"Некорректный набор аргументов: {exc}")
            except Exception as exc:  # последний рубеж: tool не должен ронять агента
                result = _error("internal_error", f"{type(exc).__name__}: {exc}")
        return result, int((time.monotonic() - started) * 1000)

    # ----- доступ к индексу -----

    def _get_retriever(self) -> Tuple[Any, Optional[ToolResult]]:
        if self._retriever is not None:
            return self._retriever, None
        try:
            self._retriever = self._factory()
            return self._retriever, None
        except ImportError as exc:
            return None, _error("index_unavailable",
                                f"не установлены зависимости поиска ({exc}). Выполните pip install -r requirements.txt.")
        except Exception as exc:
            # RuntimeError «коллекция не найдена» / «storage already accessed»,
            # ValueError несовпадения размерности, OSError при чтении хранилища
            return None, _error("index_unavailable", f"индекс недоступен: {exc}")

    def close(self) -> None:
        if self._retriever is not None and hasattr(self._retriever, "close"):
            try:
                self._retriever.close()
            except Exception:
                pass
        self._retriever = None

    # ----- валидация -----

    def _parse_top_k(self, value: Any) -> Tuple[Optional[int], Optional[str]]:
        if value is None:
            return self.default_top_k, None
        if isinstance(value, bool):
            return None, f"top_k должен быть целым числом, получено логическое значение {value}."
        if isinstance(value, float):
            if not value.is_integer():
                return None, f"top_k должен быть целым числом, получено {value}."
            value = int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text.lstrip("-").isdigit():
                return None, f"top_k должен быть целым числом, получено «{value}»."
            value = int(text)
        if not isinstance(value, int):
            return None, f"top_k должен быть целым числом, получено {type(value).__name__}."
        if not 1 <= value <= self.max_top_k:
            return None, f"top_k должен быть в диапазоне от 1 до {self.max_top_k}, получено {value}."
        return value, None

    # ----- tools -----

    def search_knowledge_base(self, query: Any = None, top_k: Any = None, category: Any = None) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return _error("invalid_argument", "query обязателен и должен быть непустой строкой.")
        query = query.strip()
        truncated = len(query) > self.max_query_chars
        if truncated:
            query = query[: self.max_query_chars]

        k, problem = self._parse_top_k(top_k)
        if problem:
            return _error("invalid_argument", problem, argument="top_k")

        if category is not None and not isinstance(category, str):
            return _error("invalid_argument", "category должна быть строкой.", argument="category")
        category = (category or "").strip() or None

        retriever, failure = self._get_retriever()
        if failure:
            return failure

        if category:
            try:
                known = sorted({d["category"] for d in retriever.list_documents()})
            except Exception as exc:
                return _error("index_unavailable", f"не удалось прочитать категории: {exc}")
            if category not in known:
                return _error("invalid_argument",
                              f"неизвестная категория «{category}». Допустимые: {', '.join(known)}.",
                              argument="category")

        try:
            hits = retriever.search(query, k, category)
        except Exception as exc:
            return _error("index_unavailable", f"ошибка поиска в индексе: {exc}")

        return {
            "status": "success",
            "query": query,
            "query_truncated": truncated,
            "top_k": k,
            "category": category,
            "count": len(hits),
            "hits": hits,
        }

    def list_documents(self) -> ToolResult:
        retriever, failure = self._get_retriever()
        if failure:
            return failure
        try:
            documents = retriever.list_documents()
        except Exception as exc:
            return _error("index_unavailable", f"не удалось получить список документов: {exc}")
        return {
            "status": "success",
            "count": len(documents),
            "documents": documents[: self.max_documents_listed],
            "truncated": len(documents) > self.max_documents_listed,
        }


# ---------------------------------------------------------------------------
# Представление результатов для LLM
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0] + " […]"
    return text


def hits_to_fragments(hits: List[Dict[str, Any]], start_number: int, max_chars_per_chunk: int,
                      max_context_chars: int) -> List[Fragment]:
    """
    Нумерует фрагменты сквозной нумерацией в рамках запуска (start_number+1, ...) и
    отбрасывает хвост, не влезающий в бюджет контекста: на отброшенные фрагменты
    модель ссылаться не сможет — их нет ни в контексте, ни в реестре.
    """
    fragments: List[Fragment] = []
    used = 0
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        text = _truncate(hit.get("text", ""), max_chars_per_chunk)
        if fragments and max_context_chars and used + len(text) > max_context_chars:
            break
        used += len(text)
        fragments.append(Fragment(
            number=start_number + len(fragments) + 1,
            chunk_id=hit.get("chunk_id"),
            score=float(hit.get("score", 0.0)),
            document_name=str(meta.get("document_name", "неизвестно")),
            file_name=str(meta.get("file_name", meta.get("source", "n/a"))),
            category=str(meta.get("category", "n/a")),
            chunk_index=meta.get("chunk_index", "?"),
            text=text,
        ))
    return fragments


def format_fragments(fragments: List[Fragment], query: str) -> str:
    if not fragments:
        return f"Поиск по запросу «{query}» не вернул фрагментов."
    blocks = [f"Найдено фрагментов: {len(fragments)} (запрос: «{query}»)."]
    for f in fragments:
        blocks.append(
            f"[Фрагмент {f.number}]\n"
            f"Источник: {f.document_name} (категория {f.category})\n"
            f"Файл: {f.file_name} | чанк {f.chunk_index} | score {f.score:.4f}\n"
            f"Текст: {f.text}"
        )
    return "\n\n".join(blocks)


def format_documents(result: ToolResult) -> str:
    lines = [f"В базе знаний документов: {result['count']}."]
    for d in result["documents"]:
        lines.append(f"- {d['document_name']} | категория {d['category']} | файл {d['file_name']} | чанков {d['chunks']}")
    if result.get("truncated"):
        lines.append(f"(показаны первые {len(result['documents'])})")
    return "\n".join(lines)


def format_error(result: ToolResult) -> str:
    return f"ОШИБКА [{result.get('error_code')}]: {result.get('error')}"
