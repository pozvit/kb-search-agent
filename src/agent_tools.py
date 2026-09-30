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

import inspect
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
    """Обёртка над src/retriever.py: поиск с фильтрами по payload, список документов и фасеты."""

    SCAN_FIELDS = ["document_name", "category", "file_name", "version",
                   "industry", "year", "year_end", "card_type"]

    def __init__(self, rag_config: Dict[str, Any]):
        from retriever import Retriever  # ленивый импорт: тесты работают без qdrant/torch
        self._retriever = Retriever(rag_config)
        self._scan: Optional[Dict[str, Any]] = None

    def search(self, query: str, top_k: int, category: Optional[str] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._retriever.retrieve(query, top_k=top_k,
                                        query_filter=build_qdrant_filter(category, filters))

    def _scan_payloads(self) -> Dict[str, Any]:
        """Один проход scroll по payload без векторов: документы и фасеты, с кэшем."""
        if self._scan is not None:
            return self._scan
        client, collection = self._retriever.client, self._retriever.collection_name
        acc = FacetAccumulator()
        offset = None
        while True:
            points, offset = client.scroll(collection_name=collection, limit=1024, offset=offset,
                                           with_payload=self.SCAN_FIELDS, with_vectors=False)
            for point in points:
                acc.add(point.payload or {}, str(point.id))
            if offset is None:
                break
        self._scan = acc.result()
        return self._scan

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._scan_payloads()["documents"]

    def facets(self) -> Dict[str, Any]:
        return self._scan_payloads()["facets"]

    def close(self) -> None:
        self._retriever.close()


class FacetAccumulator:
    """Собирает по payload чанков список документов и фасеты: категории, отрасли, годы, типы карточек."""

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, int] = {}
        self.industries: Dict[str, int] = {}
        self.card_types: Dict[str, int] = {}
        self.years: List[int] = []

    def add(self, payload: Dict[str, Any], fallback_key: str = "?") -> None:
        key = payload.get("file_name") or payload.get("document_name") or fallback_key
        entry = self.documents.setdefault(key, {
            "document_name": payload.get("document_name", key),
            "category": payload.get("category", "n/a"),
            "file_name": payload.get("file_name", key),
            "version": payload.get("version", "n/a"),
            "chunks": 0,
        })
        entry["chunks"] += 1
        cat = payload.get("category") or "n/a"
        self.categories[cat] = self.categories.get(cat, 0) + 1
        for value in payload.get("industry") or []:
            self.industries[value] = self.industries.get(value, 0) + 1
        if payload.get("card_type"):
            self.card_types[payload["card_type"]] = self.card_types.get(payload["card_type"], 0) + 1
        for field_name in ("year", "year_end"):
            if isinstance(payload.get(field_name), int):
                self.years.append(payload[field_name])

    def result(self) -> Dict[str, Any]:
        order = lambda d: dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))
        return {
            "documents": sorted(self.documents.values(), key=lambda d: d["file_name"]),
            "facets": {
                "categories": order(self.categories),
                "industries": order(self.industries),
                "card_types": order(self.card_types),
                "year_min": min(self.years) if self.years else None,
                "year_max": max(self.years) if self.years else None,
            },
        }


def build_qdrant_filter(category: Optional[str], filters: Optional[Dict[str, Any]]):
    """
    category      -> category == значение
    year_from     -> year_end >= year_from   (карточка пересекается с периодом)
    year_to       -> year <= year_to
    industry      -> industry содержит хотя бы одно из значений
    Чанки без года или отрасли (коммерческие предложения, PDF продуктов) под
    фильтр по году или отрасли не попадают — фильтр по сути означает «реестры».
    """
    filters = filters or {}
    if not category and not any(filters.get(k) for k in ("year_from", "year_to", "industry")):
        return None
    from qdrant_client.http import models
    must = []
    if category:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if filters.get("year_from") is not None:
        must.append(models.FieldCondition(key="year_end", range=models.Range(gte=filters["year_from"])))
    if filters.get("year_to") is not None:
        must.append(models.FieldCondition(key="year", range=models.Range(lte=filters["year_to"])))
    if filters.get("industry"):
        must.append(models.FieldCondition(key="industry", match=models.MatchAny(any=list(filters["industry"]))))
    return models.Filter(must=must)


def payload_matches(meta: Dict[str, Any], category: Optional[str], filters: Optional[Dict[str, Any]]) -> bool:
    """Та же семантика фильтра, что build_qdrant_filter, — для лексического и тестового индексов."""
    filters = filters or {}
    if category and meta.get("category") != category:
        return False
    if filters.get("year_from") is not None:
        if not isinstance(meta.get("year_end"), int) or meta["year_end"] < filters["year_from"]:
            return False
    if filters.get("year_to") is not None:
        if not isinstance(meta.get("year"), int) or meta["year"] > filters["year_to"]:
            return False
    if filters.get("industry"):
        if not set(meta.get("industry") or []) & set(filters["industry"]):
            return False
    return True


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
                                    if k in ("query", "top_k", "category", "year_from", "year_to", "industry")})
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

    def facets(self) -> Optional[Dict[str, Any]]:
        """Фасеты индекса (категории, отрасли, годы) или None, если индекс недоступен или их нет."""
        retriever, failure = self._get_retriever()
        if failure or not hasattr(retriever, "facets"):
            return None
        try:
            return retriever.facets()
        except Exception:
            return None

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

    @staticmethod
    def _parse_year(name: str, value: Any) -> Tuple[Optional[int], Optional[str]]:
        if value is None or value == "":
            return None, None
        if isinstance(value, bool):
            return None, f"{name} должен быть годом (целым числом), получено логическое значение."
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if not isinstance(value, int):
            return None, f"{name} должен быть годом (целым числом), получено «{value}»."
        if not 1900 <= value <= 2100:
            return None, f"{name} должен быть в диапазоне 1900–2100, получено {value}."
        return value, None

    @staticmethod
    def _parse_industry(value: Any, known: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
        if value in (None, "", []):
            return None, None
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
            return None, "industry должен быть строкой или списком строк."
        if not known:
            return None, "в этом индексе нет отраслевой разметки: фильтр industry недоступен."
        by_lower = {k.lower(): k for k in known}
        resolved, unknown = [], []
        for v in items:
            canon = by_lower.get(v.strip().lower())
            (resolved if canon else unknown).append(canon or v)
        if unknown:
            shown = ", ".join(known[:30]) + (" …" if len(known) > 30 else "")
            return None, f"неизвестная отрасль: {', '.join(unknown)}. Допустимые: {shown}."
        return resolved, None

    # ----- tools -----

    def search_knowledge_base(self, query: Any = None, top_k: Any = None, category: Any = None,
                              year_from: Any = None, year_to: Any = None, industry: Any = None) -> ToolResult:
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

        y_from, problem = self._parse_year("year_from", year_from)
        if problem:
            return _error("invalid_argument", problem, argument="year_from")
        y_to, problem = self._parse_year("year_to", year_to)
        if problem:
            return _error("invalid_argument", problem, argument="year_to")
        if y_from and y_to and y_from > y_to:
            return _error("invalid_argument", f"year_from ({y_from}) больше year_to ({y_to}).", argument="year_from")

        retriever, failure = self._get_retriever()
        if failure:
            return failure

        if category:
            try:
                facets = self.facets() or {}
                known = list((facets.get("categories") or {}).keys()) or \
                    sorted({d["category"] for d in retriever.list_documents()})
            except Exception as exc:
                return _error("index_unavailable", f"не удалось прочитать категории: {exc}")
            if not known:
                return _error("invalid_argument", "в индексе нет разметки по категориям: фильтр category недоступен.",
                              argument="category")
            if category not in known:
                return _error("invalid_argument",
                              f"неизвестная категория «{category}». Допустимые: {', '.join(known)}.",
                              argument="category")

        industries = None
        if industry not in (None, "", []):
            facets = self.facets() or {}
            industries, problem = self._parse_industry(industry, list((facets.get("industries") or {}).keys()))
            if problem:
                return _error("invalid_argument", problem, argument="industry")

        filters = {k: v for k, v in (("year_from", y_from), ("year_to", y_to), ("industry", industries)) if v}
        supports_filters = "filters" in inspect.signature(retriever.search).parameters
        if filters and not supports_filters:
            return _error("invalid_argument", "этот индекс не поддерживает фильтры по году и отрасли.")
        try:
            if supports_filters:
                hits = retriever.search(query, k, category, filters or None)
            else:
                hits = retriever.search(query, k, category)
        except Exception as exc:
            return _error("index_unavailable", f"ошибка поиска в индексе: {exc}")

        return {
            "status": "success",
            "query": query,
            "query_truncated": truncated,
            "top_k": k,
            "category": category,
            "filters": filters,
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
            "facets": self.facets(),
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
            card={k: meta[k] for k in CARD_KEYS if meta.get(k) not in (None, "", [])},
        ))
    return fragments


CARD_KEYS = ("card_type", "card_id", "card_title", "year", "year_end", "industry", "service", "status")
CARD_TYPE_RU = {"project": "проект", "proposal": "пропозал", "project_accounting": "проект учётной базы"}


def card_label(card: Dict[str, Any]) -> str:
    """«проект учётной базы | 2019–2020 | Industrial goods; Машиностроение и ОПК»"""
    parts = [CARD_TYPE_RU.get(card.get("card_type"), card.get("card_type", ""))]
    year, year_end = card.get("year"), card.get("year_end")
    if year:
        parts.append(f"{year}–{year_end}" if year_end and year_end != year else str(year))
    if card.get("industry"):
        parts.append("; ".join(card["industry"][:4]))
    return " | ".join(p for p in parts if p)


def format_fragments(fragments: List[Fragment], query: str) -> str:
    if not fragments:
        return f"Поиск по запросу «{query}» не вернул фрагментов."
    blocks = [f"Найдено фрагментов: {len(fragments)} (запрос: «{query}»)."]
    for f in fragments:
        source = f"Источник: {f.document_name} (категория {f.category})"
        if f.card:
            source = f"Карточка: {f.card.get('card_title', '')} ({card_label(f.card)})"
        blocks.append(
            f"[Фрагмент {f.number}]\n"
            f"{source}\n"
            f"Файл: {f.file_name} | чанк {f.chunk_index} | score {f.score:.4f}\n"
            f"Текст: {f.text}"
        )
    return "\n\n".join(blocks)


def format_documents(result: ToolResult) -> str:
    lines = [f"В базе знаний документов: {result['count']}."]
    facets = result.get("facets") or {}
    if facets.get("categories"):
        lines.append("Категории (чанков): " + "; ".join(f"{k} ({v})" for k, v in facets["categories"].items()))
    if facets.get("year_min"):
        lines.append(f"Годы в реестрах: {facets['year_min']}–{facets['year_max']}")
    for d in result["documents"]:
        lines.append(f"- {d['document_name']} | категория {d['category']} | файл {d['file_name']} | чанков {d['chunks']}")
    if result.get("truncated"):
        lines.append(f"(показаны первые {len(result['documents'])})")
    return "\n".join(lines)


def format_error(result: ToolResult) -> str:
    return f"ОШИБКА [{result.get('error_code')}]: {result.get('error')}"
