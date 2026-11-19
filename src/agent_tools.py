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
import re
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
    if not category and not any(filters.get(k) for k in ("year_from", "year_to", "industry", "cards", "files")):
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
    if filters.get("files"):
        must.append(models.FieldCondition(key="file_name", match=models.MatchAny(any=list(filters["files"]))))
    if filters.get("cards"):
        # ID карточек уникальны только внутри своего реестра, поэтому пара (card_type, card_id)
        must.append(models.Filter(should=[
            models.Filter(must=[
                models.FieldCondition(key="card_type", match=models.MatchValue(value=c["card_type"])),
                models.FieldCondition(key="card_id", match=models.MatchValue(value=c["card_id"])),
            ]) for c in filters["cards"]
        ]))
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
    if filters.get("files") and meta.get("file_name") not in filters["files"]:
        return False
    if filters.get("cards"):
        if not any(meta.get("card_type") == c["card_type"] and str(meta.get("card_id")) == str(c["card_id"])
                   for c in filters["cards"]):
            return False
    return True


# Русские синонимы отраслевых категорий реестров: для сопоставления значений от модели и
# для подсказок в описании инструмента. Ключи — значения поля industry в индексе.
INDUSTRY_SYNONYMS = {
    "Pharma & HC": ["медицина", "здравоохранение", "фармацевтика", "фарма", "фарм", "клиники", "больницы"],
    "Energy (O&G / chem / utilities)": ["энергетика", "нефтегаз", "нефтянка", "нефтяная", "нефть", "газ", "газовая", "химия",
                                        "электроэнергетика", "ЖКХ", "ТЭК"],
    "Industrial goods": ["промышленность", "металлургия", "горнодобывающая", "машиностроение", "ОПК"],
    "Transportation": ["транспорт", "логистика", "перевозки", "железная дорога", "ЖД", "авиация"],
    "Telecom & Media & IT": ["телеком", "медиа", "ИТ", "IT", "телекоммуникации"],
    "Public": ["госсектор", "государство", "государственный", "некоммерческий", "наука"],
    "Consumer goods & retail": ["ритейл", "розница", "торговля", "магазины", "потребительские товары", "FMCG",
                                "туризм", "агро", "сельское хозяйство"],
    "Financial services": ["финансы", "банки", "банк", "страхование"],
    "Construction & Real estate": ["строительство", "недвижимость", "девелопмент"],
}


# Синонимы типов услуг (поле «Тип услуги» реестров). Тип распознаётся по всем значимым словам названия
# («операционной эффективности») или по синониму.
SERVICE_SYNONYMS = {
    "Операционная эффективность": ["операционка", "операционные улучшения", "бережливое производство", "lean"],
    "Стратегия": ["стратегический", "стратегическое"],
    "Организационное развитие / HR": ["HR", "персонал", "кадры", "оргструктура", "организационная структура"],
    "Маркетинг и продажи": ["маркетинг", "продажи", "коммерческая функция"],
    "Digital": ["цифровизация", "цифровая трансформация", "диджитал"],
    "Корпоративные финансы и инвестиции": ["корпоративные финансы", "M&A", "сделки", "инвестиционный"],
    "Экономические исследования": ["исследование рынка", "экономическое исследование", "маркетинговое исследование"],
    "Законодательное регулирование": ["регулирование", "законодательство", "нормативное"],
}


# общие слова в названиях детальных тегов: по ним одним тег не распознаём
TAG_GENERIC = {"управление", "разработка", "стратегия", "оптимизация", "развитие", "анализ", "внедрение",
               "поддержка", "обзор", "корпоративная", "корпоративное", "функциональная", "отраслевая",
               "отраслевое", "проектами", "программами", "решения", "лучшие", "практики", "модели",
               "модель", "структура", "система", "систем", "бизнес", "процессов", "бизнес-процессов",
               "проект", "проекта", "проекты", "проектов", "проектный", "проектная", "офис"}


def tag_keys(tag: str) -> List[str]:
    """Значимые слова детального тега: «Закупки / управление поставщиками» -> ["закупки", "поставщиками"]."""
    words = [w for w in _words(tag) if len(w) >= 4 and w not in TAG_GENERIC]
    return words


def detect_services(text: str, known: List[str], details: Optional[List[str]] = None) -> List[str]:
    """
    Типы услуг (все значимые слова названия или синоним) и детальные теги услуг
    (хотя бы одно значимое слово тега), названные в вопросе.
    """
    from agent_lexical import stems
    q_words = _words(text)
    q_stems = set(stems(text))
    found, consumed = [], set()
    for service in known:
        name_stems = set(stems(service))
        synonyms = SERVICE_SYNONYMS.get(service, [])
        if (name_stems and name_stems <= q_stems) or _matches_synonyms(text, synonyms):
            found.append(service)
            consumed |= {w for w in q_words if w[:4] in name_stems or _matches_synonyms(w, synonyms)}
    for tag in details or []:
        keys = tag_keys(tag)
        if tag in found or not keys:
            continue
        matched = {w for w in q_words if _matches_synonyms(w, keys)}
        # тег берём, только если его зацепило слово, которое не объясняется уже найденным типом услуги:
        # «операционной эффективности» — тип, а не ещё и тег «КПЭ / Управление эффективностью»
        if matched - consumed:
            found.append(tag)
    return found


def covered_words(text: str, phrases: List[str]) -> set:
    """Слова текста, которые объясняются фразами-синонимами, найденными в тексте (для многословных — все их слова)."""
    words = _words(text)
    covered = set()
    for phrase in phrases:
        if _matches_synonyms(text, [phrase]):
            keys = [_syn_key(w) for w in _words(phrase) if len(w) >= 2]
            covered |= {w for w in words if any(_syn_match(w, k) for k in keys)}
    return covered


# слова, которые в вопросе чаще обозначают тему, а не отрасль («ИТ-стратегия»): по тексту вопроса их не распознаём
DETECT_EXCLUDE = {"ит", "it"}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-zа-яё]+", text.lower().replace("ё", "е"))


def _syn_key(word: str) -> str:
    """
    Ключ синонима: до 5 букв — слово целиком («банк», «газ», «медиа»), 6–7 букв — первые 5 («закуп»),
    8–11 букв — первые 6 («трансп», «медици»): по 5 буквам «транспорт» совпадал с «трансформацией»;
    от 12 букв — первые 8 («электроэ»): по 6 буквам «электроэнергетика» совпадала с «электрификацией».
    """
    if len(word) <= 5:
        return word
    if len(word) <= 7:
        return word[:5]
    # от 12 букв — 8: по 6 буквам «электроэнергетика» совпадала с «электрификацией» («электр»)
    return word[:6] if len(word) <= 11 else word[:8]


def _syn_match(word: str, key: str) -> bool:
    """Слово вопроса начинается с ключа синонима: «банках» ~ «банк», «медицине» ~ «медиц», но не ~ «медиа»."""
    if len(key) <= 3:                 # «газ», «жкх», «тэк»: допускаем только окончания («газа», «газовой» — нет)
        return word.startswith(key) and len(word) <= len(key) + 2
    return word.startswith(key)


def _matches_synonyms(text: str, phrases: List[str], exclude: set = frozenset()) -> bool:
    """
    Есть ли в тексте хотя бы одна фраза-синоним. Многословная фраза засчитывается, только если в тексте есть
    ВСЕ её значимые слова: иначе «цифровая трансформация» срабатывала на «транспорте» по слову «трансформация».
    """
    words = [w for w in _words(text) if w not in exclude]
    for phrase in phrases:
        keys = [_syn_key(w) for w in _words(phrase) if len(w) >= 2]
        if keys and all(any(_syn_match(w, k) for w in words) for k in keys):
            return True
    return False


def detect_industries(text: str, known: List[str]) -> List[str]:
    """Отрасли, названные в вопросе словами: «в медицине» -> Pharma & HC. Только по синонимам и названиям категорий."""
    found = []
    for category in known:
        phrases = INDUSTRY_SYNONYMS.get(category, []) + re.findall(r"[A-Za-z]{4,}", category)
        if _matches_synonyms(text, phrases, DETECT_EXCLUDE):
            found.append(category)
    return found


NUMBER_WORDS = {"один": 1, "одного": 1, "два": 2, "двух": 2, "три": 3, "трёх": 3, "трех": 3, "четыре": 4,
                "четырёх": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
                "восемь": 8, "восьми": 8, "десять": 10, "десяти": 10}


def detect_period(text: str, current_year: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    «за 2020–2023», «с 2018 по 2021», «после 2019», «до 2015», «в 2020 году» -> (год_с, год_по);
    относительные: «за последние 5 лет» -> (текущий−4, текущий), «в прошлом году», «в этом году», «за последний год».
    """
    import datetime
    year = current_year or datetime.date.today().year
    t = text.lower().replace("ё", "е")
    m = re.search(r"последни[еx]\s+(\d{1,2}|[а-я]+)\s+(лет|года|год)", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS.get(m.group(1))
        if n:
            return year - n + 1, year
    if re.search(r"последн(ий|ие)\s+год|прошл(ый|ом|ого)\s+год", t):
        return (year - 1, year) if "последн" in t else (year - 1, year - 1)
    if re.search(r"(этом|текущем|нынешнем)\s+году|(этот|текущий)\s+год", t):
        return year, year
    m = re.search(r"((?:19|20)\d{2})\s*(?:-|–|—|по|до)\s*((?:19|20)\d{2})", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    m = re.search(r"(?:после|позже|начиная с|с)\s+((?:19|20)\d{2})", t)
    if m:
        return int(m.group(1)) + (1 if "после" in m.group(0) or "позже" in m.group(0) else 0), None
    m = re.search(r"(?:до|раньше|ранее)\s+((?:19|20)\d{2})", t)
    if m:
        return None, int(m.group(1)) - 1
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", t)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None, None


RetrieverFactory = Callable[[], Any]


# ---------------------------------------------------------------------------
# Реестр tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, retriever_factory: RetrieverFactory, tool_config: Dict[str, Any],
                 catalog_path: Optional[str] = None, escalation_queue: Any = None):
        self._factory = retriever_factory
        self._retriever = None
        self._catalog_path = catalog_path
        self._catalog = None
        self.max_cards_listed = int(tool_config.get("list_cards", {}).get("max_items", 50))
        search_cfg = tool_config.get("search", {})
        self.default_top_k = int(search_cfg.get("default_top_k", 4))
        self.max_top_k = int(search_cfg.get("max_top_k", 10))
        self.max_query_chars = int(search_cfg.get("max_query_chars", 500))
        self.max_documents_listed = int(tool_config.get("list_documents", {}).get("max_items", 50))
        self._handlers = {
            "search_knowledge_base": self.search_knowledge_base,
            "list_documents": self.list_documents,
        }
        if catalog_path:
            self._handlers["list_cards"] = self.list_cards
        self._escalation = escalation_queue
        if escalation_queue is not None:
            self._handlers["escalate_to_human"] = self.escalate_to_human

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
                allowed = {
                    "search_knowledge_base": ("query", "top_k", "category", "year_from", "year_to", "industry",
                                              "cards", "files"),
                    "list_documents": (),
                    "list_cards": ("card_type", "industry", "service", "person", "year_from", "year_to",
                                   "include_people", "group_by", "cards", "limit"),
                    "escalate_to_human": ("reason", "question", "route_path", "evidence", "topic", "request_id"),
                }[name]
                result = handler(**{k: v for k, v in arguments.items() if k in allowed})
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
            if not canon:
                # модель может скопировать значение вместе с синонимами из описания: «Pharma & HC (медицина, …)»
                canon = by_lower.get(re.sub(r"\s*\([^()]*\)\s*$", "", v).strip().lower())
            if canon:
                resolved.append(canon)
                continue
            # неточное значение («Здравоохранение», «фармацевтика») — ищем по вхождению и основам слов;
            # принимаем, только если вариантов немного, иначе просим модель выбрать из списка
            close = ToolRegistry._close_industries(v, known)
            if 0 < len(close) <= 3:
                resolved.extend(c for c in close if c not in resolved)
            else:
                unknown.append(v)
        if unknown:
            shown = ", ".join(known[:30]) + (" …" if len(known) > 30 else "")
            return None, f"неизвестная отрасль: {', '.join(unknown)}. Допустимые: {shown}."
        return resolved, None

    @staticmethod
    def _close_industries(value: str, known: List[str]) -> List[str]:
        """
        Неточное значение отрасли -> значения из индекса:
        вхождение строки («Energy» -> «Energy (O&G / chem / utilities)»), совпадение основ слов,
        затем русские синонимы категорий («Здравоохранение и фармацевтика», «медицина» -> «Pharma & HC»).
        """
        from agent_lexical import stems
        v = value.strip().lower()
        if len(v) < 2:
            return []
        if len(v) >= 4:
            by_substring = [k for k in known if v in k.lower() or k.lower() in v]
            if by_substring:
                return by_substring
        v_stems = set(stems(v)) | ({v} if len(v) < 4 else set())
        by_stems = [k for k in known if v_stems and v_stems <= set(stems(k))]
        if by_stems:
            return by_stems
        # для синонимов основы длиннее: по 4 символам «медицина» и «медиа» совпадают («меди»)
        return [category for category, words in INDUSTRY_SYNONYMS.items()
                if category in known and _matches_synonyms(v, words)]

    # ----- tools -----

    def search_knowledge_base(self, query: Any = None, top_k: Any = None, category: Any = None,
                              year_from: Any = None, year_to: Any = None, industry: Any = None,
                              cards: Any = None, files: Any = None) -> ToolResult:
        """cards — внутренний параметр агента: [{"card_type", "card_id"}] карточек прошлого ответа."""
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

        if cards is not None and not (isinstance(cards, list) and all(
                isinstance(c, dict) and c.get("card_type") and c.get("card_id") for c in cards)):
            return _error("invalid_argument", "cards должен быть списком {card_type, card_id}.", argument="cards")
        if files is not None and not (isinstance(files, list) and all(isinstance(f, str) for f in files)):
            return _error("invalid_argument", "files должен быть списком имён файлов.", argument="files")
        filters = {k: v for k, v in (("year_from", y_from), ("year_to", y_to), ("industry", industries),
                                     ("cards", cards), ("files", files)) if v}
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

    @property
    def has_catalog(self) -> bool:
        return self.catalog() is not None

    def catalog(self):
        if not self._catalog_path:
            return None
        if self._catalog is None:
            try:
                from agent_cards import CardCatalog
                self._catalog = CardCatalog(self._catalog_path)
            except Exception:
                self._catalog_path = None      # каталога нет — работаем без перечислений
                return None
        return self._catalog

    @staticmethod
    def _parse_named(name: str, value: Any, known: List[str], synonyms: Dict[str, List[str]]
                     ) -> Tuple[Optional[List[str]], Optional[str]]:
        """Значение из словаря известных (тип услуги, человек): точное без регистра, затем по словам и синонимам."""
        if value in (None, "", []):
            return None, None
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
            return None, f"{name} должен быть строкой или списком строк."
        by_lower = {k.lower(): k for k in known}
        resolved, unknown = [], []
        for v in items:
            canon = by_lower.get(v.strip().lower()) or \
                by_lower.get(re.sub(r"\s*\([^()]*\)\s*$", "", v).strip().lower())
            close = [canon] if canon else detect_services(v, [k for k in known if k in synonyms],
                                                          [k for k in known if k not in synonyms]) if synonyms else []
            if close:
                resolved += [c for c in close if c not in resolved]
            else:
                unknown.append(v)
        if unknown:
            shown = ", ".join(known[:20]) + (" …" if len(known) > 20 else "")
            return None, f"неизвестное значение {name}: {', '.join(unknown)}. Допустимые: {shown}."
        return resolved, None

    def list_cards(self, card_type: Any = None, industry: Any = None, year_from: Any = None,
                   year_to: Any = None, include_people: Any = False, cards: Any = None,
                   limit: Any = None, service: Any = None, person: Any = None, group_by: Any = None) -> ToolResult:
        """
        Все карточки реестров по условиям (без top_k): для перечислений и подсчётов.
        group_by="people" — рейтинг людей по числу карточек (проекты и пропозалы отдельно).
        """
        from agent_cards import CARD_TYPE_RU, format_card_list, format_people_ranking
        catalog = self.catalog()
        if catalog is None:
            return _error("index_unavailable", "каталог карточек недоступен: выполните python tools/csv_to_cards.py")
        if card_type not in (None, "") and card_type not in CARD_TYPE_RU:
            return _error("invalid_argument", f"card_type должен быть одним из: {', '.join(CARD_TYPE_RU)}.",
                          argument="card_type")
        y_from, problem = self._parse_year("year_from", year_from)
        if problem:
            return _error("invalid_argument", problem, argument="year_from")
        y_to, problem = self._parse_year("year_to", year_to)
        if problem:
            return _error("invalid_argument", problem, argument="year_to")
        if y_from and y_to and y_from > y_to:
            return _error("invalid_argument", f"year_from ({y_from}) больше year_to ({y_to}).", argument="year_from")
        industries = None
        if industry not in (None, "", []):
            industries, problem = self._parse_industry(industry, catalog.industries())
            if problem:
                return _error("invalid_argument", problem, argument="industry")
        services, problem = self._parse_named("service", service, catalog.services() + catalog.service_details(),
                                              SERVICE_SYNONYMS)
        if problem:
            return _error("invalid_argument", problem, argument="service")
        persons = None
        if person not in (None, "", []):
            items = [person] if isinstance(person, str) else person
            persons = [p for v in items for p in (catalog.find_person(v) or [catalog.person_key(v)])]
        if group_by not in (None, "", "people"):
            return _error("invalid_argument", "group_by поддерживает только значение people.", argument="group_by")
        n, problem = self._parse_top_k(limit) if limit not in (None, "") else (self.max_cards_listed, None)
        if problem:
            n = self.max_cards_listed
        n = min(max(n, 1), self.max_cards_listed)
        selected = catalog.select(card_type or None, industries, y_from, y_to,
                                  cards if isinstance(cards, list) else None, services, persons)
        people = include_people in (True, "true", "True", 1)
        filters = {k: v for k, v in (("card_type", card_type), ("industry", industries), ("service", services),
                                     ("person", persons), ("year_from", y_from), ("year_to", y_to)) if v}
        result = {
            "status": "success",
            "total": len(selected),
            "shown": min(len(selected), n),
            "filters": filters,
            "cards": selected[:n],
            "text": format_card_list(selected, n, people),
            "include_people": people,
            "group_by": group_by or None,
        }
        if group_by == "people":
            ranking = catalog.aggregate_people(selected)
            top = min(n, 15)
            result.update({
                "people": ranking[:top],
                "people_total": len(ranking),
                "cards_with_people": sum(1 for c in selected if c.get("_people")),
                "by_type": {t: sum(1 for c in selected if c.get("card_type") == t) for t in ("project", "proposal")},
                "text": format_people_ranking(ranking, top),
            })
        return result

    def files_for_product(self, product: str) -> List[str]:
        """Файлы документов, в названии которых есть продукт («ERGO — бизнес-симулятор…»)."""
        retriever, failure = self._get_retriever()
        if failure:
            return []
        try:
            documents = retriever.list_documents()
        except Exception:
            return []
        pattern = re.compile(r"(?<![a-zа-яё0-9])" + re.escape(product.lower()) + r"(?![a-zа-яё0-9])")
        return sorted({d["file_name"] for d in documents
                       if pattern.search(str(d.get("document_name", "")).lower())
                       and not str(d.get("category", "")).startswith("internal/")})

    def escalate_to_human(self, reason: Any = None, question: Any = None, route_path: Any = None,
                          evidence: Any = None, topic: Any = None, request_id: Any = None) -> ToolResult:
        """Обращение к ответственному в очереди эскалаций (правила E1–E4). Вызывает агент по правилам, не модель."""
        from agents.escalation import REASONS
        if reason not in REASONS:
            return _error("invalid_argument", f"reason должен быть одним из: {', '.join(REASONS)}.", argument="reason")
        if not isinstance(question, str):
            return _error("invalid_argument", "question должен быть строкой.", argument="question")
        ticket = self._escalation.create(reason=reason, question=question, route_path=list(route_path or []),
                                         evidence=list(evidence or []), request_id=request_id, topic=topic)
        return {"status": "success", "ticket": ticket}

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
