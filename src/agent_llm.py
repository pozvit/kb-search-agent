"""
Бэкенды LLM для агента. Интерфейс один: chat(messages, tools) -> LLMReply.

OllamaChatBackend — нативный эндпоинт Ollama POST /api/chat (stdlib, без зависимостей).
    Почему не OpenAI-совместимый /v1: через /v1 нельзя передать options.num_ctx.
    Без явного num_ctx Ollama берёт дефолтное окно модели, и промпт агента
    (system + tools + история + ~8 000 символов фрагментов ≈ 4–5 тыс. токенов)
    может молча обрезаться. Через /api/chat окно задаётся в запросе, Modelfile не нужен.
    Настройки берутся ТОЛЬКО из config/agent.yaml и переменных AGENT_LLM_* (см. agent_config.py):
    LLM_API_KEY / LLM_BASE_URL из .env (они для YandexGPT в rag_answer.py) сюда не
    попадают — ключ Yandex не уйдёт на локальный эндпоинт.

MockLLM — детерминированная модель для pytest и воспроизводимого прогона сценариев
    без Ollama. Не содержит заготовленных ответов: tool call строит из запроса и истории,
    а финальный ответ — экстрактивно из фрагментов, которые вернул tool.
"""

import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_client import LLMError

from agent_lexical import stems as _stem_list
from agent_prompts import NOT_FOUND_ANSWER, SMALLTALK_ANSWER


@dataclass
class LLMReply:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)   # [{"name", "arguments"}]
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    raw_tool_call_in_text: bool = False


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Qwen2.5 иногда возвращает вызов текстом (<tool_call>{...}</tool_call> или голым JSON),
    если шаблон Ollama его не распознал. Подбираем такие вызовы, чтобы не потерять шаг.
    """
    candidates = _TOOL_CALL_TAG.findall(content or "")
    stripped = (content or "").strip()
    if not candidates and stripped.startswith("{") and stripped.endswith("}"):
        candidates = [stripped]
    calls = []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


class OllamaChatBackend:
    name = "ollama"

    def __init__(self, llm_config: Dict[str, Any]):
        self.base_url = str(llm_config.get("base_url", "http://localhost:11434")).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]          # нативный API живёт в корне, не в /v1
        self.model = llm_config.get("model", "qwen2.5:7b-instruct")
        self.timeout = int(llm_config.get("timeout_seconds", 600))
        self.keep_alive = llm_config.get("keep_alive", "30m")
        self.options = {
            "temperature": float(llm_config.get("temperature", 0.0)),
            "num_ctx": int(llm_config.get("num_ctx", 8192)),
            "num_predict": int(llm_config.get("max_tokens", 700)),
            "seed": int(llm_config.get("seed", 42)),
        }

    def describe(self) -> str:
        return (f"Ollama {self.model} @ {self.base_url} "
                f"(num_ctx={self.options['num_ctx']}, temperature={self.options['temperature']})")

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> LLMReply:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools

        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            hint = f" Модель не загружена: выполните `ollama pull {self.model}`." if exc.code == 404 else ""
            raise LLMError(f"Ollama вернула HTTP {exc.code}: {detail}.{hint}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMError(f"Ollama не ответила за {self.timeout} с (таймаут). На CPU первый запрос "
                           f"включает загрузку модели в память — повторите или увеличьте timeout_seconds.") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"нет соединения с Ollama по адресу {self.base_url} ({exc.reason})") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError(f"Ollama вернула некорректный JSON: {exc}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        message = body.get("message") or {}
        content = message.get("content") or ""
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__raw__": args}
            tool_calls.append({"name": fn.get("name", ""), "arguments": args if isinstance(args, dict) else {}})

        from_text = False
        if not tool_calls and tools:
            tool_calls = _parse_text_tool_calls(content)
            from_text = bool(tool_calls)
            if from_text:
                content = ""

        usage = {
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
            "total_duration_ms": int(body["total_duration"] / 1e6) if body.get("total_duration") else None,
            "load_duration_ms": int(body["load_duration"] / 1e6) if body.get("load_duration") else None,
        }
        return LLMReply(content=content.strip(), tool_calls=tool_calls, usage=usage,
                        latency_ms=latency_ms, raw_tool_call_in_text=from_text)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

_FOLLOWUP = re.compile(r"(?i)^\s*(а|и)\s|\b(там|этом|этого|этой|него|неё|нем|них|такое же)\b")
_SMALLTALK = re.compile(r"(?i)^\s*(привет|здравствуй|здравствуйте|добрый (день|вечер|утро)|спасибо|благодарю)\b[\s!.,]*$")
_LIST_DOCS = re.compile(r"(?i)(какие|какой список|перечисли|что за)\s+(документ|материал|тем)|что есть в базе|список документов")
_TOP_K_IN_TEXT = re.compile(r"(?i)\b(top[\s_\-]?k\s*[:=]?\s*\S+|топ[\s\-]?\S+)")
_FRAGMENT_HEADER = re.compile(r"\[Фрагмент (\d+)\]\n")


def _parse_fragments(tool_text: str) -> List[tuple]:
    """Разбирает вывод format_fragments(): [(номер, текст), ...]."""
    parts = _FRAGMENT_HEADER.split(tool_text)
    result = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]
        marker = body.find("Текст: ")
        if marker >= 0:
            result.append((int(parts[i]), body[marker + len("Текст: "):].strip()))
    return result


def _stems(text: str) -> set:
    return set(_stem_list(text))


_BULLET = re.compile(r"^\s*([•\-–]|\d+[.)])\s")


def _units(text: str) -> List[str]:
    """
    Смысловые единицы фрагмента: строки, разорванные вёрсткой PDF, склеиваются,
    остальное режется на предложения. Пункты списков остаются отдельными единицами.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    merged: List[str] = []
    for line in lines:
        if merged and not _BULLET.match(line) and not re.search(r"[.!?;:]$", merged[-1]):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    units: List[str] = []
    for line in merged:
        if line.endswith(":") or _BULLET.match(line):
            units.append(line)
        else:
            units.extend(p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip())
    return units


class MockLLM:
    """
    Детерминированная «модель» для тестов и воспроизводимых прогонов.

    Шаг планирования: приветствие -> ответ без tools; «какие документы есть» -> list_documents;
    иначе -> search_knowledge_base(query=вопрос [+ тема из истории, если вопрос уточняющий]).
    Шаг ответа: выбирает из фрагментов предложения с наибольшим пересечением основ слов
    с вопросом; если совпадение ниже порога — возвращает NOT_FOUND_ANSWER.
    """

    name = "mock"

    def __init__(self, min_overlap: float = 0.4, min_overlap_abs: int = 2, max_sentences: int = 3):
        self.min_overlap = min_overlap
        self.min_overlap_abs = min_overlap_abs
        self.max_sentences = max_sentences

    def describe(self) -> str:
        return "MockLLM (детерминированная, экстрактивная)"

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> LLMReply:
        last = messages[-1]
        users = [m["content"] for m in messages if m["role"] == "user"]
        question = users[-1] if users else ""

        if last["role"] == "user":
            return self._plan(question, users[:-1], tools)
        if last["role"] == "tool":
            return self._answer(question, users[:-1], messages)
        return LLMReply(content=NOT_FOUND_ANSWER)

    # ----- планирование -----

    def _plan(self, question: str, history: List[str], tools) -> LLMReply:
        if _SMALLTALK.match(question):
            return LLMReply(content=SMALLTALK_ANSWER)
        if not tools:
            return LLMReply(content=NOT_FOUND_ANSWER)
        if _LIST_DOCS.search(question):
            return LLMReply(content="", tool_calls=[{"name": "list_documents", "arguments": {}}])
        query = _TOP_K_IN_TEXT.sub(" ", question)
        query = re.sub(r"\s+", " ", query).strip(" ,.;")
        if history and _FOLLOWUP.search(question):
            query = f"{history[-1]} {query}"
        return LLMReply(content="", tool_calls=[
            {"name": "search_knowledge_base", "arguments": {"query": query, "top_k": 4}}
        ])

    # ----- ответ по результатам tool -----

    def _answer(self, question: str, history: List[str], messages: List[Dict[str, Any]]) -> LLMReply:
        tool_texts = [m["content"] for m in messages if m["role"] == "tool"]
        last_tool = tool_texts[-1] if tool_texts else ""
        if last_tool.startswith("ОШИБКА"):
            return LLMReply(content="Поиск завершился ошибкой, ответ сформировать нельзя.")
        if last_tool.startswith("В базе знаний документов"):
            names = [line[2:].split(" | ")[0] for line in last_tool.splitlines() if line.startswith("- ")]
            return LLMReply(content="В базе знаний есть документы: " + "; ".join(names) + ".")

        topic = question
        if history and _FOLLOWUP.search(question):
            topic = f"{history[-1]} {question}"
        q_stems = _stems(_TOP_K_IN_TEXT.sub(" ", topic))
        if not q_stems:
            return LLMReply(content=NOT_FOUND_ANSWER)
        # порог: доля основ вопроса, но не больше их числа (для «Что такое SIPS?» достаточно 1)
        threshold = min(len(q_stems), max(self.min_overlap_abs, math.ceil(len(q_stems) * self.min_overlap)))

        candidates = []   # (overlap, номер фрагмента, индекс единицы)
        fragments: Dict[int, List[str]] = {}
        for text in tool_texts:
            for number, body in _parse_fragments(text):
                units = _units(body)
                fragments[number] = units
                for idx, unit in enumerate(units):
                    overlap = len(q_stems & _stems(unit))
                    if overlap >= threshold:
                        candidates.append((overlap, number, idx))
        if not candidates:
            return LLMReply(content=NOT_FOUND_ANSWER)
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        picked, seen = [], set()
        for _, number, idx in candidates:
            if len(picked) >= self.max_sentences:
                break
            if (number, idx) in seen:
                continue
            units = fragments[number]
            block = [units[idx]]
            seen.add((number, idx))
            if units[idx].endswith(":"):          # заголовок списка — берём его пункты
                j = idx + 1
                while j < len(units) and _BULLET.match(units[j]):
                    block.append(units[j])
                    seen.add((number, j))
                    j += 1
            picked.append((number, block))

        lines = []
        for number, block in picked:
            lines.append("\n".join(block) + f" [Фрагмент {number}]")
        cited = sorted({n for n, _ in picked})
        answer = "По данным базы знаний:\n" + "\n".join(lines)
        answer += "\nИсточники: [" + ", ".join(f"Фрагмент {n}" for n in cited) + "]"
        return LLMReply(content=answer)


def create_backend(llm_config: Dict[str, Any], backend: Optional[str] = None):
    kind = str(backend or llm_config.get("backend", "ollama")).lower()
    if kind == "mock":
        return MockLLM(**(llm_config.get("mock") or {}))
    if kind == "ollama":
        return OllamaChatBackend(llm_config)
    raise ValueError(f"Неизвестный llm.backend='{kind}'. Допустимо: ollama, mock.")
