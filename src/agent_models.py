"""
Модели данных агента: состояние запуска, trace, вызовы tools, итоговый ответ.

Всё — обычные dataclass, сериализуются в JSON через to_dict(): trace и ответ
сохраняются в отчёт сценариев и в data/agent/traces/.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Итоговые статусы запуска агента
STATUS_ANSWERED = "answered"      # ответ сформирован по найденным фрагментам
STATUS_NOT_FOUND = "not_found"    # поиск выполнен, ответа в документах нет
STATUS_REJECTED = "rejected"      # запрос отклонён guardrails до обращения к LLM и tools
STATUS_ERROR = "error"            # tool завершился ошибкой / некорректные параметры
STATUS_DEGRADED = "degraded"      # LLM недоступна: показаны найденные фрагменты без генерации
STATUS_SMALLTALK = "smalltalk"    # приветствие / вопрос вне базы, поиск не нужен

ALL_STATUSES = (STATUS_ANSWERED, STATUS_NOT_FOUND, STATUS_REJECTED,
                STATUS_ERROR, STATUS_DEGRADED, STATUS_SMALLTALK)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    iteration: int
    status: str = "pending"             # success | error
    error_code: Optional[str] = None
    error: Optional[str] = None
    result_summary: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    source: str = "llm"                 # llm | forced | degraded — кто инициировал вызов

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Fragment:
    """Фрагмент, реально возвращённый tool: на него (и только на него) можно ссылаться."""
    number: int
    chunk_id: Optional[str]
    score: float
    document_name: str
    file_name: str
    category: str
    chunk_index: Any
    text: str
    card: Dict[str, Any] = field(default_factory=dict)   # метаданные карточки реестра, если это карточка

    def source_dict(self) -> Dict[str, Any]:
        data = {
            "fragment": self.number,
            "document_name": self.document_name,
            "file_name": self.file_name,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }
        if self.card:
            data["card"] = dict(self.card)
        return data


@dataclass
class TraceEvent:
    ts: float
    type: str       # state | decision | llm_call | tool_call | tool_result | tool_error | memory | validation | final
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"t_ms": int(self.ts * 1000), "type": self.type, "data": self.data}


class Trace:
    """Хронология одного запуска агента; время — от старта запуска, в мс."""

    def __init__(self):
        self._t0 = time.monotonic()
        self.events: List[TraceEvent] = []

    def add(self, event_type: str, **data: Any) -> None:
        self.events.append(TraceEvent(time.monotonic() - self._t0, event_type, data))

    def decisions(self) -> List[Dict[str, Any]]:
        return [e.data for e in self.events if e.type == "decision"]

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]


@dataclass
class AgentState:
    """Состояние одного запуска; меняется по ходу цепочки и целиком попадает в trace."""
    query: str
    locked_params: Dict[str, Any] = field(default_factory=dict)
    route: Optional[str] = None
    iteration: int = 0
    llm_calls: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    fragments: List[Fragment] = field(default_factory=list)
    documents_listed: bool = False
    documents_result: Optional[Dict[str, Any]] = None
    cards_result: Optional[Dict[str, Any]] = None        # результат list_cards (перечисление по каталогу)
    followup_terms: List[str] = field(default_factory=list)   # термины темы из памяти для уточняющего вопроса
    followup_cards: List[Dict[str, str]] = field(default_factory=list)   # карточки прошлого ответа для «эти проекты»
    last_tool_error: Optional[ToolCall] = None
    llm_error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def successful_calls(self, name: Optional[str] = None) -> List[ToolCall]:
        return [c for c in self.tool_calls
                if c.status == "success" and (name is None or c.name == name)]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "iteration": self.iteration,
            "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_calls),
            "fragments": len(self.fragments),
            "locked_params": dict(self.locked_params),
            "followup_terms": list(self.followup_terms),
            "followup_cards": len(self.followup_cards),
            "llm_error": self.llm_error,
        }


@dataclass
class AgentResponse:
    status: str
    answer: str
    route: Optional[str]
    sources: List[Dict[str, Any]]
    tool_calls: List[ToolCall]
    llm_calls: int
    warnings: List[str]
    error_code: Optional[str]
    trace: Trace
    memory_size: int = 0
    duration_ms: int = 0

    @property
    def tool_names(self) -> List[str]:
        return [c.name for c in self.tool_calls]

    def to_dict(self, include_trace: bool = True) -> Dict[str, Any]:
        data = {
            "status": self.status,
            "answer": self.answer,
            "route": self.route,
            "sources": self.sources,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "llm_calls": self.llm_calls,
            "warnings": self.warnings,
            "error_code": self.error_code,
            "memory_size": self.memory_size,
            "duration_ms": self.duration_ms,
        }
        if include_trace:
            data["trace"] = self.trace.to_list()
        return data
