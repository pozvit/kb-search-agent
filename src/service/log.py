"""
Структурированный журнал сервиса (требования S2, S3; docs/architecture.md, раздел 5).

Одна строка JSON на событие: ts, level, event, request_id и поля события. Поля задаются белым
списком для каждого события (ALLOWED): всё, чего нет в списке, отбрасывается. Поэтому в журнал
не попадают, даже если появятся в trace агента:
  - текст вопроса и ответа, поисковые запросы, аргументы инструментов;
  - имена файлов и названия источников (в полной базе в именах КП — наименования клиентов);
  - фамилии, тексты утверждений, сообщения исключений (в них бывают данные и пути).
Идентификатор сессии пишется хешем.

События строятся из trace агента после выполнения запроса: логика агента от журнала не зависит.
"""

import datetime
import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Iterator, Optional, TextIO, Tuple

LOGGER_NAME = "kb_agent"

ALLOWED: Dict[str, set] = {
    "service_started": {"profile", "version", "agent_ready", "startup_error"},
    "request_received": {"question_len", "has_session", "session_hash", "new_session"},
    "invalid_request": {"fields"},
    "guardrail_decision": {"allowed", "reason", "secret_kinds"},
    "route_decided": {"route", "rule", "kind", "classifier"},
    "tool_call": {"name", "source", "status", "error_code", "latency_ms", "count", "ticket", "queued"},
    "llm_call": {"stage", "ok", "latency_ms", "prompt_tokens", "completion_tokens"},
    "grounding_check": {"grounded", "units", "supported", "regenerations"},
    "escalation_created": {"ticket", "reason", "owner"},
    "response_sent": {"status", "route", "route_path", "sources_count", "duration_ms", "llm_calls",
                      "warnings_count", "error_code"},
    "error": {"code", "exception", "where"},
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
                   "level": record.levelname, "event": getattr(record, "event", record.getMessage()),
                   "request_id": getattr(record, "request_id", None), **getattr(record, "fields", {})}
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in getattr(record, "fields", {}).items())
        return f"{record.levelname:5} {getattr(record, 'event', record.getMessage())} " \
               f"[{(getattr(record, 'request_id', None) or '-')[:8]}] {fields}"


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None,
                      stream: Optional[TextIO] = None) -> logging.Logger:
    """LOG_LEVEL и LOG_FORMAT (json | text) перекрывают значения из конфигурации."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel((os.environ.get("LOG_LEVEL") or level or "INFO").upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(TextFormatter() if (os.environ.get("LOG_FORMAT") or fmt or "json") == "text"
                         else JsonFormatter())
    logger.addHandler(handler)
    return logger


def session_hash(session_id: Optional[str]) -> Optional[str]:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12] if session_id else None


def log_event(event: str, request_id: Optional[str], level: int = logging.INFO, **fields: Any) -> None:
    allowed = ALLOWED.get(event, set())
    clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"event": event, "request_id": request_id,
                                                             "fields": clean})


def events_from_trace(trace: Iterable[Dict[str, Any]]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """События журнала из trace агента. Берутся только поля из белого списка (ALLOWED)."""
    last_source: Dict[str, str] = {}
    for item in trace:
        kind, data = item.get("type"), item.get("data", {})
        point = data.get("point")
        if kind == "decision" and point == "guardrails":
            yield "guardrail_decision", {"allowed": data.get("allowed"), "reason": data.get("reason"),
                                         "secret_kinds": data.get("secret_kinds") or None}
        elif kind == "decision" and point == "route_decided":
            yield "route_decided", {"route": data.get("route"), "rule": data.get("rule"), "kind": data.get("kind"),
                                    "classifier": data.get("classifier")}
        elif kind == "tool_call":
            last_source[data.get("name")] = data.get("source")
        elif kind == "tool_result":
            yield "tool_call", {"name": data.get("name"), "source": last_source.get(data.get("name")),
                                "status": data.get("status", "success"), "latency_ms": data.get("latency_ms"),
                                "count": data.get("count", data.get("total")),          # list_cards: total
                                "ticket": data.get("ticket"), "queued": data.get("queued")}
        elif kind == "tool_error":
            yield "tool_call", {"name": data.get("name"), "source": last_source.get(data.get("name")), "status": "error",
                                "error_code": data.get("error_code"), "latency_ms": data.get("latency_ms")}
        elif kind == "llm_call":
            usage = data.get("usage") or {}
            yield "llm_call", {"stage": data.get("stage"), "ok": data.get("ok"), "latency_ms": data.get("latency_ms"),
                               "prompt_tokens": usage.get("prompt_tokens"),
                               "completion_tokens": usage.get("completion_tokens")}
        elif kind == "validation" and data.get("kind") == "grounding":
            units = data.get("units") or 0
            yield "grounding_check", {"grounded": data.get("grounded"), "units": units,
                                      "supported": units - len(data.get("unsupported") or []),
                                      "regenerations": data.get("regenerations")}
        elif kind == "decision" and point == "escalation":
            yield "escalation_created", {"ticket": data.get("ticket"), "reason": data.get("reason"),
                                         "owner": data.get("owner")}
