"""
HTTP API сервиса: POST /ask и GET /health (требования F1, F2, F7, S4, S5).

Один агент на сервис (локальный Qdrant открывается одним клиентом на процесс), у каждой сессии
своя память; запросы выполняются по одному. Сервис однопользовательский — ограничение пилота.

Запуск: python src/serve.py  (или uvicorn с фабрикой create_app).
"""

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from service.health import health
from service.schemas import AskRequest, AskResponse, HealthResponse
from service.sessions import SessionStore

VERSION = "1.0.0"
ROUTES = {"catalog_list": "registry", "catalog_people": "registry", "catalog_people_rank": "registry",
          "tools": "documents", "forced_search": "documents", "degraded": "documents",
          "guardrail_block": "rejected", "smalltalk": "smalltalk", "escalation": "escalation"}


def error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "request_id": request_id}})


def to_source(s: Dict[str, Any]) -> Dict[str, Any]:
    card = s.get("card") or {}
    if card:
        return {"type": "card", "id": f"{card.get('card_type')}:{card.get('card_id')}",
                "title": card.get("card_title", ""), "file": s.get("file_name") or None,
                "year": card.get("year"), "industry": card.get("industry")}
    score = s.get("score")
    return {"type": "document", "id": f"{s.get('file_name')}#{s.get('chunk_index')}",
            "title": s.get("document_name", ""), "file": s.get("file_name"), "fragment": s.get("fragment"),
            "score": round(float(score), 4) if isinstance(score, (int, float)) else None}


def warning_code(text: str) -> str:
    low = text.lower()
    for marker, code in (("аргументы вне схемы", "argument_dropped"), ("по всей базе", "document_filter_fallback"),
                         ("обрезан", "query_truncated"), ("лимит", "limit_reached"), ("ссылк", "citations")):
        if marker in low:
            return code
    return "warning"


def create_app(agent: Any = None, agent_factory: Optional[Callable[[], Any]] = None,
               memory_factory: Optional[Callable[[], Any]] = None, service_config: Optional[Dict[str, Any]] = None,
               clock: Callable[[], float] = time.monotonic) -> FastAPI:
    """
    agent — готовый агент (тесты); agent_factory — создать агента при старте (сервис).
    Если агента создать не удалось, сервис поднимается и отвечает 503 service_unavailable.
    """
    cfg = service_config or {}
    app = FastAPI(title="KB Search Agent", version=VERSION,
                  description="Поиск по базе знаний: реестры проектов и пропозалов, документы продуктов и КП.")
    state: Dict[str, Any] = {"agent": agent, "startup_error": None}
    if agent is None and agent_factory is not None:
        try:
            state["agent"] = agent_factory()
        except Exception as exc:                            # сервис поднимается, но честно говорит, что не готов
            state["startup_error"] = type(exc).__name__
    lock = threading.Lock()

    def default_memory():
        from agent_memory import ShortTermMemory
        mem = (state["agent"].config.get("memory", {}) if state["agent"] else {}) or {}
        return ShortTermMemory(int(mem.get("max_messages", 6)), int(mem.get("max_chars_per_message", 500)))

    sessions = SessionStore(memory_factory or default_memory, ttl_seconds=float(cfg.get("session_ttl_minutes", 30)) * 60,
                            max_sessions=int(cfg.get("max_sessions", 200)), clock=clock)
    app.state.sessions = sessions

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        try:
            return await call_next(request)
        except Exception:                                   # S4: без traceback, путей и конфигурации
            return error(500, "internal_error", "Внутренняя ошибка сервиса. Подробности — в журнале по request_id.",
                         request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        problems = []
        for e in exc.errors():
            field = ".".join(str(p) for p in e.get("loc", []) if p != "body") or "body"
            message = str(e.get("msg", "")).replace("Value error, ", "")
            problems.append(f"{field}: {message}")
        return error(422, "invalid_request", "; ".join(problems) or "некорректный запрос",
                     getattr(request.state, "request_id", str(uuid.uuid4())))

    @app.post("/ask", response_model=AskResponse, responses={422: {}, 500: {}, 503: {}})
    def ask(body: AskRequest, request: Request):
        rid = request.state.request_id
        agent_ = state["agent"]
        if agent_ is None:
            return error(503, "service_unavailable", "Сервис не готов: агент не инициализирован.", rid)
        session_id, session, _ = sessions.get_or_create(body.session_id)
        with lock:                                          # один агент — запросы по одному
            agent_.memory = session.memory
            response = agent_.run(body.question)
        esc = response.escalation
        return {
            "request_id": rid, "session_id": session_id, "status": response.status,
            "route": "escalation" if response.status == "escalated" else ROUTES.get(response.route, response.route),
            "route_path": list(getattr(response, "route_path", []) or []),
            "answer": response.answer, "sources": [to_source(s) for s in response.sources or []],
            "escalation": dict(esc) if esc else None,
            "warnings": [{"code": warning_code(w), "message": w} for w in response.warnings or []],
            "grounding": response.grounding, "error_code": response.error_code,
            "duration_ms": response.duration_ms, "llm_calls": response.llm_calls,
        }

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    def get_health(request: Request):
        agent_ = state["agent"]
        if agent_ is None:
            body = {"status": "down", "version": VERSION, "profile": cfg.get("profile", "?"),
                    "components": {"agent": {"status": "down", "detail": f"не инициализирован ({state['startup_error']})"}}}
            return JSONResponse(status_code=503, content=body)
        code, body = health(agent_)
        body = {**body, "version": VERSION, "profile": str(agent_.config.get("profile", "?"))}
        return JSONResponse(status_code=code, content=HealthResponse(**body).model_dump(exclude_none=True))

    return app
