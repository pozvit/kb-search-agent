"""
Схемы HTTP API (pydantic). Формат описан в docs/architecture.md, раздел 3.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

SESSION_ID_PATTERN = r"^[A-Za-z0-9-]{1,64}$"


class AskRequest(BaseModel):
    question: str = Field(..., description="Вопрос на русском языке, 1–1000 символов после обрезки пробелов.")
    session_id: Optional[str] = Field(None, pattern=SESSION_ID_PATTERN,
                                      description="Идентификатор сессии для уточняющих вопросов; если не передан — создаётся.")

    model_config = {"extra": "forbid", "str_strip_whitespace": False}

    @field_validator("question", mode="before")
    @classmethod
    def question_is_string(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("question должен быть строкой")
        return value

    @field_validator("question")
    @classmethod
    def question_length(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question не должен быть пустым")
        if len(stripped) > 1000:
            raise ValueError("question длиннее 1000 символов")
        return stripped


class Source(BaseModel):
    type: str                                   # card | document
    id: str
    title: str = ""
    file: Optional[str] = None
    fragment: Optional[int] = None
    score: Optional[float] = None
    year: Optional[int] = None
    industry: Optional[List[str]] = None


class Escalation(BaseModel):
    id: Optional[str]
    reason: str
    owner: str
    queued: bool


class Warning_(BaseModel):
    code: str
    message: str


class AskResponse(BaseModel):
    request_id: str
    session_id: str
    status: str                                 # answered | not_found | escalated | smalltalk | rejected | degraded | error
    route: Optional[str]                        # registry | documents | escalation | smalltalk | rejected
    route_path: List[str] = []
    answer: str
    sources: List[Source] = []
    escalation: Optional[Escalation] = None
    warnings: List[Warning_] = []
    grounding: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    duration_ms: int
    llm_calls: int


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Component(BaseModel):
    status: str                                 # ok | down | absent
    detail: Optional[str] = None
    model: Optional[str] = None
    points: Optional[int] = None
    cards: Optional[int] = None


class HealthResponse(BaseModel):
    status: str                                 # ok | degraded | down
    version: str
    profile: str
    components: Dict[str, Component]
