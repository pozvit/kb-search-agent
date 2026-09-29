"""
Guardrails — проверки ДО обращения к LLM и tools.

1. Тип и длина запроса (пустой, не строка, длиннее max_query_chars).
2. Секреты в запросе: API-ключи, токены, приватные ключи, пароли, строки подключения,
   номера банковских карт (с проверкой Луна). Такой запрос не уходит в LLM,
   не пишется в память, а в trace попадает только в замаскированном виде.
3. Извлечение параметров, явно заданных пользователем (top_k=N, «топ-N»).
   Они «фиксируются»: агент подставляет их в вызов tool поверх значения от LLM,
   чтобы модель не могла молча исправить некорректное значение пользователя.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SECRET_PATTERNS = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("yandex_api_key", re.compile(r"\bAQVN[A-Za-z0-9_\-]{20,}")),
    ("yandex_iam_token", re.compile(r"\bt1\.[A-Za-z0-9_\-.]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{20,}")),
    ("connection_string", re.compile(r"(?i)\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/]+:[^\s@]+@")),
    ("password_assignment", re.compile(r"(?i)\b(password|passwd|pwd|пароль|api[_-]?key|secret|token)\s*[:=]\s*\S{4,}")),
]

CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")

TOP_K_PATTERNS = [
    re.compile(r"(?i)\btop[\s_\-]?k\s*[:=]?\s*([^\s,;.?!]+)"),
    # «топ-5», «топ 3»; только число или числительное — иначе «топливо»/«топ-менеджмент» дают ложное срабатывание
    re.compile(r"(?i)(?<![а-яё])топ[\s\-]?(-?\d+|ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)"
               r"(?![а-яё\d])"),
]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_secrets(text: str) -> List[str]:
    """Возвращает список типов найденных секретов (без самих значений)."""
    kinds = [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]
    for match in CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            kinds.append("payment_card")
            break
    return kinds


def redact(text: str) -> str:
    """Маскирует секреты для trace; сам секрет никуда не сохраняется."""
    result = text
    for name, pattern in SECRET_PATTERNS:
        result = pattern.sub(f"[REDACTED:{name}]", result)

    def _card(match: "re.Match") -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return f"[REDACTED:card ****{digits[-4:]}]"
        return match.group(0)

    return CARD_CANDIDATE.sub(_card, result)


def extract_locked_params(text: str) -> Dict[str, Any]:
    """
    Параметры, явно заданные пользователем. Значение сохраняется «как есть» (строкой),
    приведение типа и проверку диапазона делает tool — так ошибка типа
    («top_k=пять») и ошибка диапазона («top_k=0») обрабатываются одинаково.
    """
    for pattern in TOP_K_PATTERNS:
        match = pattern.search(text)
        if match:
            return {"top_k": match.group(1).strip()}
    return {}


@dataclass
class GuardrailResult:
    allowed: bool
    reason_code: Optional[str] = None
    message: Optional[str] = None
    secret_kinds: List[str] = field(default_factory=list)
    safe_query: str = ""                       # версия запроса, допустимая для trace


def check_query(query: Any, max_query_chars: int = 1000, block_secrets: bool = True) -> GuardrailResult:
    if not isinstance(query, str):
        return GuardrailResult(False, "invalid_type",
                               f"Запрос должен быть строкой, получено {type(query).__name__}.",
                               safe_query=repr(type(query).__name__))

    stripped = query.strip()
    if not stripped:
        return GuardrailResult(False, "empty_query",
                               "Пустой запрос. Сформулируйте вопрос по базе знаний, например: "
                               "«Какие решения входят в цифровое ядро AIMS?»", safe_query="")

    if len(stripped) > max_query_chars:
        return GuardrailResult(False, "query_too_long",
                               f"Запрос слишком длинный: {len(stripped)} символов при лимите {max_query_chars}. "
                               f"Сократите вопрос до сути.",
                               safe_query=redact(stripped[:200]) + "…")

    if block_secrets:
        kinds = find_secrets(stripped)
        if kinds:
            return GuardrailResult(
                False, "secret_detected",
                "В запросе обнаружены секретные данные (" + ", ".join(sorted(set(kinds))) + "). "
                "Запрос не отправлен в модель и не сохранён. Удалите ключ/пароль/номер карты "
                "из текста; если секрет уже где-то опубликован — перевыпустите его.",
                secret_kinds=kinds, safe_query=redact(stripped))

    return GuardrailResult(True, safe_query=stripped)
