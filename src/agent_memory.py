"""
Краткосрочная память диалога (в пределах одной сессии CLI / одного сценария).

Хранит последние max_messages реплик (user/assistant), каждая обрезается до
max_chars_per_message. Нужна для уточняющих вопросов: «Что такое SIPS?» ->
«А какой там был объём электрификации?» — LLM восстанавливает тему из истории
и включает её в query для search_knowledge_base.

Долговременная память не реализована сознательно: агент отвечает по базе знаний,
а не по истории пользователя; хранение диалогов между сессиями только добавило бы
риск утечки данных из полной базы (ФИО, клиенты). См. README, раздел «Память».
"""

from collections import deque
from typing import Deque, Dict, List


class ShortTermMemory:
    def __init__(self, max_messages: int = 6, max_chars_per_message: int = 500):
        if max_messages < 0:
            raise ValueError("max_messages не может быть отрицательным")
        self.max_messages = max_messages
        self.max_chars = max_chars_per_message
        self._messages: Deque[Dict[str, str]] = deque(maxlen=max_messages or None)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        if self.max_messages == 0:
            return
        self._messages.append({"role": "user", "content": self._cut(user_text)})
        self._messages.append({"role": "assistant", "content": self._cut(assistant_text)})

    def messages(self) -> List[Dict[str, str]]:
        return [dict(m) for m in self._messages]

    def last_user_message(self) -> str:
        for message in reversed(self._messages):
            if message["role"] == "user":
                return message["content"]
        return ""

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def _cut(self, text: str) -> str:
        text = (text or "").strip()
        return text if len(text) <= self.max_chars else text[: self.max_chars] + "…"
