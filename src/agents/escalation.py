"""
Эскалация к человеку (требование F6, правила E1–E4).

Агент не выдаёт уверенный ответ и создаёт обращение к ответственному:
  E1 ungrounded_answer  — ответ не прошёл проверку привязки к источникам после перегенерации;
  E2 no_data            — вопрос о данных, которых в базе нет (стоимость, контакты, планы);
  E3 component_failure  — недоступен индекс или каталог;
  E4 user_request       — пользователь просит связаться с человеком.

Обращение — строка JSON в локальной очереди (по умолчанию data/escalations/escalations.jsonl,
в .gitignore). Ответственный — роль из конфигурации, а не человек: имена и контакты в репозиторий
не попадают. Уведомления не рассылаются: очередь разбирает ответственный.

Текст вопроса хранится только в очереди. Запросы с секретами сюда не доходят: их раньше
отклоняют guardrails.
"""

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

REASONS = ("ungrounded_answer", "no_data", "component_failure", "user_request")

# E2: вопросы о данных, которых в базе знаний не бывает
NO_DATA_INTENT = re.compile(
    r"(?i)(сколько\s+стоит|стоимост\w*|(?<![а-яё])цен[аыуе](?![а-яё])|прайс\w*|тариф\w*|бюджет\w*|"
    r"(?<![а-яё])руб\w*|контакт\w*|телефон\w*|e-?mail|электронн\w+\s+почт\w*|(?<![а-яё])адрес\w*|"
    r"зарплат\w*|окладом?|планы?\s+на|внутренн\w+\s+оценк\w*)")

# E4: просьба связаться с человеком
USER_REQUEST = re.compile(
    r"(?i)((соедин|свяж|переда|перенаправ|переключ|позов)\w*\s+(меня\s+)?(с\s+)?(\w+\s+){0,2}"
    r"(человек|ответствен|специалист|менеджер|оператор|сотрудник|эксперт)\w*|"
    r"(хочу|нужно|можно)\s+(поговорить|связаться)\s+с\s+(\w+\s+)?(человек|ответствен|специалист|менеджер)\w*|"
    r"живой\s+(человек|оператор))")

DEFAULT_OWNERS = {
    "ungrounded_answer": "владелец базы коммерческих предложений",
    "no_data": "владелец базы коммерческих предложений",
    "component_failure": "администратор сервиса",
    "user_request": "владелец реестра проектов",
}


class EscalationQueue:
    def __init__(self, path: str, owners: Optional[Dict[str, str]] = None, profile: str = "public"):
        self.path = Path(os.environ.get("ESCALATION_QUEUE_PATH") or path)
        self.owners = {**DEFAULT_OWNERS, **(owners or {})}
        self.profile = profile

    def owner_for(self, reason: str, topic: Optional[str] = None) -> str:
        if reason == "user_request" and topic == "documents":
            return self.owners.get("no_data", DEFAULT_OWNERS["no_data"])
        return self.owners.get(reason, DEFAULT_OWNERS.get(reason, "ответственный"))

    def _next_id(self, today: str) -> str:
        count = 0
        if self.path.exists():
            prefix = f'"id": "ESC-{today}-'
            with open(self.path, encoding="utf-8") as f:
                count = sum(1 for line in f if prefix in line)
        return f"ESC-{today}-{count + 1:04d}"

    def create(self, *, reason: str, question: str, route_path: List[str], evidence: List[Dict[str, Any]],
               request_id: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        if reason not in REASONS:
            raise ValueError(f"неизвестная причина эскалации: {reason}")
        now = datetime.datetime.now().astimezone()
        ticket = {
            "id": None, "created_at": now.isoformat(timespec="seconds"), "status": "new", "reason": reason,
            "owner": self.owner_for(reason, topic), "route_path": route_path, "question": question,
            "evidence": evidence[:10], "request_id": request_id, "profile": self.profile,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ticket["id"] = self._next_id(now.strftime("%Y%m%d"))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
            ticket["queued"] = True
        except OSError as exc:
            # обращение не записалось: пользователь всё равно получает ответ, а сбой виден в ответе и trace
            ticket["id"] = ticket["id"] or f"ESC-{now.strftime('%Y%m%d')}-не записано"
            ticket["queued"] = False
            ticket["error"] = type(exc).__name__
        return ticket
