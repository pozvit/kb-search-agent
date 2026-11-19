"""
KB Search Agent — точка входа.

Вся логика в пакете agents/:
  orchestrator.py    порядок шагов: guardrails → память → маршрутизатор → подагент → ответ
  router.py          выбор подагента по правилам; классификатор для неоднозначных вопросов
  registry_agent.py  подагент «Реестры»: перечисления, люди, рейтинги по каталогу карточек
  documents_agent.py подагент «Документы»: поиск по документам, ответ модели, проверка привязки
  grounding.py       проверка привязки ответа к источникам
  escalation.py      эскалация к человеку: очередь обращений, правила E1–E4

KBAgent сохранён как имя для CLI, сценариев, инструмента замера и тестов.
"""

from agents.orchestrator import Orchestrator


class KBAgent(Orchestrator):
    """Агент поиска по базе знаний. Конструктор: KBAgent(config, llm=None, retriever_factory=None)."""
