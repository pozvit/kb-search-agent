"""
Тесты проверки привязки ответа (шаг 2.4.3): модуль agents.grounding и его использование агентом.
Синтетические данные.
"""

import pytest

from conftest import ScriptedLLM, reply, search_call

from agents.grounding import check_grounding

FR = {1: "Цифровое ядро ORBIT: цифровой журнал дефектов, цифровой прогноз отказов, цифровой склад запчастей.",
      2: "Эффекты ORBIT на химическом заводе: сокращение простоев на 18%, срок окупаемости 9 месяцев, 5 цехов."}


@pytest.mark.parametrize("answer", [
    "Цифровое ядро ORBIT включает журнал дефектов, прогноз отказов и склад запчастей [Фрагмент 1].",
    "Цифровое ядро ORBIT включает:\n1. цифровой журнал дефектов;\n2. цифровой прогноз отказов [Фрагмент 1].",
    "Цифровой журнал дефектов и цифровой прогноз отказов.\n\nИсточники: [Фрагмент 1]",
    "Простои сократились на 18% за 9 месяцев в 5 цехах [Фрагмент 2].",
    "Состав:\n1. Журнал [Фрагмент 1]\n2. Склад [Фрагмент 1]",
], ids=["inline", "list_with_header", "sources_line", "all_numbers", "short_items"])
def test_grounded(answer):
    assert check_grounding(answer, FR).grounded


@pytest.mark.parametrize("answer, why", [
    ("Цифровое ядро ORBIT включает журнал дефектов и прогноз отказов.", "нет ссылки"),
    ("Цифровое ядро ORBIT включает журнал дефектов и прогноз отказов [Фрагмент 7].", "нет ссылки"),
    ("Компания открыла завод в Казани и наняла сотрудников [Фрагмент 1].", "нет в процитированном"),
    ("Простои сократились на 18% в 7 цехах [Фрагмент 2].", "чисел 7"),
    ("Простои на химическом заводе сократились на 18% [Фрагмент 1].", "нет в процитированном"),
], ids=["no_citation", "unknown_fragment", "invented", "single_digit_number", "wrong_fragment"])
def test_not_grounded(answer, why):
    result = check_grounding(answer, FR)
    assert not result.grounded and why in result.unsupported[0]["why"]


def test_list_numbering_is_not_a_fact():
    assert check_grounding("1. Простои сократились на 18% [Фрагмент 2].\n2. Окупаемость 9 месяцев [Фрагмент 2].", FR).grounded


# ---------------------------------------------------------------- интеграция с агентом

def test_not_found_after_regeneration_is_escalation(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("ORBIT внедрён на сорока заводах в Европе и Азии."),
                       reply("Недостаточно информации в предоставленных документах.")])
    r = make_agent(llm=llm).run("Где внедрён ORBIT?")
    assert r.status == "escalated" and r.escalation["reason"] == "ungrounded_answer"


def test_honest_not_found_stays_not_found(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("Недостаточно информации в предоставленных документах.")])
    r = make_agent(llm=llm).run("Где внедрён ORBIT?")
    assert r.status == "not_found" and r.llm_calls == 2


def test_model_down_during_regeneration_is_escalation(make_agent):
    from llm_client import LLMError
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("ORBIT — платформа планирования ремонтов оборудования."),
                       LLMError("нет соединения")])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert r.status == "escalated" and r.escalation["reason"] == "ungrounded_answer"


def test_registry_and_document_list_answers_are_not_checked(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[{"name": "list_documents", "arguments": {}}]),
                       reply("В базе два документа про ORBIT и LUMEN.")])
    r = make_agent(llm=llm).run("Какие документы есть в базе знаний?")
    assert r.status == "answered" and r.grounding is None


def test_escalated_answer_is_not_remembered(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("ORBIT внедрён на сорока заводах."), reply("ORBIT внедрён на сорока заводах.")])
    agent = make_agent(llm=llm)
    agent.run("Где внедрён ORBIT?")
    assert len(agent.memory) == 0
